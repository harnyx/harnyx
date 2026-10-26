"""Cloudflare Workers AI chat, decision and extraction transports."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig
from harnyx_commons.errors import (
    ProviderCredentialUnavailableError,
    ToolInvocationCancelledError,
    ToolInvocationTimeoutError,
    ToolProviderError,
)
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.cost_settlement import settled_response_cost
from harnyx_commons.llm.pricing import price_decision
from harnyx_commons.llm.provider import (
    LlmInvocationCancelledError,
    LlmProviderError,
    LlmRetryExhaustedError,
    RetryContext,
)
from harnyx_commons.llm.providers.openai_compatible import (
    OpenAiCompatibleLlmProvider,
    _OpenAiCompatibleChatRequest,
    _OpenAiCompatiblePromptTokenDetails,
    _OpenAiCompatibleUsageDetails,
    _OpenAiCompatibleUsagePayload,
)
from harnyx_commons.llm.providers.openai_stream import _OpenAiStreamEvent
from harnyx_commons.llm.providers.thinking import resolve_request_thinking
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.llm.timeout import LlmAttemptTimeoutError
from harnyx_commons.tools.ports import DecisionProviderResult
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata
from harnyx_miner_sdk.tools.cloudflare import CLOUDFLARE_CHAT_MODELS, CloudflareCredential
from harnyx_miner_sdk.tools.decision_models import (
    BooleanAnswer,
    ChoiceAnswer,
    DecisionContext,
    DecisionQueryRequest,
    DecisionQueryResponse,
    DecisionUsage,
    Probability,
    ScoreAnswer,
    validate_decision_answers,
)
from harnyx_miner_sdk.tools.extraction_models import ExtractionAnswer


def parse_cloudflare_credential(value: SecretStr | str) -> CloudflareCredential:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    try:
        return CloudflareCredential.model_validate_json(raw)
    except ValidationError:
        raise ValueError("Cloudflare requires a configured account ID and API token") from None


def configured_cloudflare_credential(*, account_id: str, api_token: SecretStr) -> CloudflareCredential:
    """A local credential is available only when both components are valid."""
    try:
        return CloudflareCredential.model_validate({"account_id": account_id, "api_token": api_token})
    except ValidationError:
        raise ProviderCredentialUnavailableError("cloudflare") from None


class _CloudflarePromptTokenDetails(_OpenAiCompatiblePromptTokenDetails):
    cached_tokens: int | None = Field(default=None, ge=0)


class _CloudflareCompletionTokenDetails(_OpenAiCompatibleUsageDetails):
    reasoning_tokens: int | None = Field(default=None, ge=0)


class _CloudflareChatUsage(_OpenAiCompatibleUsagePayload):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    prompt_tokens_details: _CloudflarePromptTokenDetails | None = None
    completion_tokens_details: _CloudflareCompletionTokenDetails | None = None

    @model_validator(mode="after")
    def _token_subsets(self) -> _CloudflareChatUsage:
        if self.prompt_tokens_details is not None:
            cached = self.prompt_tokens_details.cached_tokens
            if cached is not None and cached > self.prompt_tokens:
                raise ValueError("Cloudflare cached tokens must be a subset of prompt tokens")
        for reasoning in (
            self.reasoning_tokens,
            self.completion_tokens_details.reasoning_tokens if self.completion_tokens_details is not None else None,
        ):
            if reasoning is not None and reasoning > self.completion_tokens:
                raise ValueError("Cloudflare reasoning tokens must be a subset of completion tokens")
        return self


class CloudflareLlmProvider(OpenAiCompatibleLlmProvider):
    def __init__(self, *, credential: CloudflareCredential, client: httpx.AsyncClient | None = None) -> None:
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{credential.account_id}/ai/v1"
        endpoint = OpenAiCompatibleEndpointConfig.model_validate(
            {"id": "cloudflare", "base_url": base_url, "auth": {"type": "none"}}
        )
        transport = client or httpx.AsyncClient(timeout=300.0)
        transport.headers["Authorization"] = f"Bearer {credential.api_token.get_secret_value()}"
        super().__init__(endpoint=endpoint, client=transport, provider_label="cloudflare")
        self._owns_client = client is None

    def _parse_stream_usage(self, payload: JsonValue) -> _CloudflareChatUsage:
        return _CloudflareChatUsage.model_validate(payload)

    def _parse_stream_event(self, payload: JsonObject) -> _OpenAiStreamEvent:
        # Workers AI can encode numeric text as JSON numbers (cloudflare/ai#651).
        # Keep this wire-format adaptation local; all other fields remain strict.
        choices = payload.get("choices")
        if isinstance(choices, list):
            normalized = []
            for choice in choices:
                if isinstance(choice, dict) and isinstance(delta := choice.get("delta"), dict):
                    content = delta.get("content")
                    if type(content) is int or (type(content) is float and math.isfinite(content)):
                        choice = {**choice, "delta": {**delta, "content": str(content)}}
                normalized.append(choice)
            payload = {**payload, "choices": normalized}
        return super()._parse_stream_event(payload)

    def _build_request(self, request: AbstractLlmRequest) -> _OpenAiCompatibleChatRequest:
        payload = super()._build_request(request)
        thinking = resolve_request_thinking(
            request_thinking=request.thinking, reasoning_effort=request.reasoning_effort,
        )
        if thinking is None:
            return payload
        if thinking.budget is not None:
            raise ValueError("Cloudflare does not support thinking.budget; use thinking.effort")
        template_models = {
            "@cf/qwen/qwen3.8-27b", "@cf/zai-org/glm-5.2", "@cf/zai-org/glm-5.3-flash",
            "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
            "@cf/google/gemma-4-26b-a4b-it",
        }
        if request.model in template_models:
            if request.model == "@cf/zai-org/glm-5.3-flash" and not thinking.enabled:
                raise ValueError("Cloudflare GLM 5.3 Flash does not support disabling thinking")
            payload = payload.model_copy(update={
                "chat_template_kwargs": {**(payload.chat_template_kwargs or {}), "enable_thinking": thinking.enabled},
            })
            if not thinking.enabled or thinking.effort is None:
                return payload
        # Model-native effort levels and compatibility aliases documented by Workers AI.
        efforts = {
            "@cf/openai/gpt-oss-20b": {"low": "low", "medium": "medium", "high": "high"},
            "@cf/openai/gpt-oss-120b": {"low": "low", "medium": "medium", "high": "high"},
            "@cf/qwen/qwen3.8-27b": {"low": "low", "medium": "medium"},
            "@cf/zai-org/glm-5.2": {"low": "high", "medium": "high", "high": "high"},
            "@cf/zai-org/glm-5.3-flash": {"low": "low", "medium": "max", "high": "high"},
            "@cf/moonshotai/kimi-k2.6": {"low": "high", "medium": "high", "high": "high", "none": "none"},
            "@cf/deepseek-ai/deepseek-v4.1-flash-1017": {
                "low": "low", "medium": "high", "high": "high",
            },
        }.get(request.model)
        if efforts is None:
            raise ValueError(f"Cloudflare model {request.model!r} does not support thinking controls")
        effort = thinking.effort if thinking.enabled else "none"
        if effort is None:
            return payload  # Keep the model's documented enabled default.
        if effort not in efforts:
            raise ValueError(f"Cloudflare model {request.model!r} does not support thinking effort {effort!r}")
        return payload.model_copy(update={"reasoning_effort": efforts[effort]})

    def _build_accumulated_response(
        self, response: LlmResponse, ctx: RetryContext, processed: object | None,
    ) -> LlmResponse:
        accumulated = super()._build_accumulated_response(response, ctx, processed)
        metadata = dict(accumulated.metadata or {})
        # Both amounts use all observed token usage, including rejected attempts.
        metadata.pop("reference_cost_usd", None)
        if ctx.actual_cost_usd_total is not None:
            metadata["reference_cost_usd"] = ctx.actual_cost_usd_total
        return replace(accumulated, metadata=metadata)

    async def _annotate_response_cost(
        self, request: AbstractLlmRequest, response: LlmResponse,
    ) -> LlmResponse:
        response = await super()._annotate_response_cost(request, response)
        metadata = dict(response.metadata or {})
        if "actual_cost_usd" in metadata:
            metadata["reference_cost_usd"] = metadata["actual_cost_usd"]
        return replace(response, metadata=metadata)

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        if request.model not in CLOUDFLARE_CHAT_MODELS:
            raise ValueError(f"Cloudflare Chat Completions does not support model {request.model!r}")
        try:
            return await super()._invoke(replace(request, provider="cloudflare"))
        except (LlmProviderError, LlmRetryExhaustedError, LlmInvocationCancelledError) as exc:
            if exc.response is None:
                raise
            cost = (
                settled_response_cost(exc.response, provider="cloudflare", model=request.model)
                if exc.response.usage.prompt_tokens is not None and exc.response.usage.completion_tokens is not None
                else None
            )
            billing = None if cost is None else ProviderBillingMetadata(
                actual_cost_provider="cloudflare", actual_cost_usd=cost.cost_usd, source="response_body",
                reference_cost_usd=cost.cost_usd,
                llm_usage=exc.response.usage, provider_usage=cost.evidence,
            )
            if isinstance(exc, LlmInvocationCancelledError):
                raise ToolInvocationCancelledError("Cloudflare stream cancelled", billing=billing) from exc
            if isinstance(exc.__cause__, LlmAttemptTimeoutError):
                raise ToolInvocationTimeoutError(str(exc.__cause__), billing=billing) from exc
            raise ToolProviderError("Cloudflare chat stream failed", provider="cloudflare", billing=billing) from exc


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    success: bool
    result: JsonObject | None = None


class _Usage(DecisionUsage):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class _BillingProbe(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    usage: _Usage


class _NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


class _ChoiceAnswer(ChoiceAnswer):
    probabilities: dict[str, Probability]
    confidence: Probability


class _ScoreAnswer(ScoreAnswer):
    probabilities: dict[str, Probability]
    confidence: Probability
    legend: dict[str, DecisionContext]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    model: str
    answers: dict[
        str, Annotated[_NoulAnswer | _ChoiceAnswer | _ScoreAnswer | ExtractionAnswer, Field(discriminator="type")]
    ]
    usage: _Usage
    truncated: bool | None = None


class CloudflareDecisionProvider:
    def __init__(
        self, *, credential: CloudflareCredential, timeout_seconds: float = 300,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._credential = credential
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def query(self, request: DecisionQueryRequest) -> DecisionProviderResult:
        if request.provider != "cloudflare":
            raise ValueError("Cloudflare decision adapter requires cloudflare")
        extraction = request.model == "@cf/cloudflare/clairic"
        questions = {
            key: question.model_dump(mode="json", exclude_none=True)
            for key, question in request.questions.items()
        }
        if not extraction:
            for question in questions.values():
                if question["type"] == "boolean":
                    question["type"] = "noul"
        state = request.state
        if extraction and not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        payload = {
            "model": request.model.rsplit("/", 1)[-1], "state": state,
            "fields" if extraction else "questions": questions,
        }
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._credential.account_id}/ai/run/{request.model}"
        try:
            http_response = await self._client.post(
                url, json=payload,
                headers={"Authorization": f"Bearer {self._credential.api_token.get_secret_value()}"},
                timeout=request.timeout if request.timeout is not None else self._timeout_seconds,
            )
            http_response.raise_for_status()
            envelope = _Envelope.model_validate(http_response.json())
            if not envelope.success or envelope.result is None:
                raise ValueError("Cloudflare inference did not succeed")
            raw = envelope.result
        except (ValidationError, ValueError) as exc:
            raise ToolProviderError("Cloudflare inference request failed", provider="cloudflare") from exc

        # Establish charge evidence independently before parsing model answers.
        try:
            usage = _BillingProbe.model_validate(raw).usage
            if usage.output_tokens != 0:
                raise ValueError("Cloudflare decision models do not generate output tokens")
            canonical_usage = DecisionUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
            cost = price_decision("cloudflare", request.model, canonical_usage)
        except (ValidationError, ValueError) as exc:
            raise ToolProviderError("Cloudflare returned invalid usage", provider="cloudflare") from exc
        evidence: JsonObject = {
            "provider": "cloudflare", "model": request.model, "settlement_source": "static_pricing",
            "pricing_origin": "cloudflare_workers_ai_published_rates", "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        billing = ProviderBillingMetadata(
            actual_cost_provider="cloudflare", actual_cost_usd=cost,
            reference_cost_usd=cost,
            source="response_body", usage=canonical_usage, provider_usage=evidence,
        )
        try:
            native = _Payload.model_validate(raw)
            if native.model not in {request.model, request.model.rsplit("/", 1)[-1]}:
                raise ValueError("Cloudflare returned a different model")
            response = DecisionQueryResponse(
                provider="cloudflare", model=request.model,
                answers={
                    key: BooleanAnswer(type="boolean", probability=answer.noul)
                    if isinstance(answer, _NoulAnswer) else answer
                    for key, answer in native.answers.items()
                },
                usage=canonical_usage, truncated=native.truncated,
            )
            if extraction and native.truncated is None:
                raise ValueError("Cloudflare extraction response is missing truncation status")
            validate_decision_answers(request, response)
        except (ValidationError, ValueError) as exc:
            raise ToolProviderError(
                "Cloudflare returned invalid answers", provider="cloudflare", billing=billing,
            ) from exc
        return DecisionProviderResult(response, cost, "cloudflare", evidence, reference_cost_usd=cost)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
