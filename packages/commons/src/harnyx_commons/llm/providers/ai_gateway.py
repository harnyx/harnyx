"""Vercel AI Gateway LLM provider."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from harnyx_commons.llm.cost_settlement import settled_response_cost, with_settled_llm_cost
from harnyx_commons.llm.provider import BaseLlmProvider, LlmProviderConfigurationError
from harnyx_commons.llm.provider_types import AI_GATEWAY_PROVIDER
from harnyx_commons.llm.providers.openai_chat_codec import OpenAiChatRequestParts
from harnyx_commons.llm.providers.openai_stream import (
    OpenAiChoiceState,
    OpenAiStreamError,
    OpenAiStreamState,
    OpenAiToolCall,
    _OpenAiStreamEvent,
    iter_openai_sse_payloads,
    normalize_openai_reasoning_fragments,
)
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmThinkingConfig,
    LlmUsage,
)
from harnyx_commons.llm.tool_models import MINER_SELECTED_LLM_PROVIDER_MODELS

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_SUPPORTED_MODELS = MINER_SELECTED_LLM_PROVIDER_MODELS[AI_GATEWAY_PROVIDER]


class AiGatewayLlmProvider(BaseLlmProvider):
    def __init__(
        self,
        *,
        ai_gateway_api_key: SecretStr,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(provider_label=AI_GATEWAY_PROVIDER, max_concurrent=None)
        normalized_key = ai_gateway_api_key.get_secret_value().strip()
        if not normalized_key:
            raise LlmProviderConfigurationError("AI_GATEWAY_API_KEY must be configured to build AI Gateway provider")
        self._owns_client = client is None
        self._client = client or build_ai_gateway_client(normalized_key)
        self._chat_completions_url = f"{AI_GATEWAY_BASE_URL}/chat/completions"

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        model = request.model.strip()
        if model not in AI_GATEWAY_SUPPORTED_MODELS:
            raise ValueError(f"AI Gateway provider does not support model {request.model!r}")
        return await self._call_with_retry(
            request,
            call_coro=lambda current_request: self._request_chat(
                _AiGatewayChatRequest.from_request(current_request),
                timeout_seconds=current_request.timeout_seconds,
            ),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
            policy=request.retry_policy,
        )

    async def _annotate_response_cost(
        self,
        request: AbstractLlmRequest,
        response: LlmResponse,
    ) -> LlmResponse:
        cost = settled_response_cost(response, provider=AI_GATEWAY_PROVIDER, model=request.model)
        if cost is None:
            self._llm_logger.warning(
                "ai_gateway.cost_settlement.unavailable",
                extra={"data": {"provider": AI_GATEWAY_PROVIDER, "model": request.model}},
            )
            return response
        return with_settled_llm_cost(response, cost)

    async def _request_chat(
        self,
        payload: _AiGatewayChatRequest,
        *,
        timeout_seconds: float | None,
    ) -> LlmResponse:
        request_kwargs: dict[str, Any] = {
            "json": payload.model_dump(mode="json", by_alias=True, exclude_none=True),
        }
        if timeout_seconds is not None:
            request_kwargs["timeout"] = timeout_seconds
        body, ttft_ms = await self._stream_chat_completions(**request_kwargs)
        response = body.to_llm_response()
        metadata = dict(response.metadata or {})
        metadata.setdefault("raw_response", body.raw_payload())
        if ttft_ms is not None:
            metadata.setdefault("ttft_ms", ttft_ms)
        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=response.usage,
            metadata=metadata,
            finish_reason=response.finish_reason,
        )

    async def _stream_chat_completions(
        self,
        **request_kwargs: Any,
    ) -> tuple[_AiGatewayChatResponse, float | None]:
        started_at = time.perf_counter()
        state = OpenAiStreamState()
        provider_metadata: dict[str, Any] | None = None
        ttft_ms: float | None = None
        async with self._client.stream("POST", self._chat_completions_url, **request_kwargs) as response:
            if response.is_error:
                await response.aread()
            response.raise_for_status()
            async for payload in iter_openai_sse_payloads(
                response,
                invalid_data_message="AI Gateway chat completions returned non-JSON SSE data",
                invalid_event_message="AI Gateway chat completions SSE event must be a JSON object",
            ):
                provider_metadata = _provider_metadata_from_payload(payload) or provider_metadata
                try:
                    event = _OpenAiStreamEvent.model_validate(payload)
                except ValidationError as exc:
                    raise OpenAiStreamError(
                        message="AI Gateway chat completions SSE event must be a JSON object",
                        error_type="server_error",
                        code=502,
                    ) from exc
                if state.merge_event(
                    event,
                    reasoning_keys=("reasoning", "reasoning_content", "reasoning_details"),
                    normalize_reasoning_fragment=normalize_openai_reasoning_fragments,
                ):
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return _AiGatewayChatResponse.from_stream_state(state, provider_metadata=provider_metadata), ttft_ms

    @staticmethod
    def _verify_response(response: LlmResponse) -> tuple[bool, bool, str | None]:
        if not response.choices:
            return False, True, "empty_choices"
        if not response.raw_text and not response.tool_calls:
            return False, True, "empty_output"
        for call in _iter_tool_calls(response):
            if not _is_valid_json(call.arguments):
                return False, True, "tool_call_args_invalid_json"
        return True, False, None

    @staticmethod
    def _classify_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        match exc:
            case httpx.HTTPStatusError():
                status = exc.response.status_code if exc.response else None
                retryable = status is not None and (status == 429 or status >= 500)
                detail = _summarize_response(exc.response) if exc.response is not None else ""
                if detail:
                    return retryable, f"http_{status}: {detail}"
                return retryable, f"http_{status}"
            case httpx.HTTPError():
                return True, exc.__class__.__name__
            case OpenAiStreamError():
                return exc.retryable, exc.reason
        if classify_exception is not None:
            return classify_exception(exc)
        return False, str(exc)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class _AiGatewayChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, strict=True)

    model: str
    messages: list[dict[str, Any]]
    stream: bool = True
    stream_options: dict[str, bool] = Field(default_factory=lambda: {"include_usage": True})
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    include: list[str] | None = None
    response_format: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    provider_options: dict[str, Any] | None = Field(default=None, alias="providerOptions")

    @classmethod
    def from_request(cls, request: AbstractLlmRequest) -> _AiGatewayChatRequest:
        if request.grounded:
            raise ValueError("grounded mode is not supported for AI Gateway provider")
        request_parts = OpenAiChatRequestParts.from_request(
            request,
            provider_name=AI_GATEWAY_PROVIDER,
            image_error_message="AI Gateway provider does not support image content parts",
            tool_mix_error_message="AI Gateway input_tool_result messages cannot mix text parts",
            tool_count_error_message="AI Gateway input_tool_result messages must include exactly one part",
        )
        payload = cls(
            model=request.model,
            messages=[message.model_dump(mode="python", exclude_none=True) for message in request_parts.messages],
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            tools=(
                [tool.model_dump(mode="python", exclude_none=True) for tool in request_parts.tools]
                if request_parts.tools
                else None
            ),
            tool_choice=request_parts.tool_choice,
            parallel_tool_calls=request_parts.parallel_tool_calls,
            include=request_parts.include,
            response_format=(
                request_parts.response_format.model_dump(mode="python", exclude_none=True)
                if request_parts.response_format is not None
                else None
            ),
        )
        if request.extra:
            payload = payload.model_copy(update=dict(request.extra))
        payload = payload.model_copy(
            update={
                "reasoning": _merge_reasoning_extra(
                    payload.reasoning,
                    request.thinking,
                )
            }
        )
        return payload.model_copy(update={"stream": True})


class _AiGatewayUsageDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    reasoning_tokens: int | None = None


class _AiGatewayUsagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    completion_tokens_details: _AiGatewayUsageDetails | None = None

    def to_usage(self) -> LlmUsage:
        reasoning_tokens = self._reasoning_tokens()
        return LlmUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=_completion_tokens_excluding_reasoning(
                completion_tokens=self.completion_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
            total_tokens=self.total_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def _reasoning_tokens(self) -> int | None:
        if self.reasoning_tokens is not None:
            return self.reasoning_tokens
        if self.completion_tokens_details is None:
            return None
        return self.completion_tokens_details.reasoning_tokens


class _AiGatewayChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str
    choices: list[_AiGatewayChoicePayload] = Field(default_factory=list)
    usage: _AiGatewayUsagePayload | None = None
    provider_metadata: dict[str, Any] | None = None

    @classmethod
    def from_stream_state(
        cls,
        state: OpenAiStreamState,
        *,
        provider_metadata: dict[str, Any] | None,
    ) -> _AiGatewayChatResponse:
        choices = [
            _AiGatewayChoicePayload.from_choice_state(index=index, state=choice_state)
            for index, choice_state in sorted(state.choices.items())
        ]
        usage = _AiGatewayUsagePayload.model_validate(state.usage) if state.usage is not None else None
        return cls(
            id=state.response_id,
            choices=choices,
            usage=usage,
            provider_metadata=provider_metadata,
        )

    def raw_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="python", exclude_none=True, exclude={"provider_metadata"})
        if self.provider_metadata is not None:
            payload["providerMetadata"] = self.provider_metadata
        return payload

    def to_llm_response(self) -> LlmResponse:
        choices = tuple(choice.to_choice() for choice in self.choices)
        return LlmResponse(
            id=self.id,
            choices=choices,
            usage=self.usage.to_usage() if self.usage is not None else LlmUsage(),
            finish_reason=choices[0].finish_reason if choices else None,
        )


class _AiGatewayChoicePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int
    content: str
    reasoning: str | None = None
    reasoning_details: tuple[dict[str, Any], ...] | None = None
    tool_calls: tuple[LlmMessageToolCall, ...] | None = None
    finish_reason: str | None = None

    @classmethod
    def from_choice_state(cls, *, index: int, state: OpenAiChoiceState) -> _AiGatewayChoicePayload:
        return cls(
            index=index,
            content=state.content_text,
            reasoning=state.reasoning_text or None,
            reasoning_details=tuple(state.reasoning_details) or None,
            tool_calls=_to_llm_tool_calls(state),
            finish_reason=state.finish_reason,
        )

    def to_choice(self) -> LlmChoice:
        return LlmChoice(
            index=self.index,
            message=LlmChoiceMessage(
                role="assistant",
                content=(LlmMessageContentPart(type="text", text=self.content),),
                reasoning=self.reasoning,
                reasoning_details=self.reasoning_details,
                tool_calls=self.tool_calls,
            ),
            finish_reason=self.finish_reason or "stop",
        )


def build_ai_gateway_client(api_key: str) -> httpx.AsyncClient:
    normalized_key = api_key.strip()
    if not normalized_key:
        raise LlmProviderConfigurationError("AI_GATEWAY_API_KEY must be configured to build AI Gateway provider")
    return httpx.AsyncClient(
        base_url=AI_GATEWAY_BASE_URL,
        headers={"Authorization": f"Bearer {normalized_key}"},
    )


def _provider_metadata_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    provider_metadata = payload.get("providerMetadata")
    if not isinstance(provider_metadata, Mapping):
        provider_metadata = payload.get("provider_metadata")
    if isinstance(provider_metadata, Mapping):
        return dict(provider_metadata)

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        choice_metadata = _provider_metadata_from_choice(choice)
        if choice_metadata is not None:
            return choice_metadata
    return None


def _provider_metadata_from_choice(choice: Mapping[str, Any]) -> dict[str, Any] | None:
    for message_key in ("delta", "message"):
        message = choice.get(message_key)
        if not isinstance(message, Mapping):
            continue
        provider_metadata = message.get("provider_metadata")
        if isinstance(provider_metadata, Mapping):
            return dict(provider_metadata)
        provider_metadata = message.get("providerMetadata")
        if isinstance(provider_metadata, Mapping):
            return dict(provider_metadata)
    return None


def _completion_tokens_excluding_reasoning(
    *,
    completion_tokens: int | None,
    reasoning_tokens: int | None,
) -> int | None:
    if completion_tokens is None or reasoning_tokens is None:
        return completion_tokens
    return max(0, completion_tokens - reasoning_tokens)


def _to_llm_tool_calls(state: OpenAiChoiceState) -> tuple[LlmMessageToolCall, ...] | None:
    tool_calls = state.tool_call_values()
    if not tool_calls:
        return None
    return tuple(_to_llm_tool_call(tool_call) for tool_call in tool_calls)


def _to_llm_tool_call(tool_call: OpenAiToolCall) -> LlmMessageToolCall:
    return LlmMessageToolCall(
        id=tool_call.id,
        type=tool_call.type,
        name=tool_call.name,
        arguments=tool_call.arguments,
    )


def _summarize_response(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (ValueError, RuntimeError):
        try:
            data = response.text
        except RuntimeError:
            data = ""
    text = str(data.get("detail", data)) if isinstance(data, dict) else str(data)
    return text if len(text) <= 500 else text[:500] + "..."


def _iter_tool_calls(response: LlmResponse) -> tuple[LlmMessageToolCall, ...]:
    calls: list[LlmMessageToolCall] = []
    for choice in response.choices:
        calls.extend(choice.message.tool_calls or ())
    return tuple(calls)


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _merge_reasoning_extra(
    request_reasoning: object,
    thinking: LlmThinkingConfig | None,
) -> dict[str, Any] | None:
    reasoning_payload = _reasoning_payload(thinking)
    if request_reasoning is None:
        return reasoning_payload
    if not isinstance(request_reasoning, Mapping):
        raise ValueError("AI Gateway request extra.reasoning must be an object")
    merged = dict(request_reasoning)
    if reasoning_payload is not None:
        merged.update(reasoning_payload)
    return merged


def _reasoning_payload(thinking: LlmThinkingConfig | None) -> dict[str, Any] | None:
    if thinking is None:
        return None
    if not thinking.enabled:
        return {"effort": "none"}
    payload: dict[str, Any] = {"enabled": True}
    if thinking.effort is not None:
        payload["effort"] = thinking.effort
    if thinking.budget is not None:
        payload["max_tokens"] = thinking.budget
    return payload


__all__ = [
    "AI_GATEWAY_BASE_URL",
    "AI_GATEWAY_SUPPORTED_MODELS",
    "AiGatewayLlmProvider",
    "build_ai_gateway_client",
]
