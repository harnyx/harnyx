"""OpenRouter chat, embedding and Decision adapters."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from typing import Any, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig
from harnyx_commons.errors import ToolInvocationCancelledError, ToolInvocationTimeoutError, ToolProviderError
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.cost_settlement import normalized_provider_cost, settled_response_cost
from harnyx_commons.llm.provider import (
    LlmProviderConfigurationError,
    LlmProviderError,
    LlmProviderPort,
    LlmRetryExhaustedError,
)
from harnyx_commons.llm.provider_types import OPENROUTER_PROVIDER
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.providers.openrouter_pricing import (
    OpenRouterPricingCatalog,
    OpenRouterPricingSnapshot,
    shared_openrouter_pricing_catalog,
)
from harnyx_commons.llm.providers.thinking import resolve_request_thinking
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse, LlmThinkingConfig
from harnyx_commons.llm.timeout import LlmAttemptTimeoutError
from harnyx_commons.llm.tool_models import MINER_SELECTED_LLM_PROVIDER_MODELS, validate_paid_openrouter_model
from harnyx_commons.tools.provider_billing import OpenRouterAccounting, ProviderBillingMetadata
from harnyx_miner_sdk.tools.llm_provider_extra import OpenRouterProviderSelection

OPENROUTER_ENDPOINT_ID = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
logger = logging.getLogger(__name__)
OPENROUTER_NATIVE_SUPPORTED_MODELS = MINER_SELECTED_LLM_PROVIDER_MODELS[OPENROUTER_PROVIDER]
OPENROUTER_EMBEDDING_SUPPORTED_MODELS = ("qwen/qwen3-embedding-8b",)
OPENROUTER_INTERNAL_TO_NATIVE_MODEL: Mapping[str, str] = {
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "deepseek-ai/DeepSeek-V4-Flash-0731-TEE": "deepseek/deepseek-v4-flash-0731",
    "zai-org/GLM-5-TEE": "z-ai/glm-5",
    "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
    "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
}
OPENROUTER_INTERNAL_SUPPORTED_MODELS = tuple(OPENROUTER_INTERNAL_TO_NATIVE_MODEL)
OPENROUTER_SUPPORTED_MODELS = tuple(
    dict.fromkeys((*OPENROUTER_NATIVE_SUPPORTED_MODELS, *OPENROUTER_INTERNAL_SUPPORTED_MODELS))
)


OpenRouterChatProviderFactory = Callable[[str], tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]]


class _MinerBillingUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    cost: Decimal | None = Field(default=None, ge=0)
    is_byok: bool | None = Field(default=None, strict=True)
    cost_details: JsonObject | None = None


class _MinerBillingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    provider: str | None = None
    usage: _MinerBillingUsage
    openrouter_metadata: JsonObject | None = None


class _MinerCostDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    upstream_inference_cost: Decimal = Field(ge=0)


def _miner_openrouter_billing(
    response: LlmResponse | None,
    *,
    snapshot: OpenRouterPricingSnapshot | None,
    requested_model: str,
) -> ProviderBillingMetadata:
    actual: float | None = None
    reference: float | None = None
    evidence: JsonObject = {"settlement_source": "unavailable", "requested_model": requested_model}
    if response is not None and response.usage is not None:
        evidence["token_usage"] = asdict(response.usage)
    if snapshot is not None:
        evidence.update(
            snapshot_model=snapshot.model,
            snapshot_time=snapshot.fetched_at,
            snapshot_source="openrouter_public_endpoints",
        )
    try:
        if response is None or snapshot is None:
            raise ValueError("no original response and pricing snapshot")
        settled = settled_response_cost(response, provider=OPENROUTER_PROVIDER, model=requested_model)
        if settled is not None and settled.evidence.get("settlement_source") == "provider_returned":
            actual = settled.cost_usd
            evidence.update(settled.evidence)
        if actual is None:
            raise ValueError("original OpenRouter billing unavailable; static estimates are not actual charges")
        aggregate = (response.metadata or {}).get("actual_cost_usd_total")
        if (response.metadata or {}).get("billable_response_count", 1) != 1:
            actual = normalized_provider_cost(aggregate, field_name="actual_cost_usd_total")
            raise ValueError("multiple billable responses cannot use one serving-provider discount")
        raw = _MinerBillingResponse.model_validate((response.metadata or {}).get("raw_response"))
        evidence["original_usage"] = raw.usage.model_dump(mode="json")
        reference, reference_evidence = _openrouter_reference_cost(raw, snapshot)
        evidence.update(reference_evidence)
    except (ValueError, ArithmeticError) as exc:
        evidence["reference_unavailable_reason"] = str(exc)
    return OpenRouterAccounting(
        inference="attempted",
        actual_cost_usd=actual,
        reference_cost_usd=reference,
        llm_usage=None if response is None else response.usage,
    ).to_billing(evidence)


def _openrouter_reference_cost(
    raw: _MinerBillingResponse, snapshot: OpenRouterPricingSnapshot
) -> tuple[float, JsonObject]:
    evidence: JsonObject = {}
    resolved_model, serving_provider = raw.model, raw.provider
    if resolved_model is None or serving_provider is None:
        routing = _OpenRouterRouterMetadata.model_validate(raw.openrouter_metadata)
        selected = () if routing.endpoints is None else tuple(e for e in routing.endpoints.available if e.selected)
        providers = {e.provider.casefold() for e in selected}
        models = {e.model for e in selected if e.model is not None}
        if serving_provider is None and len(providers) == 1:
            serving_provider = selected[0].provider
        if resolved_model is None and len(models) == 1:
            resolved_model = next(iter(models))
    if not isinstance(resolved_model, str) or not isinstance(serving_provider, str):
        raise ValueError("original OpenRouter serving model/provider is missing")
    validate_paid_openrouter_model(resolved_model)
    resolved = resolved_model.split(":", 1)[0]
    if resolved not in {snapshot.model, snapshot.canonical_model} and not any(
        e.model_id.split(":", 1)[0] == resolved for e in snapshot.endpoints
    ):
        raise ValueError("resolved OpenRouter model does not match the pricing snapshot")
    if raw.usage.cost is None:
        raise ValueError("original OpenRouter charge is missing")
    inference, separate = raw.usage.cost, Decimal(0)
    if raw.usage.is_byok is True:
        inference = _MinerCostDetails.model_validate(raw.usage.cost_details).upstream_inference_cost
        separate = raw.usage.cost
    reference, discount = snapshot.reference_cost(
        provider=serving_provider, inference=inference, separate_charge=separate
    )
    evidence.update(
        resolved_model=resolved_model,
        upstream_provider=serving_provider,
        maximum_discount=float(discount),
    )
    return reference, evidence


class _OpenRouterEmbeddingDatum(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    embedding: list[float] = Field(min_length=1)
    index: int | None = None


class _OpenRouterEmbeddingUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    prompt_tokens: int | None = None
    total_tokens: int | None = None
    cost: JsonValue | None = None
    cost_details: JsonObject | None = None


class _OpenRouterEmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    data: list[_OpenRouterEmbeddingDatum] = Field(min_length=1)
    model: str
    id: str | None = None
    usage: _OpenRouterEmbeddingUsage | None = None


@dataclass(frozen=True, slots=True)
class OpenRouterEmbeddingUsage:
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    cost: JsonValue | None = None
    cost_details: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class OpenRouterEmbeddingResponse:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    usage: OpenRouterEmbeddingUsage | None = None
    id: str | None = None


@dataclass(slots=True)
class OpenRouterEmbeddingClient:
    model: str
    api_key: SecretStr = field(repr=False)
    base_url: str = OPENROUTER_BASE_URL
    timeout_seconds: float = 30.0
    dimensions: int | None = None
    client: httpx.AsyncClient | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        normalized_model = self.model.strip()
        if normalized_model not in OPENROUTER_EMBEDDING_SUPPORTED_MODELS:
            raise ValueError(f"OpenRouter embedding provider does not support model {self.model!r}")
        api_key_value = self.api_key.get_secret_value().strip()
        if not api_key_value:
            raise ValueError("OpenRouter API key must be provided for embeddings")
        self.model = normalized_model
        self.api_key = SecretStr(api_key_value)
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("OpenRouter embedding base_url must not be empty")
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                headers={"Authorization": f"Bearer {api_key_value}"},
            )
            self._owns_client = True
        else:
            self._owns_client = False

    async def embed_many(
        self,
        texts: Sequence[str],
        *,
        extra: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> OpenRouterEmbeddingResponse:
        normalized = tuple(text.strip() for text in texts)
        if not normalized or any(not text for text in normalized):
            raise ValueError("embedding input texts must contain non-empty strings")
        response = await self._require_client().post(
            "embeddings",
            json=self._request_body(normalized, extra=extra),
            timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
        )
        response.raise_for_status()
        payload = _OpenRouterEmbeddingResponse.model_validate(response.json())
        ordered = sorted(
            enumerate(payload.data),
            key=lambda item: item[1].index if item[1].index is not None else item[0],
        )
        vectors = tuple(tuple(float(value) for value in item.embedding) for _, item in ordered)
        for vector in vectors:
            if self.dimensions is not None and len(vector) != self.dimensions:
                raise RuntimeError(f"embedding dimensions mismatch: expected={self.dimensions} actual={len(vector)}")
        usage = None
        if payload.usage is not None:
            usage = OpenRouterEmbeddingUsage(
                prompt_tokens=payload.usage.prompt_tokens,
                total_tokens=payload.usage.total_tokens,
                cost=payload.usage.cost,
                cost_details=payload.usage.cost_details,
            )
        return OpenRouterEmbeddingResponse(
            vectors=vectors,
            model=payload.model,
            usage=usage,
            id=payload.id,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._require_client().aclose()

    def _request_body(self, texts: Sequence[str], *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        input_value: str | list[str]
        if len(texts) == 1:
            input_value = texts[0]
        else:
            input_value = list(texts)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_value,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        if extra is not None:
            payload.update(extra)
        return payload

    def _require_client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("OpenRouter embedding client is not initialized")
        return self.client


class OpenRouterLlmProvider(LlmProviderPort):
    def __init__(
        self,
        *,
        openrouter_api_key: SecretStr,
        openrouter_chat_provider_factory: OpenRouterChatProviderFactory | None = None,
        additional_supported_models: Sequence[str] = (),
        pricing_catalog: OpenRouterPricingCatalog | None = None,
    ) -> None:
        self._pricing_catalog = pricing_catalog or shared_openrouter_pricing_catalog()
        self._openrouter_api_key = openrouter_api_key
        self._openrouter_chat_provider_factory = openrouter_chat_provider_factory or build_openrouter_chat_provider
        self._supported_models = frozenset(
            (*OPENROUTER_SUPPORTED_MODELS, *(model.strip() for model in additional_supported_models if model.strip()))
        )
        self._openrouter_provider: OpenAiCompatibleLlmProvider | None = None
        self._openrouter_client: httpx.AsyncClient | None = None

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        model = request.model.strip()
        if request.use_case == "tool_runtime_invoker":
            return await self._invoke_miner(request)
        if model not in self._supported_models:
            raise ValueError(f"OpenRouter provider does not support model {request.model!r}")
        openrouter_provider = self._ensure_openrouter_provider(model=model)
        response = await openrouter_provider.invoke(self._openrouter_request(request, model=model))
        metadata = dict(response.metadata or {})
        metadata["effective_provider"] = OPENROUTER_PROVIDER
        metadata["effective_model"] = model
        routing_evidence = _openrouter_routing_evidence(metadata.get("raw_response"))
        cost_evidence = metadata.get("actual_cost_evidence")
        if routing_evidence and isinstance(cost_evidence, Mapping):
            metadata["actual_cost_evidence"] = {**cost_evidence, **routing_evidence}
        return replace(response, metadata=metadata)

    async def _invoke_miner(self, request: AbstractLlmRequest) -> LlmResponse:
        dispatched = False
        snapshot: OpenRouterPricingSnapshot | None = None
        response: LlmResponse | None = None

        def dispatch() -> None:
            nonlocal dispatched
            dispatched = True
            if request.on_provider_dispatch is not None:
                request.on_provider_dispatch()

        try:
            model = validate_paid_openrouter_model(request.model)
            if request.retry_policy is None or request.retry_policy.attempts != 1:
                raise ValueError("miner OpenRouter accounting requires one provider attempt")
            provider = self._ensure_openrouter_provider(model=model)
            selection_payload = (request.extra or {}).get("provider")
            selection = (
                None if selection_payload is None else OpenRouterProviderSelection.model_validate(selection_payload)
            )
            snapshot = await self._pricing_catalog.snapshot(
                model, provider_only=None if selection is None else selection.only
            )
            response = await provider.invoke(
                replace(self._openrouter_request(request, model=model), on_provider_dispatch=dispatch)
            )
            billing = _miner_openrouter_billing(response, snapshot=snapshot, requested_model=model)
            if billing.reference_cost_unavailable:
                raise ToolProviderError(
                    "OpenRouter regular cost unavailable", provider=OPENROUTER_PROVIDER, billing=billing
                )
            metadata = dict(response.metadata or {})
            # The canonical single-response settlement supersedes the delegate's
            # retry total, which would otherwise rewrap and hide these billing facts.
            metadata.pop("actual_cost_usd_total", None)
            metadata.update(
                actual_cost_usd=billing.actual_cost_usd,
                actual_cost_provider=OPENROUTER_PROVIDER,
                actual_cost_evidence=billing.provider_usage,
                reference_cost_usd=billing.reference_cost_usd,
                effective_provider=OPENROUTER_PROVIDER,
                effective_model=model,
            )
            return replace(response, metadata=metadata)
        except BaseException as exc:
            if isinstance(exc, (ToolProviderError, ToolInvocationTimeoutError)) and exc.billing is not None:
                raise
            # Rejected completions can still be billable; retry wrappers retain their last response.
            failed_response = getattr(exc, "response", None)
            if isinstance(failed_response, LlmResponse):
                response = failed_response
            billing = (
                _miner_openrouter_billing(response, snapshot=snapshot, requested_model=request.model)
                if dispatched or response is not None
                else OpenRouterAccounting(
                    inference="not_dispatched", actual_cost_usd=0.0, reference_cost_usd=0.0
                ).to_billing({"settlement_source": "pre_dispatch", "reason": str(exc)}, source="request_body")
            )
            if isinstance(exc, asyncio.CancelledError):
                raise ToolInvocationCancelledError("OpenRouter call cancelled", billing=billing) from exc
            if isinstance(exc, ToolInvocationTimeoutError):
                exc.billing = billing
                raise
            if isinstance(exc, (LlmProviderError, LlmRetryExhaustedError)) and isinstance(
                exc.__cause__, LlmAttemptTimeoutError
            ):
                raise ToolInvocationTimeoutError(str(exc.__cause__), billing=billing) from exc
            if not isinstance(exc, Exception):
                raise
            raise ToolProviderError(str(exc), provider=OPENROUTER_PROVIDER, billing=billing) from exc

    async def aclose(self) -> None:
        errors: list[Exception] = []
        if self._openrouter_provider is not None:
            try:
                await self._openrouter_provider.aclose()
            except Exception as exc:
                exc.add_note("OpenRouter OpenAI-compatible delegate close failed")
                errors.append(exc)
        if self._openrouter_client is not None:
            try:
                await self._openrouter_client.aclose()
            except Exception as exc:
                exc.add_note("OpenRouter HTTP client close failed")
                errors.append(exc)
        if errors:
            raise ExceptionGroup("OpenRouter provider cleanup failed", errors)

    def _ensure_openrouter_provider(self, *, model: str) -> OpenAiCompatibleLlmProvider:
        if self._openrouter_provider is not None:
            return self._openrouter_provider
        normalized_key = self._openrouter_api_key.get_secret_value().strip()
        if not normalized_key:
            raise LlmProviderConfigurationError(
                f"OPENROUTER_API_KEY must be configured to use OpenRouter model {model}"
            )
        provider, client = self._openrouter_chat_provider_factory(normalized_key)
        self._openrouter_provider = provider
        self._openrouter_client = client
        return provider

    def _openrouter_request(self, request: AbstractLlmRequest, *, model: str) -> AbstractLlmRequest:
        extra = dict(request.extra or {})
        thinking = resolve_request_thinking(
            request_thinking=request.thinking,
            reasoning_effort=request.reasoning_effort,
        )
        extra = _merge_reasoning_extra(extra, thinking)
        return replace(
            request,
            provider=OPENROUTER_PROVIDER,
            model=OPENROUTER_INTERNAL_TO_NATIVE_MODEL.get(model, model),
            extra=extra or None,
        )


def build_openrouter_chat_provider(api_key: str) -> tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]:
    normalized_key = api_key.strip()
    if not normalized_key:
        raise LlmProviderConfigurationError("OPENROUTER_API_KEY must be configured to build OpenRouter provider")
    client = httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL,
        headers={
            "Authorization": f"Bearer {normalized_key}",
            "X-OpenRouter-Metadata": "enabled",
        },
    )
    endpoint = OpenAiCompatibleEndpointConfig.model_validate(
        {
            "id": OPENROUTER_ENDPOINT_ID,
            "base_url": OPENROUTER_BASE_URL,
            "auth": {"type": "none"},
        }
    )
    return OpenAiCompatibleLlmProvider(
        endpoint=endpoint,
        client=client,
        response_metadata_extractor=_openrouter_response_metadata,
    ), client


class _OpenRouterEndpointMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    provider: str
    model: str | None = None
    selected: bool = False


class _OpenRouterEndpointsMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    available: list[_OpenRouterEndpointMetadata] = Field(default_factory=list)


class _OpenRouterRouterMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    endpoints: _OpenRouterEndpointsMetadata | None = None


class _OpenRouterResponseIdentity(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    id: str | None = None


def _openrouter_response_metadata(payload: Mapping[str, Any]) -> JsonObject | None:
    metadata: JsonObject = {}
    router_metadata = payload.get("openrouter_metadata")
    if isinstance(router_metadata, Mapping):
        metadata["openrouter_metadata"] = dict(router_metadata)
    for key in ("model", "provider"):
        if isinstance(payload.get(key), str):
            metadata[key] = payload[key]
    return metadata or None


def _openrouter_routing_evidence(raw_response: object) -> JsonObject:
    if not isinstance(raw_response, Mapping):
        return {}
    response_payload = cast(Mapping[str, object], raw_response)

    try:
        provider_request_id = _OpenRouterResponseIdentity.model_validate(response_payload).id
    except ValidationError:
        logger.warning("OpenRouter response identity is malformed")
        provider_request_id = None

    evidence: JsonObject = {}
    if provider_request_id is not None:
        evidence["provider_request_id"] = provider_request_id

    raw_router_metadata = response_payload.get("openrouter_metadata")
    if raw_router_metadata is None:
        return evidence
    try:
        router_metadata = _OpenRouterRouterMetadata.model_validate(raw_router_metadata)
    except ValidationError:
        logger.warning(
            "OpenRouter router metadata is malformed",
            extra={"provider_request_id": provider_request_id},
        )
        return evidence

    selected_endpoint = None
    if router_metadata.endpoints is not None:
        selected_endpoint = next(
            (endpoint for endpoint in router_metadata.endpoints.available if endpoint.selected),
            None,
        )
    if selected_endpoint is None:
        return evidence
    evidence["upstream_provider"] = selected_endpoint.provider
    if selected_endpoint.model is not None:
        evidence["upstream_model"] = selected_endpoint.model
    return evidence


def _merge_reasoning_extra(
    request_extra: Mapping[str, Any] | None,
    thinking: LlmThinkingConfig | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(request_extra or {})
    request_reasoning = merged.get("reasoning")
    if request_reasoning is not None and not isinstance(request_reasoning, Mapping):
        raise ValueError("OpenRouter request extra.reasoning must be an object")
    reasoning = _reasoning_payload(thinking)
    if reasoning is None:
        return merged
    merged_reasoning = dict(request_reasoning or {})
    merged_reasoning.update(reasoning)
    merged["reasoning"] = merged_reasoning
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
    "OPENROUTER_BASE_URL",
    "OPENROUTER_EMBEDDING_SUPPORTED_MODELS",
    "OPENROUTER_ENDPOINT_ID",
    "OPENROUTER_INTERNAL_SUPPORTED_MODELS",
    "OPENROUTER_INTERNAL_TO_NATIVE_MODEL",
    "OPENROUTER_NATIVE_SUPPORTED_MODELS",
    "OPENROUTER_SUPPORTED_MODELS",
    "OpenRouterEmbeddingClient",
    "OpenRouterEmbeddingResponse",
    "OpenRouterEmbeddingUsage",
    "OpenRouterLlmProvider",
    "build_openrouter_chat_provider",
]


@dataclass(frozen=True, slots=True)
class OpenRouterDecisionResponse:
    raw_payload: JsonObject


@dataclass(slots=True)
class OpenRouterDecisionClient:
    model: str
    api_key: SecretStr = field(repr=False)
    base_url: str = "https://openrouter.ai/api/alpha"
    timeout_seconds: float = 120.0
    client: httpx.AsyncClient | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key.get_secret_value().strip():
            raise ValueError("OpenRouter API key must be provided for decisions")
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {self.api_key.get_secret_value().strip()}"},
                timeout=self.timeout_seconds,
            )
            self._owns_client = True

    async def query(
        self,
        state: JsonValue,
        questions: JsonObject,
        *,
        timeout_seconds: float | None = None,
    ) -> OpenRouterDecisionResponse:
        native_questions: JsonObject = {}
        for key, question in questions.items():
            if not isinstance(question, dict):
                raise ValueError("decision question must be an object")
            native_questions[key] = {
                **question,
                "type": "noul" if question.get("type") == "boolean" else question.get("type"),
            }
        assert self.client is not None
        response = await self.client.post(
            "decisions",
            headers={"Authorization": f"Bearer {self.api_key.get_secret_value().strip()}"},
            json={"model": self.model, "state": state, "questions": native_questions},
            timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
        )
        response.raise_for_status()
        return OpenRouterDecisionResponse(TypeAdapter(JsonObject).validate_python(response.json()))

    async def aclose(self) -> None:
        if self._owns_client:
            assert self.client is not None
            await self.client.aclose()
