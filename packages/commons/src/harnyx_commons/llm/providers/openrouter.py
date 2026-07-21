"""Hardcoded OpenRouter provider for repo-owned OpenRouter routes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.provider import LlmProviderConfigurationError, LlmProviderPort
from harnyx_commons.llm.provider_types import OPENROUTER_PROVIDER
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse, LlmThinkingConfig

OPENROUTER_ENDPOINT_ID = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
logger = logging.getLogger(__name__)
OPENROUTER_NATIVE_SUPPORTED_MODELS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-v3.2",
    "z-ai/glm-5",
    "qwen/qwen3.6-27b",
    "google/gemma-4-31b-it",
)
OPENROUTER_EMBEDDING_SUPPORTED_MODELS = ("qwen/qwen3-embedding-8b",)
OPENROUTER_INTERNAL_TO_NATIVE_MODEL: Mapping[str, str] = {
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "zai-org/GLM-5-TEE": "z-ai/glm-5",
    "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
    "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
}
OPENROUTER_INTERNAL_SUPPORTED_MODELS = tuple(OPENROUTER_INTERNAL_TO_NATIVE_MODEL)
OPENROUTER_SUPPORTED_MODELS = tuple(
    dict.fromkeys((*OPENROUTER_NATIVE_SUPPORTED_MODELS, *OPENROUTER_INTERNAL_SUPPORTED_MODELS))
)


OpenRouterChatProviderFactory = Callable[[str], tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]]


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
    api_key: str
    base_url: str = OPENROUTER_BASE_URL
    timeout_seconds: float = 30.0
    dimensions: int | None = None
    client: httpx.AsyncClient | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        normalized_model = self.model.strip()
        if normalized_model not in OPENROUTER_EMBEDDING_SUPPORTED_MODELS:
            raise ValueError(f"OpenRouter embedding provider does not support model {self.model!r}")
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided for embeddings")
        self.model = normalized_model
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("OpenRouter embedding base_url must not be empty")
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                headers={"Authorization": f"Bearer {self.api_key}"},
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
                raise RuntimeError(
                    f"embedding dimensions mismatch: expected={self.dimensions} actual={len(vector)}"
                )
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
    ) -> None:
        self._openrouter_api_key = openrouter_api_key
        self._openrouter_chat_provider_factory = openrouter_chat_provider_factory or build_openrouter_chat_provider
        self._openrouter_provider: OpenAiCompatibleLlmProvider | None = None
        self._openrouter_client: httpx.AsyncClient | None = None

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        model = request.model.strip()
        if model not in OPENROUTER_SUPPORTED_MODELS:
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
        extra = _merge_reasoning_extra(extra, request.thinking)
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
    metadata = payload.get("openrouter_metadata")
    if not isinstance(metadata, Mapping):
        return None
    return {"openrouter_metadata": dict(metadata)}


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
