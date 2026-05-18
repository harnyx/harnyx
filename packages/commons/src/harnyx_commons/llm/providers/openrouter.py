"""Hardcoded OpenRouter provider for repo-owned OpenRouter routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import httpx
from pydantic import SecretStr

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig, OpenRouterModelProviderOptions
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_types import OPENROUTER_PROVIDER
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse, LlmThinkingConfig

OPENROUTER_ENDPOINT_ID = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_SUPPORTED_MODELS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3.6-27B-TEE",
)
OPENROUTER_MODEL_ALIASES: Mapping[str, str] = {
    "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
}


OpenRouterChatProviderFactory = Callable[[str], tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]]


class OpenRouterLlmProvider(LlmProviderPort):
    def __init__(
        self,
        *,
        openrouter_api_key: SecretStr,
        model_provider_options: Mapping[str, OpenRouterModelProviderOptions],
        openrouter_chat_provider_factory: OpenRouterChatProviderFactory | None = None,
    ) -> None:
        unknown_models = set(model_provider_options) - set(OPENROUTER_SUPPORTED_MODELS)
        if unknown_models:
            unknown = ", ".join(sorted(unknown_models))
            raise ValueError(f"OpenRouter provider options configured for unsupported models: {unknown}")
        self._openrouter_api_key = openrouter_api_key
        self._model_provider_options = dict(model_provider_options)
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
            raise ValueError(f"OPENROUTER_API_KEY must be configured to use OpenRouter model {model}")
        provider, client = self._openrouter_chat_provider_factory(normalized_key)
        self._openrouter_provider = provider
        self._openrouter_client = client
        return provider

    def _openrouter_request(self, request: AbstractLlmRequest, *, model: str) -> AbstractLlmRequest:
        extra = _merge_extra(request.extra, self._model_provider_options.get(model))
        extra = _merge_reasoning_extra(extra, request.thinking)
        return replace(
            request,
            provider=OPENROUTER_PROVIDER,
            model=OPENROUTER_MODEL_ALIASES.get(model, model),
            extra=extra or None,
        )


def build_openrouter_chat_provider(api_key: str) -> tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]:
    normalized_key = api_key.strip()
    if not normalized_key:
        raise ValueError("OPENROUTER_API_KEY must be configured to build OpenRouter provider")
    client = httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL,
        headers={"Authorization": f"Bearer {normalized_key}"},
    )
    endpoint = OpenAiCompatibleEndpointConfig.model_validate(
        {
            "id": OPENROUTER_ENDPOINT_ID,
            "base_url": OPENROUTER_BASE_URL,
            "auth": {"type": "none"},
        }
    )
    return OpenAiCompatibleLlmProvider(endpoint=endpoint, client=client), client


def _merge_extra(
    request_extra: Mapping[str, Any] | None,
    options: OpenRouterModelProviderOptions | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(request_extra or {})
    if options is None:
        return merged
    options_extra = dict(options.to_request_extra())
    provider_options = options_extra.get("provider")
    if provider_options is None:
        return merged
    if not isinstance(provider_options, Mapping):
        raise AssertionError("OpenRouter provider options must serialize provider as an object")
    request_provider = merged.get("provider")
    if request_provider is not None and not isinstance(request_provider, Mapping):
        raise ValueError("OpenRouter request extra.provider must be an object")
    merged_provider = dict(request_provider or {})
    merged_provider.update(dict(provider_options))
    merged["provider"] = merged_provider
    return merged


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
    "OPENROUTER_ENDPOINT_ID",
    "OPENROUTER_MODEL_ALIASES",
    "OPENROUTER_SUPPORTED_MODELS",
    "OpenRouterLlmProvider",
    "build_openrouter_chat_provider",
]
