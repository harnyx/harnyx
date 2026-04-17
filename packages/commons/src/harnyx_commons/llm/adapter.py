"""Unified request adapter for LLM providers.

All LLM calls should flow through this adapter so provider-specific quirks are
handled in one place (model aliasing, output-mode compatibility, etc.).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse

# Translate the public open-model ids only for the explicit Vertex MaaS
# contract. Plain `vertex` stays on the standard Vertex model surface.
_DEFAULT_MODEL_ALIASES: Mapping[str, str] = {
    "bedrock:openai/gpt-oss-20b-TEE": "openai.gpt-oss-20b-1:0",
    "bedrock:openai/gpt-oss-120b-TEE": "openai.gpt-oss-120b-1:0",
    "bedrock:moonshotai/Kimi-K2.5-TEE": "moonshotai.kimi-k2.5",
    "vertex-maas:openai/gpt-oss-20b-TEE": "publishers/openai/models/gpt-oss-20b-maas",
    "vertex-maas:openai/gpt-oss-120b-TEE": "publishers/openai/models/gpt-oss-120b-maas",
    "vertex-maas:Qwen/Qwen3-Next-80B-A3B-Instruct": "publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas",
}


class LlmProviderAdapter(LlmProviderPort):
    """Wraps another provider and applies per-provider request adaptations."""

    def __init__(
        self,
        *,
        provider_name: str,
        delegate: LlmProviderPort,
        model_aliases: Mapping[str, str] = _DEFAULT_MODEL_ALIASES,
    ) -> None:
        self._provider_name = provider_name
        self._delegate = delegate
        self._model_aliases = _normalize_aliases(model_aliases)

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        provider = (request.provider or self._provider_name).strip().lower()
        adapted_request = _adapt_model_aliases(provider, request, self._model_aliases)
        return await self._delegate.invoke(adapted_request)

    async def aclose(self) -> None:
        await self._delegate.aclose()


def _normalize_aliases(aliases: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in aliases.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("model aliases must use string keys and values")
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise ValueError("model alias key must be non-empty")
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("model alias value must be non-empty")
        normalized[normalized_key] = normalized_value
    return normalized


def _adapt_model_aliases(provider: str, request: AbstractLlmRequest, aliases: Mapping[str, str]) -> AbstractLlmRequest:
    model = request.model
    if not model:
        return request
    normalized_model = model.strip()
    if not normalized_model:
        return request

    provider_key = f"{provider}:{normalized_model}".lower()
    global_key = normalized_model.lower()
    if provider == "bedrock":
        resolved = aliases.get(provider_key)
    else:
        resolved = aliases.get(provider_key) or aliases.get(global_key)
    if resolved is None or resolved == model:
        return request
    return replace(request, model=resolved)


__all__ = [
    "LlmProviderAdapter",
]
