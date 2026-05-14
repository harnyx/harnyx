"""Unified request adapter for LLM providers.

All LLM calls should flow through this adapter so provider-specific quirks are
handled in one place (model aliasing, output-mode compatibility, etc.).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse

_DEFAULT_MODEL_ALIASES: Mapping[str, str] = {
    "bedrock:openai/gpt-oss-20b-TEE": "openai.gpt-oss-20b-1:0",
    "bedrock:openai/gpt-oss-120b-TEE": "openai.gpt-oss-120b-1:0",
    "bedrock:MiniMaxAI/MiniMax-M2.5-TEE": "minimax.minimax-m2.5",
    "bedrock:moonshotai/Kimi-K2.5-TEE": "moonshotai.kimi-k2.5",
    "vertex:deepseek-ai/DeepSeek-V3.1-TEE": "deepseek-ai/deepseek-v3.1-maas",
    "vertex:deepseek-ai/DeepSeek-V3.2-TEE": "deepseek-ai/deepseek-v3.2-maas",
    "vertex:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
    "vertex:openai/gpt-oss-120b": "publishers/openai/models/gpt-oss-120b-maas",
    "vertex:zai-org/GLM-5-TEE": "zai-org/glm-5-maas",
    "vertex:Qwen/Qwen3-235B-A22B-Instruct-2507-TEE": "qwen3-235b-a22b-instruct-2507-maas",
    "vertex:Qwen/Qwen3-Next-80B-A3B-Instruct": "publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas",
    "custom-openai-compatible:gemma4-cloud-run-turbo:google/gemma-4-31B-turbo-TEE": (
        "nvidia/Gemma-4-31B-IT-NVFP4"
    ),
    "custom-openai-compatible:qwen36-cloud-run:Qwen/Qwen3.6-27B-TEE": "Qwen/Qwen3.6-27B-FP8",
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


def canonical_model_for_provider_model(
    *,
    provider_name: str,
    model: str,
    model_aliases: Mapping[str, str] = _DEFAULT_MODEL_ALIASES,
) -> str:
    normalized_provider = provider_name.strip().lower()
    normalized_model = model.strip()
    if not normalized_provider or not normalized_model:
        return normalized_model
    normalized_model_key = normalized_model.lower()
    for alias_key, provider_model in model_aliases.items():
        normalized_alias_key = alias_key.strip().lower()
        provider_prefix = f"{normalized_provider}:"
        if not normalized_alias_key.startswith(provider_prefix):
            continue
        if provider_model.strip().lower() != normalized_model_key:
            continue
        return alias_key[len(provider_prefix) :].strip()
    return normalized_model


__all__ = [
    "LlmProviderAdapter",
    "canonical_model_for_provider_model",
]
