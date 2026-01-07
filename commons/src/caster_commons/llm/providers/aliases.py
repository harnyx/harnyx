"""Helpers for translating user-facing LLM model names."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from caster_commons.llm.provider import LlmProviderPort
from caster_commons.llm.schema import AbstractLlmRequest, LlmResponse

# Translate specific OSS ids when routed through Vertex MaaS. We allow both
# provider labels so callers can distinguish regions (e.g., vertex-maas for
# us-central1) while keeping backward compatibility with plain vertex.
DEFAULT_MODEL_ALIASES: Mapping[str, str] = {
    "vertex:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
    "vertex:openai/gpt-oss-120b": "publishers/openai/models/gpt-oss-120b-maas",
    "vertex-maas:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
    "vertex-maas:openai/gpt-oss-120b": "publishers/openai/models/gpt-oss-120b-maas",
}


class LlmModelAliasResolver:
    """Resolves friendly model ids to provider-specific targets."""

    def __init__(self, aliases: Mapping[str, str]) -> None:
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
        self._aliases = normalized

    def resolve(self, model: str, *, provider: str | None = None) -> str:
        if not model:
            return model
        normalized_model = model.strip()
        if not normalized_model:
            return model
        candidates: list[str] = []
        if provider:
            provider_key = provider.strip().lower()
            if provider_key:
                candidates.append(f"{provider_key}:{normalized_model.lower()}")
        candidates.append(normalized_model.lower())
        for candidate in candidates:
            resolved = self._aliases.get(candidate)
            if resolved:
                return resolved
        return model

    def has_aliases(self) -> bool:
        return bool(self._aliases)


class AliasingLlmProvider(LlmProviderPort):
    """Wraps another provider and rewrites model ids using the resolver."""

    def __init__(
        self,
        *,
        provider_name: str,
        delegate: LlmProviderPort,
        resolver: LlmModelAliasResolver,
    ) -> None:
        self._provider_name = provider_name
        self._delegate = delegate
        self._resolver = resolver

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        provider = request.provider or self._provider_name
        resolved_model = self._resolver.resolve(request.model, provider=provider)
        if resolved_model == request.model:
            return await self._delegate.invoke(request)
        updated_request = replace(request, model=resolved_model)
        return await self._delegate.invoke(updated_request)

    async def aclose(self) -> None:
        # delegate should have been closely separated from this provider, so we don't need to close it here
        pass


__all__ = [
    "AliasingLlmProvider",
    "DEFAULT_MODEL_ALIASES",
    "LlmModelAliasResolver",
]
