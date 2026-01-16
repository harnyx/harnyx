"""Unified request adapter for LLM providers.

All LLM calls should flow through this adapter so provider-specific quirks are
handled in one place (model aliasing, output-mode compatibility, etc.).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from pydantic import BaseModel

from caster_commons.llm.provider import LlmProviderPort
from caster_commons.llm.schema import AbstractLlmRequest, LlmMessage, LlmMessageContentPart, LlmResponse

# Translate specific OSS ids when routed through Vertex MaaS. We allow both
# provider labels so callers can distinguish regions (e.g., vertex-maas for
# us-central1) while keeping backward compatibility with plain vertex.
_DEFAULT_MODEL_ALIASES: Mapping[str, str] = {
    "vertex:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
    "vertex:openai/gpt-oss-120b": "publishers/openai/models/gpt-oss-120b-maas",
    "vertex-maas:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
    "vertex-maas:openai/gpt-oss-120b": "publishers/openai/models/gpt-oss-120b-maas",
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
        adapted_request = _adapt_output_mode(provider, request)
        adapted_request = _adapt_model_aliases(provider, adapted_request, self._model_aliases)
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
    resolved = aliases.get(provider_key) or aliases.get(global_key)
    if resolved is None or resolved == model:
        return request
    return replace(request, model=resolved)


def _adapt_output_mode(provider: str, request: AbstractLlmRequest) -> AbstractLlmRequest:
    if provider != "chutes":
        return request
    if request.output_mode == "text":
        return request

    instruction = _json_only_instruction(request.output_schema)
    messages = (LlmMessage(role="system", content=(LlmMessageContentPart.input_text(instruction),)),) + tuple(
        request.messages
    )
    return replace(
        request,
        output_mode="text",
        output_schema=None,
        messages=messages,
    )


def _json_only_instruction(schema: type[BaseModel] | None) -> str:
    if schema is None:
        return "Reply with JSON only. Do not wrap in Markdown."
    required = _required_keys(schema)
    if required:
        keys = ", ".join(required)
        return f"Reply with JSON only. Do not wrap in Markdown. Output keys: {keys}."
    return "Reply with JSON only. Do not wrap in Markdown."


def _required_keys(schema: type[BaseModel]) -> Sequence[str]:
    json_schema = schema.model_json_schema()
    required = json_schema.get("required") if isinstance(json_schema, dict) else None
    if isinstance(required, list) and all(isinstance(item, str) for item in required):
        return tuple(required)
    return tuple(schema.model_fields.keys())


__all__ = [
    "LlmProviderAdapter",
]
