"""Shared type aliases for LLM providers.

Kept separate from provider implementations to avoid import cycles with settings modules.
"""

from __future__ import annotations

from typing import Literal, cast

BEDROCK_PROVIDER = "bedrock"
CHUTES_PROVIDER = "chutes"
VERTEX_PROVIDER = "vertex"
CUSTOM_OPENAI_COMPATIBLE_PROVIDER_TAG = "custom-openai-compatible"
OPENROUTER_PROVIDER = "openrouter"
AI_GATEWAY_PROVIDER = "ai_gateway"

LlmProviderName = Literal["bedrock", "chutes", "vertex"]
LlmRouteTarget = str

ALLOWED_LLM_PROVIDERS: tuple[LlmProviderName, ...] = (
    BEDROCK_PROVIDER,
    CHUTES_PROVIDER,
    VERTEX_PROVIDER,
)


def parse_builtin_provider_name(raw: str | None, *, component: str) -> LlmProviderName:
    if raw is None:
        raise ValueError(f"{component} llm provider must be specified")
    value = raw.strip()
    if not value or value not in ALLOWED_LLM_PROVIDERS:
        raise ValueError(f"{component} llm provider {value!r} is not allowed")
    return cast(LlmProviderName, value)


def custom_openai_compatible_target(endpoint_id: str) -> LlmRouteTarget:
    normalized = endpoint_id.strip()
    if not normalized:
        raise ValueError("custom OpenAI-compatible endpoint id must be non-empty")
    return f"{CUSTOM_OPENAI_COMPATIBLE_PROVIDER_TAG}:{normalized}"


def parse_custom_openai_compatible_target(raw: str) -> str | None:
    value = raw.strip()
    prefix = f"{CUSTOM_OPENAI_COMPATIBLE_PROVIDER_TAG}:"
    if not value.startswith(prefix):
        return None
    endpoint_id = value.removeprefix(prefix).strip()
    if not endpoint_id:
        raise ValueError("custom OpenAI-compatible endpoint id must be non-empty")
    return endpoint_id


def parse_provider_route_target(raw: str | None, *, component: str) -> LlmRouteTarget:
    if raw is None:
        raise ValueError(f"{component} llm provider route target must be specified")
    value = raw.strip()
    if not value:
        raise ValueError(f"{component} llm provider route target must be non-empty")
    custom_endpoint_id = parse_custom_openai_compatible_target(value)
    if custom_endpoint_id is not None:
        return custom_openai_compatible_target(custom_endpoint_id)
    if value == OPENROUTER_PROVIDER:
        return OPENROUTER_PROVIDER
    return parse_builtin_provider_name(value, component=component)


def normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    normalized = reasoning_effort.strip()
    if not normalized:
        return None
    try:
        if int(normalized) <= 0:
            return None
    except ValueError:
        return normalized
    return normalized


__all__ = [
    "ALLOWED_LLM_PROVIDERS",
    "AI_GATEWAY_PROVIDER",
    "BEDROCK_PROVIDER",
    "CHUTES_PROVIDER",
    "CUSTOM_OPENAI_COMPATIBLE_PROVIDER_TAG",
    "LlmProviderName",
    "LlmRouteTarget",
    "OPENROUTER_PROVIDER",
    "VERTEX_PROVIDER",
    "custom_openai_compatible_target",
    "normalize_reasoning_effort",
    "parse_builtin_provider_name",
    "parse_custom_openai_compatible_target",
    "parse_provider_route_target",
]
