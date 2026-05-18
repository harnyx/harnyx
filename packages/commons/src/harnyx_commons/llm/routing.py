"""Shared provider-route resolution for LLM surfaces."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, cast

from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_types import (
    CHUTES_PROVIDER,
    OPENROUTER_PROVIDER,
    LlmProviderName,
    LlmRouteTarget,
    parse_custom_openai_compatible_target,
    parse_provider_route_target,
)
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse

LlmRouteSurface = Literal["generator", "digest", "reference", "content_review", "tool", "scoring"]
LlmModelProviderOverrides = dict[LlmRouteSurface, dict[str, LlmRouteTarget]]
OPENROUTER_ROUTED_TOOL_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3.6-27B-TEE",
)

_ALLOWED_ROUTE_SURFACES: tuple[LlmRouteSurface, ...] = (
    "generator",
    "digest",
    "reference",
    "content_review",
    "tool",
    "scoring",
)


@dataclass(frozen=True, slots=True)
class ResolvedLlmRoute:
    surface: LlmRouteSurface
    provider: LlmRouteTarget
    model: str


def parse_llm_model_provider_overrides(
    raw: str | None,
    *,
    custom_openai_compatible_endpoint_ids: set[str] | frozenset[str] = frozenset(),
) -> LlmModelProviderOverrides:
    if raw is None:
        return {}
    normalized_raw = raw.strip()
    if not normalized_raw:
        return {}
    try:
        payload = json.loads(normalized_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM_MODEL_PROVIDER_OVERRIDES_JSON must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM_MODEL_PROVIDER_OVERRIDES_JSON must decode to a JSON object")

    overrides: LlmModelProviderOverrides = {}
    for surface_raw, models_raw in payload.items():
        surface = _parse_route_surface(surface_raw)
        if not isinstance(models_raw, dict):
            raise ValueError(
                f"LLM_MODEL_PROVIDER_OVERRIDES_JSON.{surface} must decode to a JSON object "
                "of model-to-provider mappings"
            )
        model_overrides: dict[str, LlmRouteTarget] = {}
        for model_raw, provider_raw in models_raw.items():
            if not isinstance(model_raw, str):
                raise ValueError(f"LLM_MODEL_PROVIDER_OVERRIDES_JSON.{surface} model keys must be strings")
            model = model_raw.strip()
            if not model:
                raise ValueError(f"LLM_MODEL_PROVIDER_OVERRIDES_JSON.{surface} model keys must be non-empty")
            if model in model_overrides:
                raise ValueError(f"LLM_MODEL_PROVIDER_OVERRIDES_JSON.{surface}.{model} is duplicated")
            if not isinstance(provider_raw, str):
                raise ValueError(
                    f"LLM_MODEL_PROVIDER_OVERRIDES_JSON.{surface}.{model} provider labels must be strings"
                )
            route_target = parse_provider_route_target(
                provider_raw,
                component=f"LLM_MODEL_PROVIDER_OVERRIDES_JSON.{surface}.{model}",
            )
            _validate_custom_target_exists(
                route_target,
                custom_openai_compatible_endpoint_ids=custom_openai_compatible_endpoint_ids,
                component=f"LLM_MODEL_PROVIDER_OVERRIDES_JSON.{surface}.{model}",
            )
            model_overrides[model] = route_target
        if model_overrides:
            overrides[surface] = model_overrides
    return overrides


def resolve_llm_route(
    *,
    surface: LlmRouteSurface,
    default_provider: LlmProviderName,
    model: str,
    overrides: LlmModelProviderOverrides,
    allowed_providers: set[LlmProviderName],
    allow_custom_openai_compatible: bool = False,
) -> ResolvedLlmRoute:
    normalized_model = model.strip()
    override_provider = overrides.get(surface, {}).get(normalized_model)
    if override_provider is None:
        selected_provider: LlmRouteTarget = default_provider
        return _route_resolved_provider(surface=surface, provider=selected_provider, model=normalized_model)
    custom_endpoint_id = parse_custom_openai_compatible_target(override_provider)
    if custom_endpoint_id is not None:
        if not allow_custom_openai_compatible:
            raise ValueError(f"{surface} override provider {override_provider!r} is not supported")
        return _route_resolved_provider(surface=surface, provider=override_provider, model=normalized_model)
    if override_provider not in allowed_providers:
        raise ValueError(f"{surface} override provider {override_provider!r} is not supported")
    return _route_resolved_provider(surface=surface, provider=override_provider, model=normalized_model)


def _route_resolved_provider(
    *,
    surface: LlmRouteSurface,
    provider: LlmRouteTarget,
    model: str,
) -> ResolvedLlmRoute:
    if surface == "tool" and provider == CHUTES_PROVIDER and model in OPENROUTER_ROUTED_TOOL_MODELS:
        return ResolvedLlmRoute(surface=surface, provider=OPENROUTER_PROVIDER, model=model)
    return ResolvedLlmRoute(surface=surface, provider=provider, model=model)


def with_effective_route_metadata(response: LlmResponse, route: ResolvedLlmRoute) -> LlmResponse:
    metadata = dict(response.metadata or {})
    metadata.setdefault("effective_provider", route.provider)
    metadata.setdefault("effective_model", route.model)
    metadata["selected_provider"] = route.provider
    metadata["selected_model"] = route.model
    return replace(response, metadata=metadata)


class RoutedLlmProvider(LlmProviderPort):
    def __init__(
        self,
        *,
        surface: LlmRouteSurface,
        default_provider: LlmProviderName,
        overrides: LlmModelProviderOverrides,
        allowed_providers: set[LlmProviderName],
        allow_custom_openai_compatible: bool = False,
        resolve_provider: Callable[[str], LlmProviderPort],
    ) -> None:
        self._surface = surface
        self._default_provider = default_provider
        self._overrides = overrides
        self._allowed_providers = allowed_providers
        self._allow_custom_openai_compatible = allow_custom_openai_compatible
        self._resolve_provider = resolve_provider

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        route = resolve_llm_route(
            surface=self._surface,
            default_provider=self._default_provider,
            model=request.model,
            overrides=self._overrides,
            allowed_providers=self._allowed_providers,
            allow_custom_openai_compatible=self._allow_custom_openai_compatible,
        )
        routed_request = replace(request, provider=route.provider, model=route.model)
        response = await self._resolve_provider(route.provider).invoke(routed_request)
        return with_effective_route_metadata(response, route)

    async def aclose(self) -> None:
        return None


def _parse_route_surface(raw: object) -> LlmRouteSurface:
    if not isinstance(raw, str):
        raise ValueError("LLM_MODEL_PROVIDER_OVERRIDES_JSON surface keys must be strings")
    value = raw.strip()
    if value not in _ALLOWED_ROUTE_SURFACES:
        raise ValueError(f"LLM_MODEL_PROVIDER_OVERRIDES_JSON surface {value!r} is not supported")
    return cast(LlmRouteSurface, value)


def _validate_custom_target_exists(
    route_target: str,
    *,
    custom_openai_compatible_endpoint_ids: set[str] | frozenset[str],
    component: str,
) -> None:
    endpoint_id = parse_custom_openai_compatible_target(route_target)
    if endpoint_id is None:
        return
    if endpoint_id not in custom_openai_compatible_endpoint_ids:
        raise ValueError(f"{component} references unknown custom OpenAI-compatible endpoint {endpoint_id!r}")


__all__ = [
    "LlmModelProviderOverrides",
    "LlmRouteSurface",
    "ResolvedLlmRoute",
    "RoutedLlmProvider",
    "parse_llm_model_provider_overrides",
    "resolve_llm_route",
    "with_effective_route_metadata",
]
