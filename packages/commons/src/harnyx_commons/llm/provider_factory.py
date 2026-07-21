"""Shared cached LLM provider construction for platform and validator."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import SecretStr

from harnyx_commons.clients import CHUTES
from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.errors import ProviderCredentialUnavailableError
from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.provider import LlmProviderName, LlmProviderPort
from harnyx_commons.llm.provider_types import (
    AI_GATEWAY_PROVIDER,
    OPENROUTER_PROVIDER,
    parse_builtin_provider_name,
    parse_custom_openai_compatible_target,
    parse_provider_route_target,
)
from harnyx_commons.llm.providers.ai_gateway import AiGatewayLlmProvider
from harnyx_commons.llm.providers.bedrock import BedrockLlmProvider
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.providers.openrouter import OpenRouterLlmProvider
from harnyx_commons.llm.providers.vertex.provider import VertexLlmProvider
from harnyx_commons.llm.routing import LlmRouteSurface, RoutedLlmProvider
from harnyx_commons.llm.tool_models import MinerSelectedLlmProviderName, parse_miner_selected_llm_provider


class CachedLlmProviderRegistry:
    def __init__(
        self,
        *,
        llm_settings: LlmSettings,
        bedrock_settings: BedrockSettings,
        vertex_settings: VertexSettings,
    ) -> None:
        self._llm_settings = llm_settings
        self._bedrock_settings = bedrock_settings
        self._vertex_settings = vertex_settings
        self._cache: dict[str, LlmProviderPort] = {}

    def resolve(self, name: str) -> LlmProviderPort:
        route_target = _parse_registry_route_target(name)
        provider = self._cache.get(route_target)
        if provider is None:
            provider = _build_provider(
                route_target=route_target,
                llm_settings=self._llm_settings,
                bedrock_settings=self._bedrock_settings,
                vertex_settings=self._vertex_settings,
            )
            self._cache[route_target] = provider
        return provider

    async def aclose(self) -> None:
        errors: list[Exception] = []
        for provider_name, provider in self._cache.items():
            try:
                await provider.aclose()
            except Exception as exc:
                exc.add_note(f"cached llm provider close failed: {provider_name}")
                errors.append(exc)
        if errors:
            raise ExceptionGroup("cached llm provider cleanup failed", errors)


def build_cached_llm_provider_registry(
    *,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
) -> CachedLlmProviderRegistry:
    """Return a cached provider registry that applies shared concurrency settings."""

    return CachedLlmProviderRegistry(
        llm_settings=llm_settings,
        bedrock_settings=bedrock_settings,
        vertex_settings=vertex_settings,
    )


def build_cached_llm_provider_resolver(
    *,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
) -> Callable[[str], LlmProviderPort]:
    registry = build_cached_llm_provider_registry(
        llm_settings=llm_settings,
        bedrock_settings=bedrock_settings,
        vertex_settings=vertex_settings,
    )
    return registry.resolve


def build_routed_llm_provider(
    *,
    surface: LlmRouteSurface,
    default_provider: LlmProviderName,
    llm_settings: LlmSettings,
    allowed_providers: set[LlmProviderName],
    provider_registry: CachedLlmProviderRegistry,
    allow_custom_openai_compatible: bool = False,
) -> RoutedLlmProvider:
    return RoutedLlmProvider(
        surface=surface,
        default_provider=default_provider,
        overrides=llm_settings.llm_model_provider_overrides,
        allowed_providers=allowed_providers,
        allow_custom_openai_compatible=allow_custom_openai_compatible,
        resolve_provider=provider_registry.resolve,
    )


def build_miner_paid_llm_provider(
    *,
    provider: MinerSelectedLlmProviderName | str,
    api_key: SecretStr | str,
    llm_settings: LlmSettings,
) -> LlmProviderPort:
    """Build an uncached miner-paid LLM provider from an explicit miner credential."""

    provider_name = parse_miner_selected_llm_provider(provider)
    explicit_key = _explicit_api_key_value(api_key, provider=provider_name)
    if provider_name == "chutes":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=ChutesLlmProvider(
                base_url=CHUTES.base_url,
                api_key=explicit_key,
                timeout=CHUTES.timeout_seconds,
                max_concurrent=None,
            ),
        )
    if provider_name == "openrouter":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=OpenRouterLlmProvider(
                openrouter_api_key=SecretStr(explicit_key),
            ),
        )
    if provider_name == AI_GATEWAY_PROVIDER:
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=AiGatewayLlmProvider(
                ai_gateway_api_key=SecretStr(explicit_key),
            ),
        )
    raise AssertionError(f"unsupported parsed miner-paid llm provider: {provider_name}")


def _build_provider(
    *,
    route_target: str,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
) -> LlmProviderPort:
    if route_target == OPENROUTER_PROVIDER:
        _require_configured_credential(route_target, llm_settings.openrouter_api_key_value)
        return LlmProviderAdapter(
            provider_name=route_target,
            delegate=OpenRouterLlmProvider(
                openrouter_api_key=llm_settings.openrouter_api_key,
            ),
        )

    if route_target == AI_GATEWAY_PROVIDER:
        _require_configured_credential(route_target, llm_settings.ai_gateway_api_key_value)
        return LlmProviderAdapter(
            provider_name=route_target,
            delegate=AiGatewayLlmProvider(
                ai_gateway_api_key=llm_settings.ai_gateway_api_key,
            ),
        )

    custom_endpoint_id = parse_custom_openai_compatible_target(route_target)
    if custom_endpoint_id is not None:
        endpoints = llm_settings.openai_compatible_endpoints
        endpoint = endpoints.get(custom_endpoint_id)
        if endpoint is None:
            raise ValueError(f"custom OpenAI-compatible endpoint '{custom_endpoint_id}' is not configured")
        return LlmProviderAdapter(
            provider_name=route_target,
            delegate=OpenAiCompatibleLlmProvider(endpoint=endpoint),
        )

    provider_name = parse_builtin_provider_name(route_target, component="shared")
    max_concurrent = _max_concurrent_for_provider(provider_name, llm_settings)

    if provider_name == "bedrock":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=BedrockLlmProvider(
                region=bedrock_settings.region_value,
                connect_timeout_seconds=bedrock_settings.connect_timeout_seconds,
                read_timeout_seconds=bedrock_settings.read_timeout_seconds,
                max_concurrent=max_concurrent,
            ),
        )

    if provider_name == "chutes":
        api_key = llm_settings.chutes_api_key_value.strip()
        _require_configured_credential(provider_name, api_key)
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=ChutesLlmProvider(
                base_url=CHUTES.base_url,
                api_key=api_key,
                timeout=CHUTES.timeout_seconds,
                max_concurrent=max_concurrent,
            ),
        )

    if provider_name == "vertex":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=VertexLlmProvider(
                project=vertex_settings.gcp_project_id,
                location=vertex_settings.gcp_location,
                timeout=vertex_settings.vertex_timeout_seconds,
                service_account_b64=vertex_settings.gcp_sa_credential_b64_value,
                max_concurrent=max_concurrent,
            ),
        )

    raise ValueError(f"unsupported llm provider: {provider_name}")


def _parse_registry_route_target(raw: str | None) -> str:
    value = (raw or "").strip()
    if value in {OPENROUTER_PROVIDER, AI_GATEWAY_PROVIDER}:
        return value
    return parse_provider_route_target(raw, component="shared")


def _explicit_api_key_value(api_key: SecretStr | str, *, provider: str) -> str:
    value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{provider} miner-paid API key must be provided")
    return normalized


def _require_configured_credential(provider: str, api_key: str) -> None:
    if not api_key.strip():
        raise ProviderCredentialUnavailableError(provider)


def _max_concurrent_for_provider(
    provider_name: LlmProviderName,
    llm_settings: LlmSettings,
) -> int:
    if provider_name == "bedrock":
        return llm_settings.bedrock_max_concurrent
    if provider_name == "vertex":
        return llm_settings.vertex_max_concurrent
    if provider_name == "chutes":
        return llm_settings.chutes_max_concurrent
    raise ValueError(f"unsupported llm provider: {provider_name}")


__all__ = [
    "CachedLlmProviderRegistry",
    "build_cached_llm_provider_registry",
    "build_cached_llm_provider_resolver",
    "build_miner_paid_llm_provider",
    "build_routed_llm_provider",
]
