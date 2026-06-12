"""Client wiring for sandboxed tool invocation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import SecretStr

from harnyx_commons.clients import DESEARCH, PARALLEL
from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings, SearchProviderName, parse_search_provider_name
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import (
    CachedLlmProviderRegistry,
    build_cached_llm_provider_registry,
    build_routed_llm_provider,
)
from harnyx_commons.llm.provider_types import BEDROCK_PROVIDER
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_provider_timeout_seconds
from harnyx_commons.tools.desearch import DeSearchClient
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.ports import WebSearchProviderPort
from harnyx_commons.tools.provider_billing import SearchProviderResult
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)


@dataclass(frozen=True, slots=True)
class ToolInvocationClients:
    search_client: WebSearchProviderPort | None
    llm_provider_registry: CachedLlmProviderRegistry
    tool_llm_provider: LlmProviderPort | None


def build_tool_invocation_clients(
    *,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
    lazy_search: bool = True,
    require_search: bool = False,
) -> ToolInvocationClients:
    validate_tool_invocation_provider_policy(llm_settings)
    provider_registry = build_cached_llm_provider_registry(
        llm_settings=llm_settings,
        bedrock_settings=bedrock_settings,
        vertex_settings=vertex_settings,
    )
    return ToolInvocationClients(
        search_client=_build_optional_search_client(
            llm_settings,
            lazy=lazy_search,
            required=require_search,
        ),
        llm_provider_registry=provider_registry,
        tool_llm_provider=build_optional_tool_llm_provider(llm_settings, provider_registry),
    )


def validate_tool_invocation_provider_policy(llm_settings: LlmSettings) -> None:
    if llm_settings.tool_llm_provider == BEDROCK_PROVIDER:
        raise ValueError("TOOL_LLM_PROVIDER='bedrock' is not supported")
    for provider_name in llm_settings.llm_model_provider_overrides.get("tool", {}).values():
        if provider_name == BEDROCK_PROVIDER:
            raise ValueError("TOOL_LLM_PROVIDER='bedrock' is not supported")


def build_optional_tool_llm_provider(
    llm_settings: LlmSettings,
    provider_registry: CachedLlmProviderRegistry,
) -> LlmProviderPort | None:
    if llm_settings.tool_llm_provider is None:
        return None
    return LazyLlmProvider(lambda: build_tool_llm_provider(llm_settings, provider_registry))


def build_tool_llm_provider(
    llm_settings: LlmSettings,
    provider_registry: CachedLlmProviderRegistry,
) -> LlmProviderPort:
    return build_routed_llm_provider(
        surface="tool",
        default_provider=llm_settings.tool_llm_provider,
        llm_settings=llm_settings,
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=provider_registry,
    )


def build_web_search_provider(llm_settings: LlmSettings) -> WebSearchProviderPort:
    provider = llm_settings.search_provider
    if provider is None:
        raise RuntimeError("SEARCH_PROVIDER must be configured")
    if provider == "desearch":
        return DeSearchClient(
            base_url=DESEARCH.base_url,
            api_key=llm_settings.desearch_api_key_value,
            timeout=DESEARCH.timeout_seconds,
            max_concurrent=llm_settings.desearch_max_concurrent,
        )
    if provider == "parallel":
        return ParallelClient(
            base_url=llm_settings.parallel_base_url,
            api_key=llm_settings.parallel_api_key_value,
            timeout=PARALLEL.timeout_seconds,
            max_concurrent=llm_settings.parallel_max_concurrent,
        )
    raise ValueError(f"unsupported search provider: {provider}")


def build_miner_paid_web_search_provider(
    *,
    provider: SearchProviderName | str,
    api_key: SecretStr | str,
    llm_settings: LlmSettings,
    timeout: float | None = None,
) -> WebSearchProviderPort:
    """Build an uncached miner-paid search provider from an explicit miner credential."""

    provider_name = parse_search_provider_name(provider)
    explicit_key = _explicit_api_key_value(api_key, provider=provider_name)
    if provider_name == "desearch":
        return DeSearchClient(
            base_url=DESEARCH.base_url,
            api_key=explicit_key,
            timeout=_effective_client_timeout(DESEARCH.timeout_seconds, timeout),
            max_concurrent=None,
        )
    if provider_name == "parallel":
        return ParallelClient(
            base_url=llm_settings.parallel_base_url,
            api_key=explicit_key,
            timeout=_effective_client_timeout(PARALLEL.timeout_seconds, timeout),
            max_concurrent=None,
        )
    raise AssertionError(f"unsupported parsed miner-paid search provider: {provider_name}")


def _build_optional_search_client(
    llm_settings: LlmSettings,
    *,
    lazy: bool,
    required: bool,
) -> WebSearchProviderPort | None:
    if llm_settings.search_provider is None:
        if required:
            raise RuntimeError("SEARCH_PROVIDER must be configured")
        return None
    if not lazy:
        return build_web_search_provider(llm_settings)
    return LazySearchProvider(lambda: build_web_search_provider(llm_settings))


def _explicit_api_key_value(api_key: SecretStr | str, *, provider: str) -> str:
    value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{provider} miner-paid API key must be provided")
    return normalized


def _effective_client_timeout(default_timeout: float, requested_timeout: float | None) -> float:
    if requested_timeout is None:
        return default_timeout
    return max(default_timeout, platform_tool_proxy_provider_timeout_seconds(requested_timeout))


class LazyLlmProvider(LlmProviderPort):
    def __init__(self, factory: Callable[[], LlmProviderPort]) -> None:
        self._factory = factory
        self._provider: LlmProviderPort | None = None
        self._lock = asyncio.Lock()

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        provider = await self._get_provider()
        return await provider.invoke(request)

    async def aclose(self) -> None:
        provider = self._provider
        if provider is not None:
            await provider.aclose()

    async def _get_provider(self) -> LlmProviderPort:
        provider = self._provider
        if provider is not None:
            return provider
        async with self._lock:
            provider = self._provider
            if provider is None:
                provider = self._factory()
                self._provider = provider
        return provider


class LazySearchProvider(WebSearchProviderPort):
    def __init__(self, factory: Callable[[], WebSearchProviderPort]) -> None:
        self._factory = factory
        self._provider: WebSearchProviderPort | None = None
        self._lock = asyncio.Lock()

    async def search_web(
        self,
        request: SearchWebSearchRequest,
    ) -> SearchProviderResult[SearchWebSearchResponse]:
        provider = await self._get_provider()
        return await provider.search_web(request)

    async def search_ai(
        self,
        request: SearchAiSearchRequest,
    ) -> SearchProviderResult[SearchAiSearchResponse]:
        provider = await self._get_provider()
        return await provider.search_ai(request)

    async def fetch_page(
        self,
        request: FetchPageRequest,
    ) -> SearchProviderResult[FetchPageResponse]:
        provider = await self._get_provider()
        return await provider.fetch_page(request)

    async def aclose(self) -> None:
        provider = self._provider
        if provider is not None:
            await provider.aclose()

    async def _get_provider(self) -> WebSearchProviderPort:
        provider = self._provider
        if provider is not None:
            return provider
        async with self._lock:
            provider = self._provider
            if provider is None:
                provider = self._factory()
                self._provider = provider
        return provider


__all__ = [
    "LazyLlmProvider",
    "LazySearchProvider",
    "ToolInvocationClients",
    "build_miner_paid_web_search_provider",
    "build_optional_tool_llm_provider",
    "build_tool_invocation_clients",
    "build_tool_llm_provider",
    "build_web_search_provider",
    "validate_tool_invocation_provider_policy",
]
