"""Client wiring for sandboxed tool invocation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import SecretStr

from harnyx_commons.clients import DESEARCH, PARALLEL
from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings, SearchProviderName, parse_search_provider_name
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.errors import ProviderCredentialUnavailableError
from harnyx_commons.llm.cost_settlement import normalized_provider_cost
from harnyx_commons.llm.pricing import MINER_TOOL_EMBEDDING_PRICING, price_embedding
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import (
    CachedLlmProviderRegistry,
    build_cached_llm_provider_registry,
    build_routed_llm_provider,
)
from harnyx_commons.llm.provider_types import BEDROCK_PROVIDER
from harnyx_commons.llm.providers.chutes import ChutesTextEmbeddingClient
from harnyx_commons.llm.providers.openrouter import OpenRouterEmbeddingClient
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_effective_provider_timeout_seconds
from harnyx_commons.tools.desearch import DeSearchClient
from harnyx_commons.tools.embedding_models import (
    QWEN3_DEFAULT_QUERY_INSTRUCTION,
    EmbeddingProviderName,
    EmbeddingUsage,
    EmbedTextRequest,
    EmbedTextResponse,
    TextEmbeddingResult,
    parse_miner_selected_embedding_provider,
)
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.ports import EmbeddingProviderPort, EmbeddingProviderResult, WebSearchProviderPort
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
    search_provider_registry: CachedWebSearchProviderRegistry
    llm_provider_registry: CachedLlmProviderRegistry
    tool_llm_provider: LlmProviderPort | None
    embedding_provider: EmbeddingProviderPort | None
    embedding_provider_registry: CachedEmbeddingProviderRegistry


def build_tool_invocation_clients(
    *,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
    lazy_search: bool = True,
    require_search: bool = False,
    build_routed_tool_llm_provider: bool = True,
) -> ToolInvocationClients:
    if build_routed_tool_llm_provider:
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
        search_provider_registry=CachedWebSearchProviderRegistry(llm_settings=llm_settings),
        llm_provider_registry=provider_registry,
        tool_llm_provider=(
            build_optional_tool_llm_provider(llm_settings, provider_registry)
            if build_routed_tool_llm_provider
            else None
        ),
        embedding_provider=build_optional_tool_embedding_provider(llm_settings),
        embedding_provider_registry=CachedEmbeddingProviderRegistry(llm_settings=llm_settings),
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


def build_optional_tool_embedding_provider(llm_settings: LlmSettings) -> EmbeddingProviderPort | None:
    provider = parse_miner_selected_embedding_provider(llm_settings.tool_embedding_provider)
    api_key = _embedding_api_key_value(provider=provider, llm_settings=llm_settings)
    if not api_key:
        return None
    return build_miner_paid_embedding_provider(
        provider=provider,
        api_key=api_key,
        llm_settings=llm_settings,
    )


def build_miner_paid_embedding_provider(
    *,
    provider: EmbeddingProviderName | str,
    api_key: SecretStr | str,
    llm_settings: LlmSettings,
    timeout: float | None = None,
) -> EmbeddingProviderPort:
    provider_name = _parse_embedding_provider(provider)
    explicit_key = _explicit_api_key_value(api_key, provider=provider_name)
    timeout_seconds = _effective_client_timeout(llm_settings.llm_timeout_seconds, timeout)
    match provider_name:
        case "chutes":
            return ChutesEmbeddingProvider(api_key=explicit_key, timeout_seconds=timeout_seconds)
        case "openrouter":
            return OpenRouterEmbeddingProvider(api_key=explicit_key, timeout_seconds=timeout_seconds)
    raise AssertionError(f"unsupported parsed embedding provider: {provider_name}")


class CachedWebSearchProviderRegistry:
    def __init__(self, *, llm_settings: LlmSettings, include_payloads_in_logs: bool = True) -> None:
        self._llm_settings = llm_settings
        self._include_payloads_in_logs = include_payloads_in_logs
        self._cache: dict[SearchProviderName, WebSearchProviderPort] = {}

    def resolve(self, provider: SearchProviderName | str) -> WebSearchProviderPort:
        provider_name = parse_search_provider_name(provider)
        search_provider = self._cache.get(provider_name)
        if search_provider is None:
            _require_configured_search_credential(provider=provider_name, llm_settings=self._llm_settings)
            search_provider = build_web_search_provider_for_name(
                self._llm_settings,
                provider_name,
                include_payloads_in_logs=self._include_payloads_in_logs,
            )
            self._cache[provider_name] = search_provider
        return search_provider

    async def aclose(self) -> None:
        errors: list[Exception] = []
        for provider_name, provider in self._cache.items():
            try:
                await provider.aclose()
            except Exception as exc:
                exc.add_note(f"cached search provider close failed: {provider_name}")
                errors.append(exc)
        if errors:
            raise ExceptionGroup("cached search provider cleanup failed", errors)


class CachedEmbeddingProviderRegistry:
    def __init__(self, *, llm_settings: LlmSettings) -> None:
        self._llm_settings = llm_settings
        self._cache: dict[EmbeddingProviderName, EmbeddingProviderPort] = {}

    def resolve(self, provider: EmbeddingProviderName | str) -> EmbeddingProviderPort:
        provider_name = parse_miner_selected_embedding_provider(provider)
        embedding_provider = self._cache.get(provider_name)
        if embedding_provider is None:
            api_key = _embedding_api_key_value(provider=provider_name, llm_settings=self._llm_settings)
            if not api_key.strip():
                raise ProviderCredentialUnavailableError(provider_name)
            embedding_provider = build_miner_paid_embedding_provider(
                provider=provider_name,
                api_key=api_key,
                llm_settings=self._llm_settings,
            )
            self._cache[provider_name] = embedding_provider
        return embedding_provider

    async def aclose(self) -> None:
        errors: list[Exception] = []
        for provider_name, provider in self._cache.items():
            try:
                await provider.aclose()
            except Exception as exc:
                exc.add_note(f"cached embedding provider close failed: {provider_name}")
                errors.append(exc)
        if errors:
            raise ExceptionGroup("cached embedding provider cleanup failed", errors)


def build_web_search_provider(llm_settings: LlmSettings) -> WebSearchProviderPort:
    if llm_settings.search_provider is None:
        raise RuntimeError("SEARCH_PROVIDER must be configured")
    return build_web_search_provider_for_name(llm_settings, llm_settings.search_provider)


def build_web_search_provider_for_name(
    llm_settings: LlmSettings,
    provider: SearchProviderName | str,
    *,
    include_payloads_in_logs: bool = True,
) -> WebSearchProviderPort:
    provider_name = parse_search_provider_name(provider)
    if provider_name == "desearch":
        return DeSearchClient(
            base_url=DESEARCH.base_url,
            api_key=llm_settings.desearch_api_key_value,
            timeout=DESEARCH.timeout_seconds,
            max_concurrent=llm_settings.desearch_max_concurrent,
            include_payloads_in_logs=include_payloads_in_logs,
        )
    if provider_name == "parallel":
        return ParallelClient(
            base_url=llm_settings.parallel_base_url,
            api_key=llm_settings.parallel_api_key_value,
            timeout=PARALLEL.timeout_seconds,
            max_concurrent=llm_settings.parallel_max_concurrent,
            include_payloads_in_logs=include_payloads_in_logs,
        )
    raise AssertionError(f"unsupported parsed search provider: {provider_name}")


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


def _embedding_api_key_value(*, provider: EmbeddingProviderName, llm_settings: LlmSettings) -> str:
    match provider:
        case "chutes":
            return llm_settings.chutes_api_key_value
        case "openrouter":
            return llm_settings.openrouter_api_key_value
    raise AssertionError(f"unsupported parsed embedding provider: {provider}")


def _require_configured_search_credential(*, provider: SearchProviderName, llm_settings: LlmSettings) -> None:
    api_key = (
        llm_settings.desearch_api_key_value if provider == "desearch" else llm_settings.parallel_api_key_value
    )
    if not api_key.strip():
        raise ProviderCredentialUnavailableError(provider)


def _effective_client_timeout(default_timeout: float, requested_timeout: float | None) -> float:
    return platform_tool_proxy_effective_provider_timeout_seconds(default_timeout, requested_timeout)


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


class ChutesEmbeddingProvider(EmbeddingProviderPort):
    def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._clients: dict[tuple[str, int | None], ChutesTextEmbeddingClient] = {}

    async def embed_text(
        self,
        request: EmbedTextRequest,
    ) -> EmbeddingProviderResult:
        if request.provider != "chutes":
            raise ValueError(f"embedding provider {request.provider!r} is not supported")
        client = self._client_for(model=request.model, dimensions=request.dimensions)
        started_at = time.perf_counter()
        provider_response = await client.embed_many(
            _format_embedding_texts(request),
            timeout_seconds=_effective_client_timeout(self._timeout_seconds, request.timeout),
        )
        elapsed_seconds = time.perf_counter() - started_at
        if len(provider_response.vectors) != len(request.texts):
            raise RuntimeError("embedding response count does not match request text count")

        usage = None
        if provider_response.usage is not None:
            usage = EmbeddingUsage(
                prompt_tokens=provider_response.usage.prompt_tokens,
                total_tokens=provider_response.usage.total_tokens,
            )

        cost_usd = price_embedding(request.provider, request.model, elapsed_seconds=elapsed_seconds)
        pricing = MINER_TOOL_EMBEDDING_PRICING[request.provider][request.model]
        response = EmbedTextResponse(
            provider=request.provider,
            model=request.model,
            input_type=request.input_type,
            data=[
                TextEmbeddingResult(index=index, embedding=list(vector))
                for index, vector in enumerate(provider_response.vectors)
            ],
            dimensions=len(provider_response.vectors[0]),
            usage=usage,
        )
        evidence = {
            "settlement_source": "static_pricing",
            "pricing_origin": "miner_tool_embedding_pricing",
            "provider": request.provider,
            "model": request.model,
            "input_type": request.input_type,
            "text_count": len(request.texts),
            "elapsed_seconds": elapsed_seconds,
            "usd_per_second": pricing.usd_per_second,
        }
        return EmbeddingProviderResult(
            response=response,
            actual_cost_usd=cost_usd,
            actual_cost_provider=request.provider,
            actual_cost_evidence=evidence,
        )

    async def aclose(self) -> None:
        errors: list[Exception] = []
        for key, client in self._clients.items():
            try:
                await client.aclose()
            except Exception as exc:
                exc.add_note(f"cached embedding provider close failed: {key}")
                errors.append(exc)
        if errors:
            raise ExceptionGroup("cached embedding provider cleanup failed", errors)

    def _client_for(self, *, model: str, dimensions: int | None) -> ChutesTextEmbeddingClient:
        key = (model, dimensions)
        client = self._clients.get(key)
        if client is None:
            client = ChutesTextEmbeddingClient(
                model=model,
                api_key=self._api_key,
                timeout_seconds=self._timeout_seconds,
                dimensions=dimensions,
            )
            self._clients[key] = client
        return client


class OpenRouterEmbeddingProvider(EmbeddingProviderPort):
    def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._clients: dict[tuple[str, int | None], OpenRouterEmbeddingClient] = {}

    async def embed_text(
        self,
        request: EmbedTextRequest,
    ) -> EmbeddingProviderResult:
        if request.provider != "openrouter":
            raise ValueError(f"embedding provider {request.provider!r} is not supported")
        client = self._client_for(model=request.model, dimensions=request.dimensions)
        formatted_texts = _format_embedding_texts(request)
        request_extra = request.provider_extra.to_request_extra() if request.provider_extra is not None else None
        timeout_seconds = _effective_client_timeout(self._timeout_seconds, request.timeout)
        if request_extra is None:
            provider_response = await client.embed_many(formatted_texts, timeout_seconds=timeout_seconds)
        else:
            provider_response = await client.embed_many(
                formatted_texts,
                extra=request_extra,
                timeout_seconds=timeout_seconds,
            )
        if len(provider_response.vectors) != len(request.texts):
            raise RuntimeError("embedding response count does not match request text count")

        provider_usage = provider_response.usage
        usage = (
            None
            if provider_usage is None
            else EmbeddingUsage(
                prompt_tokens=provider_usage.prompt_tokens,
                total_tokens=provider_usage.total_tokens,
            )
        )
        input_tokens = (
            None
            if usage is None
            else usage.prompt_tokens if usage.prompt_tokens is not None else usage.total_tokens
        )
        provider_cost_value = None if provider_usage is None else provider_usage.cost
        provider_cost = normalized_provider_cost(
            provider_cost_value,
            field_name="OpenRouter embedding usage.cost",
            strict=False,
        )
        response = EmbedTextResponse(
            provider=request.provider,
            model=request.model,
            input_type=request.input_type,
            data=[
                TextEmbeddingResult(index=index, embedding=list(vector))
                for index, vector in enumerate(provider_response.vectors)
            ],
            dimensions=len(provider_response.vectors[0]),
            usage=usage,
        )
        routing_evidence = {
            key: value
            for key, value in {
                "upstream_model": provider_response.model,
                "provider_request_id": provider_response.id,
            }.items()
            if value is not None
        }
        if provider_cost is not None:
            cost_usd = provider_cost
            evidence = {
                "settlement_source": "provider_returned",
                "pricing_origin": "openrouter_embedding_usage_cost",
                **routing_evidence,
            }
            if provider_usage is not None and provider_usage.cost_details is not None:
                evidence["provider_cost_details"] = provider_usage.cost_details
        elif input_tokens is not None:
            cost_usd = price_embedding(request.provider, request.model, input_tokens=input_tokens)
            pricing = MINER_TOOL_EMBEDDING_PRICING[request.provider][request.model]
            evidence = {
                "settlement_source": "static_pricing",
                "pricing_origin": "miner_tool_embedding_pricing",
                "provider_cost_status": (
                    "missing" if provider_cost_value is None else "malformed"
                ),
                "input_tokens": input_tokens,
                "input_per_million": pricing.input_per_million,
                **routing_evidence,
            }
        else:
            cost_usd = None
            evidence = {
                "settlement_source": "unavailable",
                "pricing_origin": "unavailable",
                "provider_cost_status": (
                    "missing" if provider_cost_value is None else "malformed"
                ),
                "usage_status": "missing" if provider_usage is None else "tokens_missing",
                **routing_evidence,
            }
        evidence.update({
            "provider": request.provider,
            "model": request.model,
            "input_type": request.input_type,
            "text_count": len(request.texts),
        })
        return EmbeddingProviderResult(
            response=response,
            actual_cost_usd=cost_usd,
            actual_cost_provider=request.provider,
            actual_cost_evidence=evidence,
        )

    async def aclose(self) -> None:
        errors: list[Exception] = []
        for key, client in self._clients.items():
            try:
                await client.aclose()
            except Exception as exc:
                exc.add_note(f"cached embedding provider close failed: {key}")
                errors.append(exc)
        if errors:
            raise ExceptionGroup("cached embedding provider cleanup failed", errors)

    def _client_for(self, *, model: str, dimensions: int | None) -> OpenRouterEmbeddingClient:
        key = (model, dimensions)
        client = self._clients.get(key)
        if client is None:
            client = OpenRouterEmbeddingClient(
                model=model,
                api_key=self._api_key,
                timeout_seconds=self._timeout_seconds,
                dimensions=dimensions,
            )
            self._clients[key] = client
        return client


def _format_embedding_texts(request: EmbedTextRequest) -> tuple[str, ...]:
    if request.input_type == "document":
        return request.texts
    instruction = request.instruction or QWEN3_DEFAULT_QUERY_INSTRUCTION
    return tuple(f"Instruct: {instruction}\nQuery:{text}" for text in request.texts)


def _parse_embedding_provider(raw: EmbeddingProviderName | str) -> EmbeddingProviderName:
    return parse_miner_selected_embedding_provider(raw)


__all__ = [
    "CachedEmbeddingProviderRegistry",
    "CachedWebSearchProviderRegistry",
    "ChutesEmbeddingProvider",
    "LazyLlmProvider",
    "LazySearchProvider",
    "OpenRouterEmbeddingProvider",
    "ToolInvocationClients",
    "build_miner_paid_embedding_provider",
    "build_miner_paid_web_search_provider",
    "build_optional_tool_embedding_provider",
    "build_optional_tool_llm_provider",
    "build_tool_invocation_clients",
    "build_tool_llm_provider",
    "build_web_search_provider",
    "build_web_search_provider_for_name",
    "validate_tool_invocation_provider_policy",
]
