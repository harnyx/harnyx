from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.errors import ProviderCredentialUnavailableError
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.tools import invocation_clients
from harnyx_commons.tools.invocation_clients import (
    ChutesEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    build_optional_tool_embedding_provider,
    build_tool_invocation_clients,
)

GEMMA_MODEL = "google/gemma-4-31B-turbo-TEE"
GEMMA_ROUTE_TARGET = "custom-openai-compatible:gemma4-cloud-run-turbo"
QWEN36_MODEL = "Qwen/Qwen3.6-27B-TEE"
QWEN36_ROUTE_TARGET = "custom-openai-compatible:qwen36-cloud-run"


class _FakeLlmProvider:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            id="resp-1",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(),
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        return None


class _FakeLlmRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, _FakeLlmProvider] = {}

    @property
    def requests_by_provider(self) -> dict[str, list[LlmRequest]]:
        return {provider_name: provider.requests for provider_name, provider in self._providers.items()}

    def resolve(self, name: str) -> _FakeLlmProvider:
        provider = self._providers.get(name)
        if provider is None:
            provider = _FakeLlmProvider()
            self._providers[name] = provider
        return provider


def _llm_settings() -> LlmSettings:
    return _llm_settings_with_tool_overrides(
        {
            GEMMA_MODEL: GEMMA_ROUTE_TARGET,
            QWEN36_MODEL: QWEN36_ROUTE_TARGET,
        }
    )


def _llm_settings_without_qwen36_override() -> LlmSettings:
    return _llm_settings_with_tool_overrides({GEMMA_MODEL: GEMMA_ROUTE_TARGET})


def _llm_settings_with_tool_overrides(tool_overrides: dict[str, str]) -> LlmSettings:
    return LlmSettings.model_construct(
        search_provider=None,
        tool_llm_provider="chutes",
        tool_embedding_provider="chutes",
        chutes_api_key=SecretStr("test-key"),
        openrouter_api_key=SecretStr(""),
        llm_timeout_seconds=300.0,
        llm_model_provider_overrides_json=json.dumps({"tool": tool_overrides}),
        openai_compatible_endpoints_json=json.dumps(
            [
                {
                    "id": "gemma4-cloud-run-turbo",
                    "base_url": "https://gemma.example.run.app/v1",
                    "auth": {"type": "none"},
                },
                {
                    "id": "qwen36-cloud-run",
                    "base_url": "https://qwen.example.run.app/v1",
                    "auth": {"type": "none"},
                },
            ]
        ),
    )


def test_optional_tool_embedding_provider_builds_chutes_from_tool_embedding_provider_setting() -> None:
    settings = LlmSettings.model_construct(
        tool_embedding_provider="chutes",
        chutes_api_key=SecretStr("chutes-key"),
        openrouter_api_key=SecretStr(""),
        llm_timeout_seconds=300.0,
    )

    provider = build_optional_tool_embedding_provider(settings)

    assert isinstance(provider, ChutesEmbeddingProvider)


def test_optional_tool_embedding_provider_builds_openrouter_from_tool_embedding_provider_setting() -> None:
    settings = LlmSettings.model_construct(
        tool_embedding_provider="openrouter",
        chutes_api_key=SecretStr(""),
        openrouter_api_key=SecretStr("openrouter-key"),
        llm_timeout_seconds=300.0,
    )

    provider = build_optional_tool_embedding_provider(settings)

    assert isinstance(provider, OpenRouterEmbeddingProvider)


def test_cached_embedding_provider_registry_resolves_provider_specific_clients() -> None:
    settings = LlmSettings.model_construct(
        chutes_api_key=SecretStr("chutes-key"),
        openrouter_api_key=SecretStr("openrouter-key"),
        llm_timeout_seconds=300.0,
    )
    registry = invocation_clients.CachedEmbeddingProviderRegistry(llm_settings=settings)

    chutes = registry.resolve("chutes")
    same_chutes = registry.resolve("chutes")
    openrouter = registry.resolve("openrouter")

    assert chutes is same_chutes
    assert isinstance(chutes, ChutesEmbeddingProvider)
    assert isinstance(openrouter, OpenRouterEmbeddingProvider)


def _gemma_tool_request() -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model=GEMMA_MODEL,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
    )


def _qwen36_tool_request() -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model=QWEN36_MODEL,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
    )


def _openrouter_tool_request(*, model: str) -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
    )


def test_tool_invocation_clients_can_require_search_provider() -> None:
    with pytest.raises(RuntimeError, match="SEARCH_PROVIDER must be configured"):
        build_tool_invocation_clients(
            llm_settings=_llm_settings(),
            bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
            vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
            require_search=True,
        )


def test_tool_invocation_clients_can_skip_routed_tool_provider_policy() -> None:
    clients = build_tool_invocation_clients(
        llm_settings=LlmSettings.model_construct(tool_llm_provider="bedrock"),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
        build_routed_tool_llm_provider=False,
    )

    assert clients.tool_llm_provider is None


@pytest.mark.parametrize(
    ("provider", "client_name", "key_field", "concurrency_field", "base_url"),
    [
        ("exa", "ExaClient", "exa_api_key", "exa_max_concurrent", "https://api.exa.ai"),
        ("tavily", "TavilyClient", "tavily_api_key", "tavily_max_concurrent", "https://api.tavily.com"),
    ],
)
def test_internal_new_search_providers_use_configured_credentials_and_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    client_name: str,
    key_field: str,
    concurrency_field: str,
    base_url: str,
) -> None:
    captured: list[dict[str, object]] = []
    sentinel = object()
    monkeypatch.setattr(
        invocation_clients,
        client_name,
        lambda **kwargs: captured.append(kwargs) or sentinel,
    )
    settings = LlmSettings.model_construct(
        search_provider=provider,
        **{key_field: SecretStr(f"operator-{provider}-key"), concurrency_field: 17},
    )

    resolved = invocation_clients.build_web_search_provider_for_name(settings, provider)

    assert resolved is sentinel
    assert captured == [
        {
            "base_url": base_url,
            "api_key": f"operator-{provider}-key",
            "timeout": 60.0,
            "max_concurrent": 17,
            "include_payloads_in_logs": True,
        }
    ]


def test_cached_search_registry_shares_ai_and_web_clients_but_rejects_firecrawl_for_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parallel = object()
    firecrawl = object()
    monkeypatch.setattr(invocation_clients, "ParallelClient", lambda **_kwargs: parallel)
    monkeypatch.setattr(invocation_clients, "FirecrawlClient", lambda **_kwargs: firecrawl)
    registry = invocation_clients.CachedWebSearchProviderRegistry(
        llm_settings=LlmSettings.model_construct(
            parallel_api_key=SecretStr("parallel-key"),
            parallel_base_url="https://parallel.example",
            parallel_max_concurrent=2,
            firecrawl_api_key=SecretStr("firecrawl-key"),
            firecrawl_max_concurrent=3,
        )
    )

    assert registry.resolve_web("parallel") is parallel
    assert registry.resolve_ai("parallel") is parallel
    assert registry.resolve_web("firecrawl") is firecrawl
    with pytest.raises(ValueError, match="AI search provider 'firecrawl' is not supported"):
        registry.resolve_ai("firecrawl")  # type: ignore[arg-type]


def test_fixed_parallel_client_is_shared_between_web_and_ai_roles() -> None:
    clients = build_tool_invocation_clients(
        llm_settings=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_base_url="https://parallel.example",
            parallel_max_concurrent=2,
            tool_llm_provider=None,
            tool_embedding_provider="chutes",
            chutes_api_key=SecretStr(""),
        ),
        bedrock_settings=BedrockSettings.model_construct(),
        vertex_settings=VertexSettings.model_construct(),
        build_routed_tool_llm_provider=False,
    )

    assert clients.search_client is clients.ai_search_client


@pytest.mark.parametrize(
    "provider",
    [
        "desearch",
    ],
)
def test_cached_web_search_provider_registry_reports_missing_platform_credential(provider: str) -> None:
    registry = invocation_clients.CachedWebSearchProviderRegistry(llm_settings=LlmSettings.model_construct())

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        registry.resolve(provider)

    assert exc_info.value.provider == provider


@pytest.mark.parametrize("provider", ["chutes", "openrouter"])
def test_cached_embedding_provider_registry_reports_missing_platform_credential(provider: str) -> None:
    registry = invocation_clients.CachedEmbeddingProviderRegistry(llm_settings=LlmSettings.model_construct())

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        registry.resolve(provider)

    assert exc_info.value.provider == provider


@pytest.mark.parametrize(
    ("provider", "client_name", "operator_settings"),
    [
        (
            "desearch",
            "DeSearchClient",
            {
                "desearch_api_key": SecretStr("operator-desearch-key"),
                "desearch_max_concurrent": 7,
            },
        ),
        (
            "parallel",
            "ParallelClient",
            {
                "parallel_api_key": SecretStr("operator-parallel-key"),
                "parallel_base_url": "https://parallel.example",
                "parallel_max_concurrent": 11,
            },
        ),
    ],
)
def test_miner_paid_search_provider_isolates_credential_and_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    client_name: str,
    operator_settings: dict[str, object],
) -> None:
    captured: list[dict[str, object]] = []
    sentinel = object()
    monkeypatch.setattr(
        invocation_clients,
        client_name,
        lambda **kwargs: captured.append(kwargs) or sentinel,
    )

    resolved = invocation_clients.build_miner_paid_web_search_provider(
        provider=provider,
        api_key=SecretStr(f"miner-{provider}-key"),
        llm_settings=LlmSettings.model_construct(**operator_settings),
    )

    assert resolved is sentinel
    assert captured[0]["api_key"] == f"miner-{provider}-key"
    assert captured[0]["max_concurrent"] is None


def test_miner_paid_web_search_provider_rejects_blank_key() -> None:
    with pytest.raises(ValueError, match="miner-paid API key must be provided"):
        invocation_clients.build_miner_paid_web_search_provider(
            provider="desearch",
            api_key="   ",
            llm_settings=LlmSettings.model_construct(),
        )


def test_miner_paid_web_search_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="search provider"):
        invocation_clients.build_miner_paid_web_search_provider(
            provider="unknown",
            api_key="miner-key",
            llm_settings=LlmSettings.model_construct(),
        )


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("model", ("openai/gpt-oss-20b",))
async def test_tool_invocation_clients_keep_chutes_selected_model_on_chutes(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    registry = _FakeLlmRegistry()
    monkeypatch.setattr(invocation_clients, "build_cached_llm_provider_registry", lambda **_: registry)
    llm_settings = _llm_settings_without_qwen36_override() if model == QWEN36_MODEL else _llm_settings()

    clients = build_tool_invocation_clients(
        llm_settings=llm_settings,
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
    )

    assert clients.tool_llm_provider is not None
    await clients.tool_llm_provider.invoke(_openrouter_tool_request(model=model))

    assert registry.requests_by_provider["chutes"][0].provider == "chutes"
    assert registry.requests_by_provider["chutes"][0].model == model


@pytest.mark.parametrize(
    "llm_settings",
    [
        LlmSettings.model_construct(tool_llm_provider="bedrock"),
        LlmSettings.model_construct(
            tool_llm_provider="chutes",
            llm_model_provider_overrides_json=json.dumps({"tool": {"sample-tool-model": "bedrock"}}),
        ),
    ],
)
def test_tool_invocation_clients_reject_bedrock_tool_routes(llm_settings: LlmSettings) -> None:
    with pytest.raises(ValueError, match="TOOL_LLM_PROVIDER='bedrock' is not supported"):
        build_tool_invocation_clients(
            llm_settings=llm_settings,
            bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
            vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
        )
