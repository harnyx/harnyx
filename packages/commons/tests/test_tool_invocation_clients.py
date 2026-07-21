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
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_provider_timeout_seconds
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
CHUTES_SELECTED_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", QWEN36_MODEL)


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


def test_tool_invocation_clients_do_not_resolve_tool_provider_until_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            calls.append(name)
            return f"provider:{name}"

    monkeypatch.setattr(
        invocation_clients,
        "build_cached_llm_provider_registry",
        lambda **_: _FakeRegistry(),
    )

    clients = build_tool_invocation_clients(
        llm_settings=_llm_settings(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
    )

    assert clients.search_client is None
    assert clients.tool_llm_provider is not None
    assert calls == []


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


def test_internal_search_provider_keeps_configured_desearch_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class _FakeDeSearchClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(invocation_clients, "DeSearchClient", _FakeDeSearchClient)

    provider = invocation_clients.build_web_search_provider(
        LlmSettings.model_construct(
            search_provider="desearch",
            desearch_api_key=SecretStr("operator-desearch-key"),
            desearch_max_concurrent=7,
        )
    )

    assert provider is not None
    assert captured == [
        {
            "base_url": invocation_clients.DESEARCH.base_url,
            "api_key": "operator-desearch-key",
            "timeout": invocation_clients.DESEARCH.timeout_seconds,
            "max_concurrent": 7,
            "include_payloads_in_logs": True,
        }
    ]


def test_internal_search_provider_keeps_configured_parallel_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class _FakeParallelClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(invocation_clients, "ParallelClient", _FakeParallelClient)

    provider = invocation_clients.build_web_search_provider(
        LlmSettings.model_construct(
            search_provider="parallel",
            parallel_api_key=SecretStr("operator-parallel-key"),
            parallel_base_url="https://parallel.example",
            parallel_max_concurrent=11,
        )
    )

    assert provider is not None
    assert captured == [
        {
            "base_url": "https://parallel.example",
            "api_key": "operator-parallel-key",
            "timeout": invocation_clients.PARALLEL.timeout_seconds,
            "max_concurrent": 11,
            "include_payloads_in_logs": True,
        }
    ]


def test_cached_web_search_provider_registry_resolves_requested_provider_without_payload_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    class _FakeDeSearchClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(("desearch", kwargs))

    class _FakeParallelClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(("parallel", kwargs))

    monkeypatch.setattr(invocation_clients, "DeSearchClient", _FakeDeSearchClient)
    monkeypatch.setattr(invocation_clients, "ParallelClient", _FakeParallelClient)
    registry = invocation_clients.CachedWebSearchProviderRegistry(
        llm_settings=LlmSettings.model_construct(
            desearch_api_key=SecretStr("operator-desearch-key"),
            desearch_max_concurrent=7,
            parallel_api_key=SecretStr("operator-parallel-key"),
            parallel_base_url="https://parallel.example",
            parallel_max_concurrent=11,
        ),
        include_payloads_in_logs=False,
    )

    parallel = registry.resolve("parallel")
    same_parallel = registry.resolve("parallel")
    desearch = registry.resolve("desearch")

    assert parallel is same_parallel
    assert parallel is not desearch
    assert captured == [
        (
            "parallel",
            {
                "base_url": "https://parallel.example",
                "api_key": "operator-parallel-key",
                "timeout": invocation_clients.PARALLEL.timeout_seconds,
                "max_concurrent": 11,
                "include_payloads_in_logs": False,
            },
        ),
        (
            "desearch",
            {
                "base_url": invocation_clients.DESEARCH.base_url,
                "api_key": "operator-desearch-key",
                "timeout": invocation_clients.DESEARCH.timeout_seconds,
                "max_concurrent": 7,
                "include_payloads_in_logs": False,
            },
        ),
    ]


@pytest.mark.parametrize("provider", ["desearch", "parallel"])
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


def test_miner_paid_desearch_provider_uses_explicit_key_without_shared_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class _FakeDeSearchClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(invocation_clients, "DeSearchClient", _FakeDeSearchClient)

    provider = invocation_clients.build_miner_paid_web_search_provider(
        provider="desearch",
        api_key=SecretStr("miner-desearch-key"),
        llm_settings=LlmSettings.model_construct(
            desearch_api_key=SecretStr("operator-desearch-key"),
            desearch_max_concurrent=7,
        ),
    )

    assert provider is not None
    assert captured == [
        {
            "base_url": invocation_clients.DESEARCH.base_url,
            "api_key": "miner-desearch-key",
            "timeout": invocation_clients.DESEARCH.timeout_seconds,
            "max_concurrent": None,
        }
    ]


def test_miner_paid_parallel_provider_uses_explicit_key_without_shared_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class _FakeParallelClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(invocation_clients, "ParallelClient", _FakeParallelClient)

    provider = invocation_clients.build_miner_paid_web_search_provider(
        provider="parallel",
        api_key=SecretStr("miner-parallel-key"),
        llm_settings=LlmSettings.model_construct(
            parallel_api_key=SecretStr("operator-parallel-key"),
            parallel_base_url="https://parallel.example",
            parallel_max_concurrent=11,
        ),
    )

    assert provider is not None
    assert captured == [
        {
            "base_url": "https://parallel.example",
            "api_key": "miner-parallel-key",
            "timeout": invocation_clients.PARALLEL.timeout_seconds,
            "max_concurrent": None,
        }
    ]


def test_miner_paid_search_provider_uses_effective_timeout_when_above_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class _FakeParallelClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(invocation_clients, "ParallelClient", _FakeParallelClient)

    provider = invocation_clients.build_miner_paid_web_search_provider(
        provider="parallel",
        api_key=SecretStr("miner-parallel-key"),
        llm_settings=LlmSettings.model_construct(
            parallel_base_url="https://parallel.example",
        ),
        timeout=180.0,
    )

    assert provider is not None
    assert captured == [
        {
            "base_url": "https://parallel.example",
            "api_key": "miner-parallel-key",
            "timeout": platform_tool_proxy_provider_timeout_seconds(180.0),
            "max_concurrent": None,
        }
    ]


def test_miner_paid_search_provider_keeps_default_when_effective_timeout_is_shorter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class _FakeDeSearchClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(invocation_clients, "DeSearchClient", _FakeDeSearchClient)

    provider = invocation_clients.build_miner_paid_web_search_provider(
        provider="desearch",
        api_key=SecretStr("miner-desearch-key"),
        llm_settings=LlmSettings.model_construct(),
        timeout=5.0,
    )

    assert provider is not None
    assert captured == [
        {
            "base_url": invocation_clients.DESEARCH.base_url,
            "api_key": "miner-desearch-key",
            "timeout": invocation_clients.DESEARCH.timeout_seconds,
            "max_concurrent": None,
        }
    ]


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
async def test_tool_invocation_clients_route_tool_model_to_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _FakeLlmRegistry()
    monkeypatch.setattr(invocation_clients, "build_cached_llm_provider_registry", lambda **_: registry)

    clients = build_tool_invocation_clients(
        llm_settings=_llm_settings(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
    )

    assert clients.tool_llm_provider is not None
    await clients.tool_llm_provider.invoke(_gemma_tool_request())

    assert registry.requests_by_provider["custom-openai-compatible:gemma4-cloud-run-turbo"][0].provider == (
        "custom-openai-compatible:gemma4-cloud-run-turbo"
    )


@pytest.mark.anyio("asyncio")
async def test_tool_invocation_clients_route_qwen36_tool_model_to_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _FakeLlmRegistry()
    monkeypatch.setattr(invocation_clients, "build_cached_llm_provider_registry", lambda **_: registry)

    clients = build_tool_invocation_clients(
        llm_settings=_llm_settings(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
    )

    assert clients.tool_llm_provider is not None
    await clients.tool_llm_provider.invoke(_qwen36_tool_request())

    assert registry.requests_by_provider[QWEN36_ROUTE_TARGET][0].provider == QWEN36_ROUTE_TARGET


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("model", CHUTES_SELECTED_MODELS)
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
