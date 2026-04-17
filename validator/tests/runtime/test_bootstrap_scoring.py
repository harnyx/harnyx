from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

import harnyx_validator.infrastructure.scoring.vertex_embedding as vertex_embedding
from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.observability import ObservabilitySettings
from harnyx_commons.config.platform_api import PlatformApiSettings
from harnyx_commons.config.sandbox import SandboxSettings
from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.errors import ConcurrencyLimitError
from harnyx_commons.llm.routing import ResolvedLlmRoute
from harnyx_validator.infrastructure.scoring.vertex_embedding import LazyVertexTextEmbeddingClient
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.bootstrap import (
    _build_llm_clients,
    _build_local_eval_tooling_clients,
    _create_scoring_service,
    _create_search_client,
    close_runtime_resources,
)
from harnyx_validator.runtime.settings import Settings


def test_create_search_client_requires_search_provider() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider=None,
        )
    )

    with pytest.raises(RuntimeError, match="SEARCH_PROVIDER must be configured"):
        _create_search_client(settings)


def test_create_search_client_uses_configured_parallel_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeParallelClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(bootstrap, "ParallelClient", _FakeParallelClient)
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
        )
    )

    _create_search_client(settings)

    assert captured["base_url"] == "https://proxy.parallel.test"
    assert captured["api_key"] == "parallel-key"
    assert captured["max_concurrent"] == 7


def test_build_llm_clients_uses_shared_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_maas_gcp_location="us-east5",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )
    calls: list[str] = []

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            calls.append(name)
            return f"provider:{name}"

    def fake_build_cached_llm_provider_registry(*, llm_settings, bedrock_settings, vertex_settings):
        assert llm_settings is settings.llm
        assert bedrock_settings is settings.bedrock
        assert vertex_settings is settings.vertex
        return _FakeRegistry()

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", fake_build_cached_llm_provider_registry)

    _, tool_provider, scoring_provider, scoring_route = _build_llm_clients(settings)

    assert tool_provider == "provider:chutes"
    assert scoring_provider == "provider:vertex"
    assert scoring_route == ResolvedLlmRoute(
        surface="scoring",
        provider="vertex",
        model=bootstrap._SCORING_LLM_MODEL,
    )
    assert calls == ["chutes", "vertex"]


def test_build_local_eval_tooling_clients_allows_missing_search_provider() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider=None,
            tool_llm_provider="chutes",
            scoring_llm_provider="chutes",
            chutes_api_key=SecretStr("test-key"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_maas_gcp_location="us-east5",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    search_client, tool_provider, scoring_provider, scoring_route = _build_local_eval_tooling_clients(settings)

    assert search_client is None
    assert tool_provider is not None
    assert scoring_provider is not None
    assert type(tool_provider).__name__ == "_LazyLlmProvider"
    assert scoring_route == ResolvedLlmRoute(
        surface="scoring",
        provider="chutes",
        model=bootstrap._SCORING_LLM_MODEL,
    )


def test_build_llm_clients_uses_scoring_model_override_for_route_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
            scoring_llm_model_override="custom/internal-model",
            llm_model_provider_overrides_json=json.dumps({"scoring": {"custom/internal-model": "bedrock"}}),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_maas_gcp_location="us-east5",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    _, _, scoring_provider, scoring_route = _build_llm_clients(settings)

    assert scoring_provider == "provider:bedrock"
    assert scoring_route == ResolvedLlmRoute(
        surface="scoring",
        provider="bedrock",
        model="custom/internal-model",
    )


def test_build_state_activates_two_parallel_tool_calls_per_token() -> None:
    state = bootstrap._build_state()

    state.token_semaphore.acquire("token")
    state.token_semaphore.acquire("token")
    with pytest.raises(ConcurrencyLimitError):
        state.token_semaphore.acquire("token")

    state.token_semaphore.release("token")
    state.token_semaphore.release("token")


def test_create_scoring_service_does_not_require_vertex_config_at_bootstrap() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr("test-key"),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id=None,
            gcp_location=None,
            vertex_maas_gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr(""),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="chutes",
            model=bootstrap._SCORING_LLM_MODEL,
        ),
    )

    assert service is not None
    assert service._config.provider == "chutes"
    assert service._config.model == bootstrap._SCORING_LLM_MODEL
    assert service._config.reasoning_effort == bootstrap._SCORING_LLM_REASONING_EFFORT


def test_create_scoring_service_uses_effective_route_model_and_provider() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="vertex",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr("test-key"),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_maas_gcp_location="us-east5",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="bedrock",
            model="custom/internal-model",
        ),
    )

    assert service._config.provider == "bedrock"
    assert service._config.model == "custom/internal-model"
    assert isinstance(service._embeddings, LazyVertexTextEmbeddingClient)


def test_create_scoring_service_uses_chutes_embeddings_for_chutes_provider() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr("test-key"),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id=None,
            gcp_location=None,
            vertex_maas_gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr(""),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="chutes",
            model=bootstrap._SCORING_LLM_MODEL,
        ),
    )

    assert service._embeddings.__class__.__name__ == "ChutesTextEmbeddingClient"
    assert service._embeddings.model == "Qwen/Qwen3-Embedding-0.6B"
    assert service._embeddings.base_url == "https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai"


def test_create_scoring_service_fails_when_chutes_embedding_model_is_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr("test-key"),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id=None,
            gcp_location=None,
            vertex_maas_gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr(""),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )
    monkeypatch.setattr(bootstrap, "_SCORING_CHUTES_EMBEDDING_MODEL", "Unknown/Embedding-Model")

    with pytest.raises(RuntimeError, match="no chutes embedding base_url configured"):
        _create_scoring_service(
            settings,
            provider=SimpleNamespace(),
            scoring_route=ResolvedLlmRoute(
                surface="scoring",
                provider="chutes",
                model=bootstrap._SCORING_LLM_MODEL,
            ),
        )


def test_create_scoring_service_requires_chutes_api_key_for_chutes_embeddings() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr(""),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id=None,
            gcp_location=None,
            vertex_maas_gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr(""),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    with pytest.raises(RuntimeError, match="CHUTES_API_KEY must be configured"):
        _create_scoring_service(
            settings,
            provider=SimpleNamespace(),
            scoring_route=ResolvedLlmRoute(
                surface="scoring",
                provider="chutes",
                model=bootstrap._SCORING_LLM_MODEL,
            ),
        )


def test_create_scoring_service_uses_vertex_maas_region_for_embeddings() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="vertex-maas",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr(""),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="test-project",
            gcp_location=None,
            vertex_maas_gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr(""),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="vertex-maas",
            model=bootstrap._SCORING_LLM_MODEL,
        ),
    )

    assert isinstance(service._embeddings, LazyVertexTextEmbeddingClient)
    assert service._embeddings.location == "us-central1"


def test_build_llm_clients_allows_bedrock_scoring_route_while_embeddings_stay_on_default_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
            llm_model_provider_overrides_json=json.dumps(
                {"scoring": {bootstrap._SCORING_LLM_MODEL: "bedrock"}}
            ),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_maas_gcp_location="us-east5",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    _, _, scoring_provider, scoring_route = _build_llm_clients(settings)
    embedding_client = bootstrap._create_scoring_embedding_client(settings)

    assert scoring_provider == "provider:bedrock"
    assert scoring_route == ResolvedLlmRoute(
        surface="scoring",
        provider="bedrock",
        model=bootstrap._SCORING_LLM_MODEL,
    )
    assert isinstance(embedding_client, LazyVertexTextEmbeddingClient)


@pytest.mark.anyio
async def test_lazy_vertex_text_embedding_client_uses_async_sdk_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, object]] = []
    captured: dict[str, object] = {}

    class _SyncModels:
        def embed_content(self, *args, **kwargs):
            raise AssertionError("sync embedding path should not be used")

    class _AsyncModels:
        async def embed_content(self, *, model: str, contents: str, config: object) -> object:
            calls.append((model, contents, config))
            return SimpleNamespace(embeddings=[SimpleNamespace(values=(1.0, 2.0))])

    class _AsyncClient:
        def __init__(self) -> None:
            self.models = _AsyncModels()
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.models = _SyncModels()
            self.aio = _AsyncClient()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(vertex_embedding, "prepare_credentials", lambda _path, _b64: (None, None))
    monkeypatch.setattr(vertex_embedding.genai, "Client", _FakeClient)

    client = LazyVertexTextEmbeddingClient(
        project="test-project",
        location="us-central1",
        service_account_b64=None,
        model="gemini-embedding-001",
        timeout_seconds=15.0,
        dimensions=2,
    )

    vector = await client.embed("hello world")

    assert vector == (1.0, 2.0)
    assert captured["vertexai"] is True
    assert captured["project"] == "test-project"
    assert captured["location"] == "us-central1"
    assert len(calls) == 1
    assert calls[0][0] == "gemini-embedding-001"
    assert calls[0][1] == "hello world"

    underlying = client._client
    assert underlying is not None

    await client.aclose()

    assert underlying.client.aio.closed is True
    assert underlying.client.closed is True


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _CountingClosable:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _ShutdownSpyExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.calls.append((wait, cancel_futures))


@pytest.mark.anyio
async def test_close_runtime_resources_closes_scoring_embedding_client() -> None:
    scoring_embedding_client = _Closable()
    blocking_executor = _ShutdownSpyExecutor()
    runtime = SimpleNamespace(
        batch_blocking_executor=blocking_executor,
        search_client=None,
        tool_llm_provider=None,
        scoring_llm_provider=None,
        scoring_embedding_client=scoring_embedding_client,
    )

    await close_runtime_resources(runtime)

    assert blocking_executor.calls == [(False, True)]
    assert scoring_embedding_client.closed is True


@pytest.mark.anyio
async def test_close_runtime_resources_dedupes_shared_llm_provider() -> None:
    shared_provider = _CountingClosable()
    blocking_executor = _ShutdownSpyExecutor()
    runtime = SimpleNamespace(
        batch_blocking_executor=blocking_executor,
        search_client=None,
        tool_llm_provider=shared_provider,
        scoring_llm_provider=shared_provider,
        scoring_embedding_client=None,
    )

    await close_runtime_resources(runtime)

    assert blocking_executor.calls == [(False, True)]
    assert shared_provider.close_calls == 1
