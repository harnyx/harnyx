from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.observability import ObservabilitySettings
from harnyx_commons.config.platform_api import PlatformApiSettings
from harnyx_commons.config.sandbox import SandboxSettings
from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_validator.infrastructure.scoring.vertex_embedding import LazyVertexTextEmbeddingClient
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.bootstrap import (
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


def test_create_scoring_service_does_not_require_vertex_config_at_bootstrap() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_model="openai/gpt-oss-20b",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_reasoning_effort=None,
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

    service = _create_scoring_service(settings, provider=SimpleNamespace())

    assert service is not None


def test_create_scoring_service_uses_chutes_embeddings_for_chutes_provider() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_model="openai/gpt-oss-20b",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_reasoning_effort=None,
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

    service = _create_scoring_service(settings, provider=SimpleNamespace())

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
            scoring_llm_model="openai/gpt-oss-20b",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_reasoning_effort=None,
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
        _create_scoring_service(settings, provider=SimpleNamespace())


def test_create_scoring_service_requires_chutes_api_key_for_chutes_embeddings() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_model="openai/gpt-oss-20b",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_reasoning_effort=None,
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
        _create_scoring_service(settings, provider=SimpleNamespace())


def test_create_scoring_service_uses_vertex_maas_region_for_embeddings() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="vertex-maas",
            scoring_llm_model="gemini-2.5-flash",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_reasoning_effort=None,
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

    service = _create_scoring_service(settings, provider=SimpleNamespace())

    assert isinstance(service._embeddings, LazyVertexTextEmbeddingClient)
    assert service._embeddings.location == "us-central1"


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_close_runtime_resources_closes_scoring_embedding_client() -> None:
    scoring_embedding_client = _Closable()
    runtime = SimpleNamespace(
        search_client=None,
        tool_llm_provider=None,
        scoring_llm_provider=None,
        scoring_embedding_client=scoring_embedding_client,
    )

    await close_runtime_resources(runtime)

    assert scoring_embedding_client.closed is True
