from __future__ import annotations

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm import provider_factory


def test_llm_settings_default_provider_concurrency_targets_match_activation_slice() -> None:
    assert LlmSettings.model_fields["bedrock_max_concurrent"].default == 20
    assert LlmSettings.model_fields["chutes_max_concurrent"].default == 20
    assert LlmSettings.model_fields["vertex_max_concurrent"].default == 10


def test_build_cached_llm_provider_resolver_caches_by_provider_name(
    monkeypatch,
) -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    class _FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.append((self.__class__.__name__, kwargs))

    class _FakeChutesProvider(_FakeProvider):
        pass

    class _FakeBedrockProvider(_FakeProvider):
        pass

    class _FakeVertexProvider(_FakeProvider):
        pass

    class _FakeAdapter:
        def __init__(self, *, provider_name: str, delegate: object) -> None:
            self.provider_name = provider_name
            self.delegate = delegate

    monkeypatch.setattr(provider_factory, "BedrockLlmProvider", _FakeBedrockProvider)
    monkeypatch.setattr(provider_factory, "ChutesLlmProvider", _FakeChutesProvider)
    monkeypatch.setattr(provider_factory, "VertexLlmProvider", _FakeVertexProvider)
    monkeypatch.setattr(provider_factory, "LlmProviderAdapter", _FakeAdapter)

    resolver = provider_factory.build_cached_llm_provider_resolver(
        llm_settings=LlmSettings.model_construct(
            bedrock_max_concurrent=5,
            chutes_api_key=SecretStr("test-key"),
            chutes_max_concurrent=7,
            vertex_max_concurrent=11,
        ),
        bedrock_settings=BedrockSettings.model_construct(
            region="us-east-1",
            connect_timeout_seconds=5.0,
            read_timeout_seconds=60.0,
        ),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_maas_gcp_location="us-east5",
            vertex_timeout_seconds=45.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    first = resolver("chutes")
    second = resolver("chutes")
    third = resolver("bedrock")
    fourth = resolver("vertex")
    fifth = resolver("vertex-maas")

    assert first is second
    assert captured == [
        (
            "_FakeChutesProvider",
            {
                "base_url": provider_factory.CHUTES.base_url,
                "api_key": "test-key",
                "timeout": provider_factory.CHUTES.timeout_seconds,
                "max_concurrent": 7,
            },
        ),
        (
            "_FakeBedrockProvider",
            {
                "region": "us-east-1",
                "connect_timeout_seconds": 5.0,
                "read_timeout_seconds": 60.0,
                "max_concurrent": 5,
            },
        ),
        (
            "_FakeVertexProvider",
            {
                "project": "project",
                "location": "us-central1",
                "timeout": 45.0,
                "service_account_b64": "vertex-creds",
                "max_concurrent": 11,
            },
        ),
        (
            "_FakeVertexProvider",
            {
                "project": "project",
                "location": "us-east5",
                "timeout": 45.0,
                "service_account_b64": "vertex-creds",
                "max_concurrent": 11,
            },
        ),
    ]
    assert third.provider_name == "bedrock"
    assert fourth.provider_name == "vertex"
    assert fifth.provider_name == "vertex-maas"


async def test_build_cached_llm_provider_registry_closes_cached_providers_once(
    monkeypatch,
) -> None:
    closed: list[str] = []

    class _FakeProvider:
        def __init__(self, *, provider_name: str) -> None:
            self.provider_name = provider_name

        async def aclose(self) -> None:
            closed.append(self.provider_name)

    def fake_build_provider(*, provider_name, llm_settings, bedrock_settings, vertex_settings):
        _ = (llm_settings, bedrock_settings, vertex_settings)
        return _FakeProvider(provider_name=provider_name)

    monkeypatch.setattr(provider_factory, "_build_provider", fake_build_provider)

    registry = provider_factory.build_cached_llm_provider_registry(
        llm_settings=LlmSettings.model_construct(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_maas_gcp_location="us-east5",
            vertex_timeout_seconds=45.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    first = registry.resolve("chutes")
    second = registry.resolve("chutes")
    registry.resolve("bedrock")

    assert first is second

    await registry.aclose()

    assert closed == ["chutes", "bedrock"]


async def test_build_cached_llm_provider_registry_closes_later_providers_after_failure(
    monkeypatch,
) -> None:
    closed: list[str] = []

    class _FakeProvider:
        def __init__(self, *, provider_name: str) -> None:
            self.provider_name = provider_name

        async def aclose(self) -> None:
            closed.append(self.provider_name)
            if self.provider_name == "chutes":
                raise RuntimeError("boom")

    def fake_build_provider(*, provider_name, llm_settings, bedrock_settings, vertex_settings):
        _ = (llm_settings, bedrock_settings, vertex_settings)
        return _FakeProvider(provider_name=provider_name)

    monkeypatch.setattr(provider_factory, "_build_provider", fake_build_provider)

    registry = provider_factory.build_cached_llm_provider_registry(
        llm_settings=LlmSettings.model_construct(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_maas_gcp_location="us-east5",
            vertex_timeout_seconds=45.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    registry.resolve("chutes")
    registry.resolve("bedrock")

    with pytest.raises(ExceptionGroup) as exc_info:
        await registry.aclose()

    assert closed == ["chutes", "bedrock"]
    assert len(exc_info.value.exceptions) == 1
    assert exc_info.value.exceptions[0].__notes__ == ["cached llm provider close failed: chutes"]
