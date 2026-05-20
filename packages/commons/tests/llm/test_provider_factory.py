from __future__ import annotations

from typing import cast

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings, OpenAiCompatibleEndpointConfig
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm import provider_factory


def test_llm_settings_default_provider_concurrency_targets_match_activation_slice() -> None:
    assert LlmSettings.model_fields["bedrock_max_concurrent"].default == 20
    assert LlmSettings.model_fields["chutes_max_concurrent"].default == 20
    assert LlmSettings.model_fields["vertex_max_concurrent"].default == 10


def test_default_llm_timeout_surfaces_are_300_seconds() -> None:
    settings = LlmSettings(_env_file=None)

    assert settings.llm_timeout_seconds == pytest.approx(300.0)
    assert settings.scoring_llm_timeout_seconds == pytest.approx(300.0)
    assert VertexSettings(_env_file=None).vertex_timeout_seconds == pytest.approx(300.0)
    assert BedrockSettings(_env_file=None).read_timeout_seconds == pytest.approx(300.0)
    assert provider_factory.CHUTES.timeout_seconds == pytest.approx(300.0)


def test_default_scoring_llm_max_output_tokens_is_20480() -> None:
    assert LlmSettings(_env_file=None).scoring_llm_max_output_tokens == 20480


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
            vertex_timeout_seconds=45.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    first = resolver("chutes")
    second = resolver("chutes")
    third = resolver("bedrock")
    fourth = resolver("vertex")

    assert first is second
    with pytest.raises(ValueError, match="vertex-maas"):
        resolver("vertex-maas")
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
    ]
    assert third.provider_name == "bedrock"
    assert fourth.provider_name == "vertex"


async def test_build_cached_llm_provider_registry_closes_cached_providers_once(
    monkeypatch,
) -> None:
    closed: list[str] = []

    class _FakeProvider:
        def __init__(self, *, provider_name: str) -> None:
            self.provider_name = provider_name

        async def aclose(self) -> None:
            closed.append(self.provider_name)

    def fake_build_provider(*, route_target, llm_settings, bedrock_settings, vertex_settings):
        _ = (llm_settings, bedrock_settings, vertex_settings)
        return _FakeProvider(provider_name=route_target)

    monkeypatch.setattr(provider_factory, "_build_provider", fake_build_provider)

    registry = provider_factory.build_cached_llm_provider_registry(
        llm_settings=LlmSettings.model_construct(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
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

    def fake_build_provider(*, route_target, llm_settings, bedrock_settings, vertex_settings):
        _ = (llm_settings, bedrock_settings, vertex_settings)
        return _FakeProvider(provider_name=route_target)

    monkeypatch.setattr(provider_factory, "_build_provider", fake_build_provider)

    registry = provider_factory.build_cached_llm_provider_registry(
        llm_settings=LlmSettings.model_construct(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
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


def test_build_cached_llm_provider_registry_caches_custom_openai_compatible_endpoint(
    monkeypatch,
) -> None:
    captured_endpoints: list[OpenAiCompatibleEndpointConfig] = []
    captured_adapters: list[tuple[str, object]] = []

    class _FakeOpenAiCompatibleProvider:
        def __init__(self, *, endpoint: OpenAiCompatibleEndpointConfig) -> None:
            self.endpoint = endpoint
            captured_endpoints.append(endpoint)

    class _FakeAdapter:
        def __init__(self, *, provider_name: str, delegate: object) -> None:
            self.provider_name = provider_name
            self.delegate = delegate
            captured_adapters.append((provider_name, delegate))

    monkeypatch.setattr(provider_factory, "OpenAiCompatibleLlmProvider", _FakeOpenAiCompatibleProvider)
    monkeypatch.setattr(provider_factory, "LlmProviderAdapter", _FakeAdapter)

    registry = provider_factory.build_cached_llm_provider_registry(
        llm_settings=LlmSettings(
            LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON=(
                '[{"id":"gemma4-cloud-run-turbo","base_url":"https://example.com/v1","auth":{"type":"none"}}]'
            )
        ),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=45.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    first = registry.resolve("custom-openai-compatible:gemma4-cloud-run-turbo")
    second = registry.resolve("custom-openai-compatible:gemma4-cloud-run-turbo")

    assert first is second
    assert len(captured_endpoints) == 1
    assert captured_endpoints[0].id == "gemma4-cloud-run-turbo"
    first_adapter = cast(_FakeAdapter, first)
    first_delegate = cast(_FakeOpenAiCompatibleProvider, first_adapter.delegate)
    assert captured_adapters == [("custom-openai-compatible:gemma4-cloud-run-turbo", first_delegate)]
    assert first_delegate.endpoint.id == "gemma4-cloud-run-turbo"


def test_build_cached_llm_provider_registry_builds_hardcoded_openrouter_target(
    monkeypatch,
) -> None:
    captured_providers: list[dict[str, object]] = []
    captured_adapters: list[tuple[str, object]] = []

    class _FakeOpenRouterProvider:
        def __init__(self, **kwargs: object) -> None:
            captured_providers.append(kwargs)

    class _FakeAdapter:
        def __init__(self, *, provider_name: str, delegate: object) -> None:
            self.provider_name = provider_name
            self.delegate = delegate
            captured_adapters.append((provider_name, delegate))

    monkeypatch.setattr(provider_factory, "OpenRouterLlmProvider", _FakeOpenRouterProvider)
    monkeypatch.setattr(provider_factory, "LlmProviderAdapter", _FakeAdapter)

    settings = LlmSettings(
        OPENROUTER_API_KEY="test-openrouter-key",
        OPENROUTER_MODEL_PROVIDER_OPTIONS_JSON=(
            '{"openai/gpt-oss-20b":{"require_parameters":true},'
            '"openai/gpt-oss-120b":{"require_parameters":true}}'
        ),
    )
    registry = provider_factory.build_cached_llm_provider_registry(
        llm_settings=settings,
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=45.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    first = registry.resolve("openrouter")
    second = registry.resolve("openrouter")

    assert first is second
    assert len(captured_providers) == 1
    assert captured_providers[0]["openrouter_api_key"] == settings.openrouter_api_key
    assert captured_providers[0]["model_provider_options"] == settings.openrouter_model_provider_options
    first_adapter = cast(_FakeAdapter, first)
    assert captured_adapters == [("openrouter", first_adapter.delegate)]


def test_build_cached_llm_provider_registry_does_not_treat_openrouter_as_configured_custom_endpoint(
    monkeypatch,
) -> None:
    captured_openrouter: list[object] = []

    class _FakeOpenRouterProvider:
        def __init__(self, **kwargs: object) -> None:
            captured_openrouter.append(kwargs)

    monkeypatch.setattr(provider_factory, "OpenRouterLlmProvider", _FakeOpenRouterProvider)

    registry = provider_factory.build_cached_llm_provider_registry(
        llm_settings=LlmSettings(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=45.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    registry.resolve("openrouter")

    assert len(captured_openrouter) == 1
