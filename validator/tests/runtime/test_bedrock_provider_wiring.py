from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.routing import ResolvedLlmRoute, RoutedLlmProvider
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.tools.invocation_clients import build_tool_invocation_clients
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.bootstrap import _build_llm_clients
from harnyx_validator.runtime.settings import Settings


def _routed_surface(provider: object) -> str:
    assert isinstance(provider, RoutedLlmProvider)
    return provider._surface


class _RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[AbstractLlmRequest] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            id="test-response",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="output_text", text="ok"),),
                    ),
                ),
            ),
            usage=LlmUsage(),
            metadata={"raw_response": {"ok": True}},
        )

    async def aclose(self) -> None:
        return None


class _FakeRegistry:
    def __init__(self, provider: _RecordingProvider | None = None) -> None:
        self.provider = provider or _RecordingProvider()

    def resolve(self, name: str) -> _RecordingProvider:
        return self.provider


def _settings() -> Settings:
    return Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
            similarity_llm_provider="chutes",
            chutes_api_key=SecretStr("test-key"),
        ),
        bedrock=BedrockSettings.model_construct(
            region="us-east-1",
            connect_timeout_seconds=5.0,
            read_timeout_seconds=60.0,
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("scoring_llm_provider", "SCORING_LLM_PROVIDER='bedrock' is not supported"),
    ],
)
def test_validator_runtime_rejects_unsupported_bedrock_surfaces(field: str, message: str) -> None:
    settings = _settings()
    settings = settings.model_copy(update={"llm": settings.llm.model_copy(update={field: "bedrock"})})

    with pytest.raises(ValueError, match=message):
        _build_llm_clients(settings)


def test_validator_runtime_ignores_tool_bedrock_for_proxy_backed_tooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(update={"llm": settings.llm.model_copy(update={"tool_llm_provider": "bedrock"})})

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    clients = _build_llm_clients(settings)

    assert clients.tool_llm_provider is None


def test_local_tool_invocation_clients_still_reject_bedrock_tool_provider() -> None:
    settings = _settings()
    settings = settings.model_copy(update={"llm": settings.llm.model_copy(update={"tool_llm_provider": "bedrock"})})

    with pytest.raises(ValueError, match="TOOL_LLM_PROVIDER='bedrock' is not supported"):
        build_tool_invocation_clients(
            llm_settings=settings.llm,
            bedrock_settings=settings.bedrock,
            vertex_settings=settings.vertex,
        )


def test_validator_runtime_ignores_tool_override_to_bedrock_for_proxy_backed_tooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps({"tool": {"sample-tool-model": "bedrock"}}),
                }
            )
        }
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    clients = _build_llm_clients(settings)

    assert clients.tool_llm_provider is None


def test_local_tool_invocation_clients_still_reject_tool_override_to_bedrock() -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps({"tool": {"sample-tool-model": "bedrock"}}),
                }
            )
        }
    )

    with pytest.raises(ValueError, match="TOOL_LLM_PROVIDER='bedrock' is not supported"):
        build_tool_invocation_clients(
            llm_settings=settings.llm,
            bedrock_settings=settings.bedrock,
            vertex_settings=settings.vertex,
        )


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("explicit_model", ("moonshotai/Kimi-K2.5-TEE", "zai-org/GLM-5-TEE"))
async def test_validator_runtime_routes_explicit_scoring_request_to_bedrock(
    monkeypatch: pytest.MonkeyPatch,
    explicit_model: str,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps({"scoring": {explicit_model: "bedrock"}}),
                }
            )
        }
    )
    registry = _FakeRegistry()
    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: registry)

    clients = _build_llm_clients(settings)
    response = await clients.scoring_llm_provider.invoke(
        LlmRequest(
            provider=settings.llm.scoring_llm_provider,
            model=explicit_model,
            messages=(),
            temperature=None,
            max_output_tokens=128,
        )
    )

    assert _routed_surface(clients.scoring_llm_provider) == "scoring"
    assert registry.provider.requests[0].provider == "bedrock"
    assert registry.provider.requests[0].model == explicit_model
    assert response.metadata is not None
    assert response.metadata["selected_provider"] == "bedrock"
    assert response.metadata["selected_model"] == explicit_model


def test_validator_runtime_routes_configured_scoring_entries_to_custom_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring_routes = {
        "google/gemma-4-31B-turbo-TEE": "custom-openai-compatible:gemma4-cloud-run-turbo",
        "Qwen/Qwen3.6-27B-TEE": "custom-openai-compatible:qwen36-cloud-run",
    }
    settings = _settings()
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps({"scoring": scoring_routes}),
                    "openai_compatible_endpoints_json": json.dumps(
                        [
                            {
                                "id": "gemma4-cloud-run-turbo",
                                "base_url": "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app/v1",
                                "auth": {"type": "none"},
                            },
                            {
                                "id": "qwen36-cloud-run",
                                "base_url": "https://qwen3-6-27b-obbrpx3ppa-uc.a.run.app/v1",
                                "auth": {"type": "none"},
                            }
                        ]
                    ),
                }
            )
        }
    )

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    clients = _build_llm_clients(settings)

    assert _routed_surface(clients.scoring_llm_provider) == "scoring"
    assert clients.scoring_routes == {
        model: ResolvedLlmRoute(surface="scoring", provider=provider, model=model)
        for model, provider in scoring_routes.items()
    }


@pytest.mark.anyio("asyncio")
async def test_validator_runtime_routes_primary_similarity_model_to_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    similarity_route_target = "custom-openai-compatible:similarity-private-route"
    settings = _settings()
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "similarity_llm_provider": "bedrock",
                    "llm_model_provider_overrides_json": json.dumps(
                        {
                            "duplication_detection": {
                                model: similarity_route_target
                                for model in bootstrap._DUPLICATION_DETECTION_MODEL_CHAIN
                            }
                        }
                    ),
                    "openai_compatible_endpoints_json": json.dumps(
                        [
                            {
                                "id": "similarity-private-route",
                                "base_url": "https://similarity-private.example/v1",
                                "auth": {"type": "none"},
                            }
                        ]
                    ),
                }
            )
        }
    )

    registry = _FakeRegistry()
    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: registry)

    clients = _build_llm_clients(settings)
    response = await clients.similarity_llm_provider.invoke(
        LlmRequest(
            provider=settings.llm.similarity_llm_provider,
            model=bootstrap._DUPLICATION_DETECTION_LLM_MODEL,
            messages=(),
            temperature=None,
            max_output_tokens=128,
        )
    )

    assert _routed_surface(clients.similarity_llm_provider) == "duplication_detection"
    assert clients.similarity_route.provider == similarity_route_target
    assert clients.similarity_route.model == bootstrap._DUPLICATION_DETECTION_LLM_MODEL
    assert registry.provider.requests[0].provider == similarity_route_target
    assert registry.provider.requests[0].model == bootstrap._DUPLICATION_DETECTION_LLM_MODEL
    assert response.metadata is not None
    assert response.metadata["selected_provider"] == similarity_route_target
    assert response.metadata["selected_model"] == bootstrap._DUPLICATION_DETECTION_LLM_MODEL


def test_validator_runtime_routes_similarity_default_provider_independently_from_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "scoring_llm_provider": "vertex",
                    "similarity_llm_provider": "chutes",
                    "llm_model_provider_overrides_json": None,
                }
            )
        }
    )

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    clients = _build_llm_clients(settings)

    assert _routed_surface(clients.scoring_llm_provider) == "scoring"
    assert _routed_surface(clients.similarity_llm_provider) == "duplication_detection"
    assert {route.provider for route in clients.scoring_routes.values()} == {"vertex"}
    assert clients.similarity_route.provider == "chutes"
