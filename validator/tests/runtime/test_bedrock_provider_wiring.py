from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.tools import invocation_clients
from harnyx_commons.tools.invocation_clients import build_tool_invocation_clients
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.bootstrap import _build_llm_clients
from harnyx_validator.runtime.settings import Settings


def _settings() -> Settings:
    return Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
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
        ("tool_llm_provider", "TOOL_LLM_PROVIDER='bedrock' is not supported"),
        ("scoring_llm_provider", "SCORING_LLM_PROVIDER='bedrock' is not supported"),
    ],
)
def test_validator_runtime_rejects_unsupported_bedrock_surfaces(field: str, message: str) -> None:
    settings = _settings()
    settings = settings.model_copy(update={"llm": settings.llm.model_copy(update={field: "bedrock"})})

    with pytest.raises(ValueError, match=message):
        _build_llm_clients(settings)
    if field == "tool_llm_provider":
        with pytest.raises(ValueError, match=message):
            build_tool_invocation_clients(
                llm_settings=settings.llm,
                bedrock_settings=settings.bedrock,
                vertex_settings=settings.vertex,
            )
    else:
        clients = build_tool_invocation_clients(
            llm_settings=settings.llm,
            bedrock_settings=settings.bedrock,
            vertex_settings=settings.vertex,
        )
        assert clients.search_client is not None
        assert clients.tool_llm_provider is not None


def test_validator_runtime_rejects_tool_override_to_bedrock() -> None:
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
        _build_llm_clients(settings)
    with pytest.raises(ValueError, match="TOOL_LLM_PROVIDER='bedrock' is not supported"):
        build_tool_invocation_clients(
            llm_settings=settings.llm,
            bedrock_settings=settings.bedrock,
            vertex_settings=settings.vertex,
        )


def test_validator_runtime_allows_scoring_override_to_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps(
                        {"scoring": {bootstrap._SCORING_LLM_MODEL: "bedrock"}}
                    ),
                }
            )
        }
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(invocation_clients, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    _, _, _, scoring_provider, _, scoring_route, _ = _build_llm_clients(settings)

    assert scoring_provider == "provider:bedrock"
    assert scoring_route.provider == "bedrock"
    assert scoring_route.model == bootstrap._SCORING_LLM_MODEL


def test_validator_runtime_allows_duplication_detection_override_to_bedrock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps(
                        {"duplication_detection": {bootstrap._DUPLICATION_DETECTION_LLM_MODEL: "bedrock"}}
                    ),
                }
            )
        }
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(invocation_clients, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    _, _, _, _, similarity_provider, _, similarity_route = _build_llm_clients(settings)

    assert similarity_provider == "provider:bedrock"
    assert similarity_route.provider == "bedrock"
    assert similarity_route.model == bootstrap._DUPLICATION_DETECTION_LLM_MODEL
