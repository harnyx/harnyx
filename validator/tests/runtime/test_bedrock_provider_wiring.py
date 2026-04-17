from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_validator.infrastructure.scoring.factory import create_scoring_embedding_client
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.bootstrap import _build_llm_clients, _build_local_eval_tooling_clients
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
            vertex_maas_gcp_location="us-east5",
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
    with pytest.raises(ValueError, match=message):
        _build_local_eval_tooling_clients(settings)


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
        _build_local_eval_tooling_clients(settings)


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

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    _, _, scoring_provider, scoring_route = _build_llm_clients(settings)
    _, _, local_scoring_provider, local_scoring_route = _build_local_eval_tooling_clients(settings)

    assert scoring_provider == "provider:bedrock"
    assert scoring_route.provider == "bedrock"
    assert scoring_route.model == bootstrap._SCORING_LLM_MODEL
    assert local_scoring_provider == "provider:bedrock"
    assert local_scoring_route == scoring_route


def test_validator_scoring_embedding_factory_rejects_bedrock() -> None:
    with pytest.raises(ValueError, match="SCORING_LLM_PROVIDER='bedrock' is not supported"):
        create_scoring_embedding_client(
            provider_name="bedrock",
            vertex_model="gemini-embedding-001",
            chutes_model="Qwen/Qwen3-Embedding-0.6B",
            chutes_api_key="test-key",
            scoring_timeout_seconds=30.0,
            vertex_project="project",
            vertex_location="us-central1",
            vertex_maas_location="us-east5",
            vertex_service_account_b64="vertex-creds",
            vertex_timeout_seconds=60.0,
        )
