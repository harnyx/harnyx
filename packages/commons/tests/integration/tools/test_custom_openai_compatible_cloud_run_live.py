from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings, OpenAiCompatibleGoogleIdTokenAuthConfig
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.json_utils import pydantic_postprocessor
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.schema import (
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmThinkingConfig,
)

pytestmark = [pytest.mark.integration]
_GEMMA_MODEL = "google/gemma-4-31B-turbo-TEE"
_GEMMA_ENDPOINT_ID = "gemma4-cloud-run-turbo"
_GEMMA_ROUTE_TARGET = "custom-openai-compatible:gemma4-cloud-run-turbo"
_GEMMA_SERVICE_URL = "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app"
_QWEN36_MODEL = "Qwen/Qwen3.6-27B-TEE"
_QWEN36_ENDPOINT_ID = "qwen36-cloud-run"
_QWEN36_ROUTE_TARGET = "custom-openai-compatible:qwen36-cloud-run"
_QWEN36_SERVICE_URL = "https://qwen3-6-27b-obbrpx3ppa-uc.a.run.app"


class JsonObjectAnswer(BaseModel):
    ping: str
    count: int


class ThrowawayStructuredAnswer(BaseModel):
    animal: str
    count: int
    approved: bool


def test_gemma_live_test_source_has_no_test_only_env_contract() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_env_names = (
        "GEMMA" + "_CLOUD_RUN_SERVICE_URL",
        "GEMMA" + "_CLOUD_RUN_MODEL",
    )
    assert not any(name in source for name in forbidden_env_names)


@pytest.mark.expensive
@pytest.mark.anyio("asyncio")
async def test_gemma_cloud_run_custom_openai_compatible_live() -> None:
    response = await _invoke_live_gemma(
        LlmRequest(
            provider="chutes",
            model=_GEMMA_MODEL,
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
                ),
            ),
            temperature=0.0,
            max_output_tokens=32,
            timeout_seconds=180.0,
        )
    )

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == _GEMMA_ROUTE_TARGET
    assert response.metadata["effective_model"] == _GEMMA_MODEL


@pytest.mark.expensive
@pytest.mark.anyio("asyncio")
async def test_gemma_cloud_run_custom_openai_compatible_scoring_route_live() -> None:
    response = await _invoke_live_gemma(
        LlmRequest(
            provider="chutes",
            model=_GEMMA_MODEL,
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
                ),
            ),
            temperature=0.0,
            max_output_tokens=32,
            timeout_seconds=180.0,
        ),
        surface="scoring",
    )

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == _GEMMA_ROUTE_TARGET
    assert response.metadata["effective_model"] == _GEMMA_MODEL


@pytest.mark.expensive
@pytest.mark.anyio("asyncio")
async def test_gemma_cloud_run_reasoning_effort_live() -> None:
    response = await _invoke_live_gemma(
        LlmRequest(
            provider="chutes",
            model=_GEMMA_MODEL,
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text('Think briefly, then reply with only "ok".'),),
                ),
            ),
            temperature=0.0,
            max_output_tokens=128,
            timeout_seconds=180.0,
            reasoning_effort="high",
        )
    )

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == _GEMMA_ROUTE_TARGET
    assert response.metadata["effective_model"] == _GEMMA_MODEL
    assert response.choices[0].message.reasoning or (
        response.usage.reasoning_tokens is not None and response.usage.reasoning_tokens > 0
    )


@pytest.mark.expensive
@pytest.mark.anyio("asyncio")
async def test_qwen36_cloud_run_custom_openai_compatible_live() -> None:
    response = await _invoke_live_tool_model(
        model=_QWEN36_MODEL,
        endpoint_id=_QWEN36_ENDPOINT_ID,
        route_target=_QWEN36_ROUTE_TARGET,
        prompt='Reply with only "ok".',
        max_output_tokens=32,
    )

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == _QWEN36_ROUTE_TARGET
    assert response.metadata["effective_model"] == _QWEN36_MODEL


@pytest.mark.expensive
@pytest.mark.anyio("asyncio")
async def test_gemma_cloud_run_json_object_live() -> None:
    response = await _invoke_live_gemma(
        LlmRequest(
            provider="chutes",
            model=_GEMMA_MODEL,
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text('Return JSON only with {"ping":"pong","count":2}.'),),
                ),
            ),
            temperature=0.0,
            max_output_tokens=128,
            timeout_seconds=180.0,
            output_mode="json_object",
            postprocessor=pydantic_postprocessor(JsonObjectAnswer),
        )
    )

    parsed = response.postprocessed
    assert isinstance(parsed, JsonObjectAnswer)
    assert parsed.ping == "pong"
    assert parsed.count == 2
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == _GEMMA_ROUTE_TARGET
    assert response.metadata["effective_model"] == _GEMMA_MODEL


@pytest.mark.expensive
@pytest.mark.anyio("asyncio")
async def test_gemma_cloud_run_json_schema_live_with_throwaway_schema() -> None:
    response = await _invoke_live_gemma(
        LlmRequest(
            provider="chutes",
            model=_GEMMA_MODEL,
            messages=(
                LlmMessage(
                    role="user",
                    content=(
                        LlmMessageContentPart.input_text(
                            'Return JSON only with {"animal":"otter","count":3,"approved":true}.'
                        ),
                    ),
                ),
            ),
            temperature=0.0,
            max_output_tokens=128,
            timeout_seconds=180.0,
            output_mode="structured",
            output_schema=ThrowawayStructuredAnswer,
            postprocessor=pydantic_postprocessor(ThrowawayStructuredAnswer),
        )
    )

    parsed = response.postprocessed
    assert isinstance(parsed, ThrowawayStructuredAnswer)
    assert parsed.animal == "otter"
    assert parsed.count == 3
    assert parsed.approved is True
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == _GEMMA_ROUTE_TARGET
    assert response.metadata["effective_model"] == _GEMMA_MODEL


def test_gemma_live_settings_use_test_endpoint_and_route() -> None:
    settings = _build_live_gemma_settings(
        {
            "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64": "present",
        }
    )

    assert "gemma4-cloud-run-turbo" in settings.openai_compatible_endpoints
    auth = settings.openai_compatible_endpoints["gemma4-cloud-run-turbo"].auth
    assert isinstance(auth, OpenAiCompatibleGoogleIdTokenAuthConfig)
    assert auth.audience == _GEMMA_SERVICE_URL
    assert auth.credential_source == "service_account_json_b64_env"
    assert auth.credential_env == "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64"
    assert settings.llm_model_provider_overrides["tool"][_GEMMA_MODEL] == _GEMMA_ROUTE_TARGET
    assert settings.llm_model_provider_overrides["scoring"][_GEMMA_MODEL] == _GEMMA_ROUTE_TARGET
    assert settings.llm_model_provider_overrides["duplication_detection"][_GEMMA_MODEL] == _GEMMA_ROUTE_TARGET


def test_qwen36_live_settings_use_test_endpoint_and_route() -> None:
    settings = _build_live_settings(
        {
            "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64": "present",
        },
        required_model=_QWEN36_MODEL,
        required_endpoint_id=_QWEN36_ENDPOINT_ID,
        required_route=_QWEN36_ROUTE_TARGET,
        service_url=_QWEN36_SERVICE_URL,
    )

    assert "qwen36-cloud-run" in settings.openai_compatible_endpoints
    auth = settings.openai_compatible_endpoints["qwen36-cloud-run"].auth
    assert isinstance(auth, OpenAiCompatibleGoogleIdTokenAuthConfig)
    assert auth.audience == _QWEN36_SERVICE_URL
    assert auth.credential_source == "service_account_json_b64_env"
    assert auth.credential_env == "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64"
    assert settings.llm_model_provider_overrides["tool"][_QWEN36_MODEL] == _QWEN36_ROUTE_TARGET


def test_qwen36_live_settings_requires_service_account_credentials() -> None:
    with pytest.raises(RuntimeError, match="GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64 must be configured"):
        _build_live_settings(
            {},
            required_model=_QWEN36_MODEL,
            required_endpoint_id=_QWEN36_ENDPOINT_ID,
            required_route=_QWEN36_ROUTE_TARGET,
            service_url=_QWEN36_SERVICE_URL,
        )


def _build_live_gemma_settings(environ: Mapping[str, str]) -> LlmSettings:
    return _build_live_settings(
        environ,
        required_model=_GEMMA_MODEL,
        required_endpoint_id=_GEMMA_ENDPOINT_ID,
        required_route=_GEMMA_ROUTE_TARGET,
        service_url=_GEMMA_SERVICE_URL,
    )


def _build_live_settings(
    environ: Mapping[str, str],
    *,
    required_model: str,
    required_endpoint_id: str,
    required_route: str,
    service_url: str,
) -> LlmSettings:
    _require_mapping_env(environ, "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64")
    settings = LlmSettings(
        LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON=json.dumps(
            [_cloud_run_endpoint_config(required_endpoint_id, service_url)]
        ),
        LLM_MODEL_PROVIDER_OVERRIDES_JSON=json.dumps(
            {
                "tool": {required_model: required_route},
                "scoring": {required_model: required_route},
                "duplication_detection": {required_model: required_route},
            }
        ),
    )
    _require_cloud_run_google_id_token_auth(settings, required_endpoint_id)
    return settings


def _cloud_run_endpoint_config(endpoint_id: str, service_url: str) -> dict[str, object]:
    return {
        "id": endpoint_id,
        "base_url": f"{service_url}/v1",
        "auth": {
            "type": "google_id_token",
            "audience": service_url,
            "credential_source": "service_account_json_b64_env",
            "credential_env": "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
        },
    }


def _require_cloud_run_google_id_token_auth(settings: LlmSettings, endpoint_id: str) -> None:
    endpoint = settings.openai_compatible_endpoints.get(endpoint_id)
    if endpoint is None:
        raise RuntimeError(f"LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON must include endpoint id {endpoint_id}")
    auth = endpoint.auth
    if not isinstance(auth, OpenAiCompatibleGoogleIdTokenAuthConfig):
        raise RuntimeError(f"OpenAI-compatible endpoint {endpoint_id} must use google_id_token auth")
    expected_audience = _cloud_run_audience_from_base_url(str(endpoint.base_url))
    if auth.audience != expected_audience:
        raise RuntimeError(
            f"OpenAI-compatible endpoint {endpoint_id} google_id_token audience must be {expected_audience}"
        )
    if auth.credential_source != "service_account_json_b64_env":
        raise RuntimeError(
            f"OpenAI-compatible endpoint {endpoint_id} must use service_account_json_b64_env credentials"
        )
    if auth.credential_env != "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64":
        raise RuntimeError(
            f"OpenAI-compatible endpoint {endpoint_id} credential_env must be GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64"
        )


def _cloud_run_audience_from_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if not stripped.endswith("/v1"):
        raise RuntimeError("Cloud Run OpenAI-compatible endpoint base_url must end with /v1")
    return stripped[: -len("/v1")]


async def _invoke_live_tool_model(
    *,
    model: str,
    endpoint_id: str,
    route_target: str,
    prompt: str,
    max_output_tokens: int,
) -> LlmResponse:
    request = LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text(prompt),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        thinking=LlmThinkingConfig(enabled=False),
        timeout_seconds=180.0,
    )
    settings = _build_live_settings(
        os.environ,
        required_model=model,
        required_endpoint_id=endpoint_id,
        required_route=route_target,
        service_url=_QWEN36_SERVICE_URL,
    )
    return await _invoke_live_request(settings=settings, request=request)


async def _invoke_live_gemma(
    request: LlmRequest,
    *,
    surface: Literal["tool", "scoring", "duplication_detection"] = "tool",
) -> LlmResponse:
    settings = _build_live_gemma_settings(os.environ)
    return await _invoke_live_request(settings=settings, request=request, surface=surface)


async def _invoke_live_request(
    *,
    settings: LlmSettings,
    request: LlmRequest,
    surface: Literal["tool", "scoring", "duplication_detection"] = "tool",
) -> LlmResponse:
    registry = build_cached_llm_provider_registry(
        llm_settings=settings,
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64="",
        ),
    )
    provider = build_routed_llm_provider(
        surface=surface,
        default_provider="chutes",
        llm_settings=settings,
        allowed_providers={"bedrock", "chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )

    try:
        return await provider.invoke(request)
    finally:
        await registry.aclose()


def _require_mapping_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value
