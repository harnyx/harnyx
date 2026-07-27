from __future__ import annotations

import json
import os
from collections.abc import Mapping
from uuid import uuid4

import pytest

from harnyx_commons.config.llm import LlmSettings, OpenAiCompatibleGoogleIdTokenAuthConfig
from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer, Response
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.miner_task_scoring import (
    EvaluationScoringConfig,
    EvaluationScoringService,
)
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]
_GEMMA_MODEL = "google/gemma-4-31B-turbo-TEE"
_GEMMA_ENDPOINT_ID = "gemma4-cloud-run-turbo"
_GEMMA_ROUTE_TARGET = "custom-openai-compatible:gemma4-cloud-run-turbo"
_GEMMA_SERVICE_URL = "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app"
_QWEN36_MODEL = "Qwen/Qwen3.6-27B-TEE"
_QWEN36_ENDPOINT_ID = "qwen36-cloud-run"
_QWEN36_ROUTE_TARGET = "custom-openai-compatible:qwen36-cloud-run"
_QWEN36_SERVICE_URL = "https://qwen3-6-27b-obbrpx3ppa-uc.a.run.app"
_GLM_MODEL = "zai-org/GLM-5-TEE"
_GLM_ROUTE_TARGET = "vertex"
_KIMI_MODEL = "moonshotai/Kimi-K2.5-TEE"
_KIMI_ROUTE_TARGET = "bedrock"


class RecordingProvider(LlmProviderPort):
    def __init__(self, delegate: LlmProviderPort) -> None:
        self._delegate = delegate
        self.requests: list[AbstractLlmRequest] = []
        self.responses: list[LlmResponse] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        response = await self._delegate.invoke(request)
        self.responses.append(response)
        return response

    async def aclose(self) -> None:
        await self._delegate.aclose()


@pytest.mark.parametrize(
    ("model", "endpoint_id", "route_target", "service_url"),
    (
        (_GEMMA_MODEL, _GEMMA_ENDPOINT_ID, _GEMMA_ROUTE_TARGET, _GEMMA_SERVICE_URL),
        (_QWEN36_MODEL, _QWEN36_ENDPOINT_ID, _QWEN36_ROUTE_TARGET, _QWEN36_SERVICE_URL),
    ),
)
async def test_evaluation_scoring_live_uses_real_structured_runtime_flow(
    model: str,
    endpoint_id: str,
    route_target: str,
    service_url: str,
) -> None:
    base_settings = Settings.load()
    settings = base_settings.model_copy(
        update={
            "llm": _build_live_scoring_settings(
                os.environ,
                base_settings.llm,
                model=model,
                endpoint_id=endpoint_id,
                route_target=route_target,
                service_url=service_url,
            ),
        }
    )
    scoring_route = bootstrap._resolve_scoring_judge_route(settings, model=model)
    assert scoring_route.provider == route_target
    assert scoring_route.model == model

    registry = build_cached_llm_provider_registry(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    routed_provider = build_routed_llm_provider(
        surface="scoring",
        default_provider=settings.llm.scoring_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )
    llm_provider = RecordingProvider(routed_provider)
    service = EvaluationScoringService(
        llm_provider=llm_provider,
        config=EvaluationScoringConfig(
            provider=settings.llm.scoring_llm_provider,
            model=scoring_route.model,
            fallback_models=(),
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            temperature=0.0,
            max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
            timeout_seconds=float(settings.llm.scoring_llm_timeout_seconds),
        ),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the capital of France?"),
        reference_answer=ReferenceAnswer(text="Paris is the capital of France."),
    )

    try:
        score = await service.score(
            task=task,
            response=Response(text="Paris is the capital of France."),
        )
    finally:
        await registry.aclose()

    assert len(llm_provider.requests) == 2
    assert all(request.output_mode == "structured" for request in llm_provider.requests)
    assert all(request.provider == settings.llm.scoring_llm_provider for request in llm_provider.requests)
    assert all(request.model == scoring_route.model for request in llm_provider.requests)
    assert all(response.metadata is not None for response in llm_provider.responses)
    assert all(response.metadata["selected_provider"] == scoring_route.provider for response in llm_provider.responses)
    assert all(response.metadata["selected_model"] == scoring_route.model for response in llm_provider.responses)
    assert score.scoring_version == "v1"
    assert 0.0 <= score.comparison_score <= 1.0
    assert score.total_score == pytest.approx(score.comparison_score)
    observed_reasoning = [
        response.choices[0].message.reasoning
        for response in llm_provider.responses
        if response.choices and response.choices[0].message.reasoning is not None
    ]
    if observed_reasoning:
        assert score.reasoning is not None
        assert score.reasoning.text is not None
        assert score.reasoning.text.strip()
        if score.reasoning.reasoning_tokens is not None:
            assert score.reasoning.reasoning_tokens >= 0


@pytest.mark.parametrize(
    ("model", "route_target"),
    (
        (_GLM_MODEL, _GLM_ROUTE_TARGET),
        (_KIMI_MODEL, _KIMI_ROUTE_TARGET),
    ),
)
async def test_evaluation_scoring_live_accepts_fallback_candidate_route(
    model: str,
    route_target: str,
) -> None:
    base_settings = Settings.load()
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps({"scoring": {model: route_target}}),
                }
            ),
        }
    )
    scoring_route = bootstrap._resolve_scoring_judge_route(settings, model=model)
    assert scoring_route.provider == route_target
    assert scoring_route.model == model

    registry = build_cached_llm_provider_registry(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    routed_provider = build_routed_llm_provider(
        surface="scoring",
        default_provider=settings.llm.scoring_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )
    llm_provider = RecordingProvider(routed_provider)
    service = EvaluationScoringService(
        llm_provider=llm_provider,
        config=EvaluationScoringConfig(
            provider=settings.llm.scoring_llm_provider,
            model=scoring_route.model,
            fallback_models=(),
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            temperature=0.0,
            max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
            timeout_seconds=float(settings.llm.scoring_llm_timeout_seconds),
        ),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the capital of France?"),
        reference_answer=ReferenceAnswer(text="Paris is the capital of France."),
    )

    try:
        score = await service.score(
            task=task,
            response=Response(text="Paris is the capital of France."),
        )
    finally:
        await registry.aclose()

    assert len(llm_provider.requests) == 2
    assert all(request.output_mode == "structured" for request in llm_provider.requests)
    assert all(request.provider == settings.llm.scoring_llm_provider for request in llm_provider.requests)
    assert all(request.model == scoring_route.model for request in llm_provider.requests)
    assert all(response.metadata is not None for response in llm_provider.responses)
    assert all(response.metadata["selected_provider"] == route_target for response in llm_provider.responses)
    assert all(response.metadata["selected_model"] == scoring_route.model for response in llm_provider.responses)
    assert score.scoring_version == "v1"
    assert 0.0 <= score.comparison_score <= 1.0
    assert score.total_score == pytest.approx(score.comparison_score)


def _build_live_scoring_settings(
    environ: Mapping[str, str],
    base_settings: LlmSettings,
    *,
    model: str,
    endpoint_id: str,
    route_target: str,
    service_url: str,
) -> LlmSettings:
    _require_mapping_env(environ, "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64")
    settings = base_settings.model_copy(
        update={
            "openai_compatible_endpoints_json": json.dumps(
                [_cloud_run_endpoint_config(endpoint_id, service_url)]
            ),
            "llm_model_provider_overrides_json": json.dumps({"scoring": {model: route_target}}),
        }
    )
    _require_cloud_run_google_id_token_auth(settings, endpoint_id=endpoint_id, service_url=service_url)
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


def _require_cloud_run_google_id_token_auth(
    settings: LlmSettings,
    *,
    endpoint_id: str,
    service_url: str,
) -> None:
    endpoint = settings.openai_compatible_endpoints.get(endpoint_id)
    if endpoint is None:
        raise RuntimeError(f"LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON must include endpoint id {endpoint_id}")
    auth = endpoint.auth
    if not isinstance(auth, OpenAiCompatibleGoogleIdTokenAuthConfig):
        raise RuntimeError(f"OpenAI-compatible endpoint {endpoint_id} must use google_id_token auth")
    if auth.audience != service_url:
        raise RuntimeError(
            f"OpenAI-compatible endpoint {endpoint_id} google_id_token audience must be {service_url}"
        )
    if auth.credential_source != "service_account_json_b64_env":
        raise RuntimeError(
            f"OpenAI-compatible endpoint {endpoint_id} must use service_account_json_b64_env credentials"
        )
    if auth.credential_env != "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64":
        raise RuntimeError(
            f"OpenAI-compatible endpoint {endpoint_id} credential_env must be "
            "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64"
        )


def _require_mapping_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value
