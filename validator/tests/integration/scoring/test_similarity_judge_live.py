from __future__ import annotations

import json
from uuid import uuid4

import pytest

from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest
from harnyx_validator.application.similarity_judge import SimilarityJudge, SimilarityJudgeConfig
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]
_GEMMA_ENDPOINT_ID = "gemma4-cloud-run-turbo"
_GEMMA_ROUTE_TARGET = f"custom-openai-compatible:{_GEMMA_ENDPOINT_ID}"
_GEMMA_SERVICE_URL = "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app"


def _gemma_cloud_run_endpoint_config() -> dict[str, object]:
    return {
        "id": _GEMMA_ENDPOINT_ID,
        "base_url": f"{_GEMMA_SERVICE_URL}/v1",
        "auth": {
            "type": "google_id_token",
            "audience": _GEMMA_SERVICE_URL,
            "credential_source": "service_account_json_b64_env",
            "credential_env": "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
        },
    }


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


async def test_similarity_judge_live_uses_real_structured_runtime_flow() -> None:
    base_settings = Settings.load()
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={
                    "openai_compatible_endpoints_json": json.dumps([_gemma_cloud_run_endpoint_config()]),
                    "llm_model_provider_overrides_json": json.dumps(
                        {"duplication_detection": {bootstrap._DUPLICATION_DETECTION_LLM_MODEL: _GEMMA_ROUTE_TARGET}}
                    )
                }
            )
        }
    )
    similarity_route = bootstrap._resolve_similarity_judge_route(settings)
    assert similarity_route.provider == _GEMMA_ROUTE_TARGET

    registry = build_cached_llm_provider_registry(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    routed_provider = build_routed_llm_provider(
        surface="duplication_detection",
        default_provider=settings.llm.scoring_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )
    llm_provider = RecordingProvider(routed_provider)
    service = SimilarityJudge(
        llm_provider=llm_provider,
        config=SimilarityJudgeConfig(
            provider=settings.llm.scoring_llm_provider,
            model=similarity_route.model,
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            temperature=0.0,
            max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
            timeout_seconds=float(settings.llm.scoring_llm_timeout_seconds),
        ),
    )
    request = SimilarityJudgeRequest(
        batch_id=uuid4(),
        candidate_artifact_id=uuid4(),
        incumbent_artifact_id=uuid4(),
        candidate_miner_uid=2,
        incumbent_miner_uid=1,
        incumbent_script="def run():\n    return 'always answer with Paris'\n",
        candidate_diff=(
            "--- incumbent\n"
            "+++ candidate\n"
            "@@\n"
            "-def run():\n"
            "-    return 'always answer with Paris'\n"
            "+def run():\n"
            "+    return 'answer from retrieved sources with citations'\n"
        ),
    )

    try:
        result = await service.judge(request)
    finally:
        await registry.aclose()

    assert result.verdict in {"not_duplicate", "duplicate"}
    assert result.model == similarity_route.model
    assert result.provider == similarity_route.provider
    assert len(llm_provider.requests) == 1
    llm_request = llm_provider.requests[0]
    assert llm_request.output_mode == "structured"
    assert llm_request.provider == settings.llm.scoring_llm_provider
    assert llm_request.model == similarity_route.model
    assert llm_request.use_case == "miner_task_similarity_judge"
    assert llm_provider.responses[0].metadata is not None
    assert llm_provider.responses[0].metadata["selected_provider"] == similarity_route.provider
    assert llm_provider.responses[0].metadata["selected_model"] == similarity_route.model
    observed_reasoning = [
        response.choices[0].message.reasoning
        for response in llm_provider.responses
        if response.choices and response.choices[0].message.reasoning is not None
    ]
    if observed_reasoning:
        assert result.reasoning is not None
        assert result.reasoning.strip()
        if result.reasoning_tokens is not None:
            assert result.reasoning_tokens >= 0
