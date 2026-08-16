from __future__ import annotations

import json
from uuid import uuid4

import pytest

from harnyx_commons.llm.provider import LlmProviderPort, LlmRetryExhaustedError
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest
from harnyx_validator.application.similarity_judge import SimilarityJudge, SimilarityJudgeConfig
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.expensive,
    pytest.mark.anyio("asyncio"),
    pytest.mark.flaky(reruns=1, only_rerun=[LlmRetryExhaustedError]),
]
_GEMMA_MODEL = "google/gemma-4-31B-turbo-TEE"
_GEMMA_ENDPOINT_ID = "gemma4-cloud-run-turbo"
_GEMMA_ROUTE_TARGET = f"custom-openai-compatible:{_GEMMA_ENDPOINT_ID}"
_GEMMA_SERVICE_URL = "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app"
_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731-TEE"
_DEEPSEEK_ROUTE_TARGET = "chutes"
_DEEPSEEK_OPENROUTER_ROUTE_TARGET = "openrouter"
_KIMI_MODEL = "moonshotai/Kimi-K2.5-TEE"
_KIMI_ROUTE_TARGET = "bedrock"
_GLM_MODEL = "zai-org/GLM-5-TEE"
_GLM_ROUTE_TARGET = "vertex"
_REFERENCE_SCRIPT = (
    "def query(agent, question):\n"
    "    messages = [question]\n"
    "    while True:\n"
    "        response = agent.run(messages)\n"
    "        if response.final_answer:\n"
    "            return response.final_answer\n"
    "        messages.extend(agent.execute_tools(response.tool_calls))\n"
)
_CANDIDATE_DIFF = (
    "--- incumbent\n"
    "+++ candidate\n"
    "@@\n"
    "-def query(agent, question):\n"
    "-    messages = [question]\n"
    "-    while True:\n"
    "-        response = agent.run(messages)\n"
    "-        if response.final_answer:\n"
    "-            return response.final_answer\n"
    "-        messages.extend(agent.execute_tools(response.tool_calls))\n"
    "+def query(pipeline, question):\n"
    "+    plan = pipeline.plan_claims(question)\n"
    "+    retrieved = pipeline.retrieve_claims_in_parallel(plan)\n"
    "+    fact_table = pipeline.verify_into_fact_table(retrieved)\n"
    "+    return pipeline.synthesize_from_verified_facts(question, fact_table)\n"
)


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


@pytest.mark.parametrize(
    ("model", "route_target"),
    (
        (_GEMMA_MODEL, _GEMMA_ROUTE_TARGET),
        (_DEEPSEEK_MODEL, _DEEPSEEK_ROUTE_TARGET),
        (_DEEPSEEK_MODEL, _DEEPSEEK_OPENROUTER_ROUTE_TARGET),
        (_KIMI_MODEL, _KIMI_ROUTE_TARGET),
        (_GLM_MODEL, _GLM_ROUTE_TARGET),
    ),
)
async def test_similarity_judge_live_supports_production_provider_contract(
    model: str,
    route_target: str,
) -> None:
    base_settings = Settings.load()
    route_overrides = {
        _GEMMA_MODEL: _GEMMA_ROUTE_TARGET,
        _DEEPSEEK_MODEL: _DEEPSEEK_ROUTE_TARGET,
        _KIMI_MODEL: _KIMI_ROUTE_TARGET,
        _GLM_MODEL: _GLM_ROUTE_TARGET,
    }
    route_overrides[model] = route_target
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={
                    "openai_compatible_endpoints_json": json.dumps([_gemma_cloud_run_endpoint_config()]),
                    "llm_model_provider_overrides_json": json.dumps(
                        {
                            "duplication_detection": route_overrides
                        }
                    ),
                    "similarity_llm_model_override": model,
                }
            )
        }
    )
    similarity_route = bootstrap._resolve_similarity_judge_route(settings)
    assert similarity_route.provider == route_target
    assert similarity_route.model == model

    registry = build_cached_llm_provider_registry(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    routed_provider = build_routed_llm_provider(
        surface="duplication_detection",
        default_provider=settings.llm.similarity_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "openrouter", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )
    llm_provider = RecordingProvider(routed_provider)
    service = SimilarityJudge(
        llm_provider=llm_provider,
        config=SimilarityJudgeConfig(
            provider=settings.llm.similarity_llm_provider,
            model=similarity_route.model,
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            temperature=settings.llm.similarity_llm_temperature,
            max_output_tokens=settings.llm.similarity_llm_max_output_tokens,
            timeout_seconds=float(settings.llm.similarity_llm_timeout_seconds),
            retry_policy=settings.llm.similarity_llm_retry_policy,
            request_extra_by_model=bootstrap._similarity_request_extra_by_model(
                (similarity_route,)
            ),
        ),
    )
    request = SimilarityJudgeRequest(
        batch_id=uuid4(),
        candidate_artifact_id=uuid4(),
        reference_artifact_id=uuid4(),
        candidate_miner_uid=2,
        reference_miner_uid=1,
        reference_script=_REFERENCE_SCRIPT,
        candidate_diff=_CANDIDATE_DIFF,
    )

    try:
        result = await service.judge(request)
    finally:
        await registry.aclose()

    print(
        json.dumps(
            {
                "event": "similarity_judge.provider_contract",
                "model": model,
                "provider": route_target,
                "classification": result.classification,
            },
            sort_keys=True,
        )
    )

    assert len(llm_provider.requests) == 1
    assert len(llm_provider.responses) == 1
    llm_request = llm_provider.requests[0]
    response = llm_provider.responses[0]
    assert result.model == similarity_route.model
    assert result.provider == similarity_route.provider
    assert result.reasoning and result.reasoning.strip()
    assert result.reasoning_tokens is None or result.reasoning_tokens >= 0
    assert llm_request.output_mode == "structured"
    assert llm_request.provider == settings.llm.similarity_llm_provider
    assert llm_request.model == similarity_route.model
    assert llm_request.reasoning_effort == "high"
    assert llm_request.max_output_tokens == settings.llm.similarity_llm_max_output_tokens
    assert llm_request.timeout_seconds == settings.llm.similarity_llm_timeout_seconds
    assert llm_request.retry_policy == settings.llm.similarity_llm_retry_policy
    assert llm_request.thinking is None
    assert llm_request.use_case == "miner_task_similarity_judge"
    expected_extra_by_model = bootstrap._similarity_request_extra_by_model(
        (similarity_route,)
    )
    assert llm_request.extra == expected_extra_by_model.get(model)
    if model == _DEEPSEEK_MODEL:
        assert response.choices[0].message.reasoning
        assert result.reasoning_tokens is not None
        assert result.reasoning_tokens > 1
    assert response.metadata is not None
    assert response.metadata["selected_provider"] == similarity_route.provider
    assert response.metadata["selected_model"] == similarity_route.model
