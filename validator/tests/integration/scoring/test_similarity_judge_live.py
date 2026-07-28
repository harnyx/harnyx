from __future__ import annotations

import json
from collections import Counter
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
_CALIBRATION_SAMPLE_COUNT = 2
_GEMMA_MODEL = "google/gemma-4-31B-turbo-TEE"
_GEMMA_ENDPOINT_ID = "gemma4-cloud-run-turbo"
_GEMMA_ROUTE_TARGET = f"custom-openai-compatible:{_GEMMA_ENDPOINT_ID}"
_GEMMA_SERVICE_URL = "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app"
_KIMI_MODEL = "moonshotai/Kimi-K2.5-TEE"
_KIMI_ROUTE_TARGET = "bedrock"
_GLM_MODEL = "zai-org/GLM-5-TEE"
_GLM_ROUTE_TARGET = "bedrock"


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
    ("reference_script", "candidate_diff", "expected_classification"),
    (
        pytest.param(
            (
                "MAX_RETRIES = 1\n\n"
                "def query(client, question):\n"
                "    for _ in range(MAX_RETRIES):\n"
                "        try:\n"
                "            return client.search(question)\n"
                "        except TimeoutError:\n"
                "            pass\n"
                "    return []\n"
            ),
            (
                "--- incumbent\n"
                "+++ candidate\n"
                "@@\n"
                "-MAX_RETRIES = 1\n"
                "+MAX_RETRIES = 5\n"
            ),
            "duplicate",
            id="parameter-only-duplicate",
        ),
        pytest.param(
            (
                "def query(client, question):\n"
                "    sources = client.search(question)\n"
                "    return client.synthesize(question, sources)\n"
            ),
            (
                "--- incumbent\n"
                "+++ candidate\n"
                "@@\n"
                "+def rank_by_authority(sources):\n"
                "+    return sorted(sources, key=lambda source: source.authority, reverse=True)\n"
                "+\n"
                "+def keep_current_sources(sources):\n"
                "+    return [source for source in sources if source.is_current]\n"
                "+\n"
                "+def remove_contradictions(sources):\n"
                "+    return [source for source in sources if not source.contradicted]\n"
                "+\n"
                " def query(client, question):\n"
                "-    sources = client.search(question)\n"
                "-    return client.synthesize(question, sources)\n"
                "+    sources = rank_by_authority(client.search(question))\n"
                "+    sources = keep_current_sources(sources)\n"
                "+    sources = remove_contradictions(sources)\n"
                "+    return client.synthesize(question, sources)\n"
            ),
            "near_duplicate",
            id="localized-multi-change-near-duplicate",
        ),
        pytest.param(
            (
                "def remove_contradictions(sources):\n"
                "    return [source for source in sources if not source.contradicted]\n"
                "\n"
                "def query(client, question):\n"
                "    sources = client.search(question)\n"
                "    sources = remove_contradictions(sources)\n"
                "    return client.synthesize(question, sources)\n"
            ),
            (
                "--- incumbent\n"
                "+++ candidate\n"
                "@@\n"
                "-def remove_contradictions(sources):\n"
                "-    return [source for source in sources if not source.contradicted]\n"
                "-\n"
                " def query(client, question):\n"
                "     sources = client.search(question)\n"
                "-    sources = remove_contradictions(sources)\n"
                "     return client.synthesize(question, sources)\n"
            ),
            "near_duplicate",
            id="removed-verification-policy-near-duplicate",
        ),
        pytest.param(
            (
                "def query(agent, question):\n"
                "    messages = [question]\n"
                "    while True:\n"
                "        response = agent.run(messages)\n"
                "        if response.final_answer:\n"
                "            return response.final_answer\n"
                "        messages.extend(agent.execute_tools(response.tool_calls))\n"
            ),
            (
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
            ),
            "novel",
            id="primary-flow-replacement-novel",
        ),
    ),
)
@pytest.mark.parametrize(
    ("model", "route_target"),
    (
        (_GEMMA_MODEL, _GEMMA_ROUTE_TARGET),
        (_KIMI_MODEL, _KIMI_ROUTE_TARGET),
        (_GLM_MODEL, _GLM_ROUTE_TARGET),
    ),
)
async def test_similarity_judge_live_reports_pairwise_classification_calibration(
    reference_script: str,
    candidate_diff: str,
    expected_classification: str,
    model: str,
    route_target: str,
) -> None:
    base_settings = Settings.load()
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={
                    "openai_compatible_endpoints_json": json.dumps([_gemma_cloud_run_endpoint_config()]),
                    "llm_model_provider_overrides_json": json.dumps(
                        {
                            "duplication_detection": {
                                _GEMMA_MODEL: _GEMMA_ROUTE_TARGET,
                                _KIMI_MODEL: _KIMI_ROUTE_TARGET,
                                _GLM_MODEL: _GLM_ROUTE_TARGET,
                            }
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
        allowed_providers={"bedrock", "chutes", "vertex"},
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
        ),
    )
    request = SimilarityJudgeRequest(
        batch_id=uuid4(),
        candidate_artifact_id=uuid4(),
        reference_artifact_id=uuid4(),
        candidate_miner_uid=2,
        reference_miner_uid=1,
        reference_script=reference_script,
        candidate_diff=candidate_diff,
    )

    try:
        results = [await service.judge(request) for _ in range(_CALIBRATION_SAMPLE_COUNT)]
    finally:
        await registry.aclose()

    observed_counts = Counter(result.classification for result in results)
    print(
        json.dumps(
            {
                "event": "similarity_judge.calibration",
                "model": model,
                "provider": route_target,
                "expected_classification": expected_classification,
                "observed_classifications": dict(sorted(observed_counts.items())),
                "sample_count": _CALIBRATION_SAMPLE_COUNT,
            },
            sort_keys=True,
        )
    )

    assert len(llm_provider.requests) == _CALIBRATION_SAMPLE_COUNT
    assert all(result.model == similarity_route.model for result in results)
    assert all(result.provider == similarity_route.provider for result in results)
    assert all(result.reasoning and result.reasoning.strip() for result in results)
    assert all(
        result.reasoning_tokens is None or result.reasoning_tokens >= 0
        for result in results
    )
    assert all(llm_request.output_mode == "structured" for llm_request in llm_provider.requests)
    assert all(
        llm_request.provider == settings.llm.similarity_llm_provider
        for llm_request in llm_provider.requests
    )
    assert all(llm_request.model == similarity_route.model for llm_request in llm_provider.requests)
    assert all(llm_request.reasoning_effort == "high" for llm_request in llm_provider.requests)
    assert all(
        llm_request.max_output_tokens == settings.llm.similarity_llm_max_output_tokens
        for llm_request in llm_provider.requests
    )
    assert all(
        llm_request.timeout_seconds == settings.llm.similarity_llm_timeout_seconds
        for llm_request in llm_provider.requests
    )
    assert all(
        llm_request.retry_policy == settings.llm.similarity_llm_retry_policy
        for llm_request in llm_provider.requests
    )
    assert all(llm_request.thinking is None for llm_request in llm_provider.requests)
    assert all(
        llm_request.use_case == "miner_task_similarity_judge"
        for llm_request in llm_provider.requests
    )
    assert all(response.metadata is not None for response in llm_provider.responses)
    assert all(
        response.metadata["selected_provider"] == similarity_route.provider
        for response in llm_provider.responses
    )
    assert all(
        response.metadata["selected_model"] == similarity_route.model
        for response in llm_provider.responses
    )
