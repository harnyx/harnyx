from __future__ import annotations

import json
from uuid import uuid4

import pytest

from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_resolver
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest
from harnyx_validator.application.similarity_judge import SimilarityJudge, SimilarityJudgeConfig
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


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
                    "llm_model_provider_overrides_json": json.dumps(
                        {"duplication_detection": {bootstrap._DUPLICATION_DETECTION_LLM_MODEL: "bedrock"}}
                    )
                }
            )
        }
    )
    similarity_route = bootstrap._resolve_similarity_judge_route(settings)

    resolve_provider = build_cached_llm_provider_resolver(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    llm_provider = RecordingProvider(resolve_provider(similarity_route.provider))
    service = SimilarityJudge(
        llm_provider=llm_provider,
        config=SimilarityJudgeConfig(
            provider=similarity_route.provider,
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
        await llm_provider.aclose()

    assert result.verdict in {"not_duplicate", "duplicate"}
    assert result.model == similarity_route.model
    assert result.provider == similarity_route.provider
    assert len(llm_provider.requests) == 1
    llm_request = llm_provider.requests[0]
    assert llm_request.output_mode == "structured"
    assert llm_request.provider == similarity_route.provider
    assert llm_request.model == similarity_route.model
    assert llm_request.use_case == "miner_task_similarity_judge"
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
