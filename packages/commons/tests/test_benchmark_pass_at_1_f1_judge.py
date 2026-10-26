"""Benchmark pass@1 grading must use the existing component F1 contract."""

from typing import cast

import pytest

from harnyx_commons.domain.miner_task import Query, ReferenceAnswer, Response
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.schema import LlmRequest, LlmResponse, LlmUsage
from harnyx_commons.miner_task_benchmark import BenchmarkCorrectnessScoringConfig, BenchmarkCorrectnessScoringService
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment, build_fast_judge_messages


@pytest.mark.anyio
async def test_benchmark_f1_preserves_shared_judgment_contract_and_partial_credit() -> None:
    requests: list[LlmRequest] = []

    class Provider:
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            requests.append(request)
            return LlmResponse(
                id="f1",
                choices=(),
                usage=LlmUsage(),
                postprocessed={
                    "expected_components": [
                        {"component_id": "A", "is_correct": True},
                        {"component_id": "B", "is_correct": False},
                    ],
                    "excessive_components": [],
                },
            )

    service = BenchmarkCorrectnessScoringService(
        cast(LlmProviderPort, Provider()), BenchmarkCorrectnessScoringConfig(provider="openrouter", model="judge")
    )
    evidence = await service.score_pass_at_1_f1(question="Q", reference_answer="A, B", generated_answer="A")
    assert evidence.computed_f1 == pytest.approx(0.666667)
    (request,) = requests
    canonical = build_fast_judge_messages(
        query=Query(text="Q", fast=True),
        reference_answer=ReferenceAnswer(text="A, B"),
        miner_response=Response(text="A"),
    )
    assert request.messages[0].content[0].text == canonical.system_prompt
    assert request.messages[1].content[0].text == canonical.user_prompt
    assert request.output_schema is FastJudgeAssessment
    assert request.max_output_tokens is None
