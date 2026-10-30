"""Protect source identity and prevent missing/malformed judgments becoming scores."""

from collections import Counter
from dataclasses import replace
from typing import cast

import pytest

from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.miner_task_benchmark.aa_omniscience.loader import (
    load_aa_omniscience_grader_prompt,
    load_aa_omniscience_snapshot,
)
from harnyx_commons.miner_task_benchmark.aa_omniscience.scoring import (
    OmniscienceGrade,
    OmniscienceScoringService,
)


def test_public_v1_source_identity_and_answer_alignment() -> None:
    snapshot = load_aa_omniscience_snapshot()
    assert len(snapshot.items) == snapshot.manifest.row_count == 600
    assert tuple(item.item_index for item in snapshot.items) == tuple(range(600))
    assert tuple(item.source_item_id for item in snapshot.items) == tuple(map(str, range(1, 601)))
    assert sorted(Counter(item.problem_category for item in snapshot.items).values()) == [100] * 6
    assert all(item.problem and item.answer for item in snapshot.items)
    assert snapshot.manifest.scoring_version == "aa-omniscience-accuracy-v1"


def _response(text: str, finish_reason: str = "stop") -> LlmResponse:
    return LlmResponse(
        id="test-judge",
        choices=(LlmChoice(index=0, message=LlmChoiceMessage(
            role="assistant", content=(LlmMessageContentPart(type="output_text", text=text),)
        )),),
        usage=LlmUsage(),
        finish_reason=finish_reason,
    )


class _Provider:
    def __init__(self, response: LlmResponse) -> None:
        self.response = response
        self.requests: list[LlmRequest] = []

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return self.response


@pytest.mark.anyio
@pytest.mark.parametrize("grade", list(OmniscienceGrade))
async def test_four_classes_keep_partial_and_abstention_separate_from_correct(grade: OmniscienceGrade) -> None:
    provider = _Provider(_response(f" {grade.value}\n"))
    decision = await OmniscienceScoringService(cast(LlmProviderPort, provider)).score(
        question="Question {literal}", reference_answer="Target", generated_answer="Prediction"
    )
    assert decision.grade is grade
    assert decision.is_correct is (grade is OmniscienceGrade.CORRECT)
    (request,) = provider.requests
    assert request.messages[0].content[0].text == load_aa_omniscience_grader_prompt().format(
        question="Question {literal}", target="Target", predicted_answer="Prediction"
    )
    assert request.extra == {"reasoning": {"enabled": True}, "provider": {"require_parameters": True}}
    assert request.max_output_tokens is None


@pytest.mark.anyio
@pytest.mark.parametrize("response", [
    _response("CORRECT"), _response("A: CORRECT"), _response(""),
    _response("A", "length"), replace(_response("A"), choices=()),
])
async def test_malformed_or_incomplete_judgment_never_becomes_a_score(response: LlmResponse) -> None:
    provider = _Provider(response)
    with pytest.raises(ValueError, match="AA judge"):
        await OmniscienceScoringService(cast(LlmProviderPort, provider)).score(
            question="Q", reference_answer="T", generated_answer="P"
        )
    assert len(provider.requests) == 1
