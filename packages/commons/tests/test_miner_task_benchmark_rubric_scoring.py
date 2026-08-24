from __future__ import annotations

import json

import pytest

from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.miner_task_benchmark import (
    BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    BenchmarkWeightedRubricScoringConfig,
    BenchmarkWeightedRubricScoringService,
    WeightedRubric,
    WeightedRubricCriterion,
    WeightedRubricCriterionDecision,
    parse_weighted_rubric,
    score_weighted_rubric_decisions,
)


def test_parse_weighted_rubric_accepts_minimized_verified_draco_answer_shape() -> None:
    draco_answer_json = json.dumps(
        {
            "id": "staggered-did-methodology-evaluation",
            "sections": [
                {
                    "id": "factual-accuracy",
                    "title": "Factual Accuracy",
                    "criteria": [
                        {
                            "id": "twfe-variance-weighted-decomposition",
                            "weight": 10,
                            "requirement": (
                                "States TWFE coefficient is a variance-weighted average of treatment effects."
                            ),
                        },
                        {
                            "id": "negative-example",
                            "weight": -5,
                            "requirement": "Penalizes a specific error when MET.",
                        },
                    ],
                },
            ],
        }
    )

    rubric = parse_weighted_rubric(draco_answer_json)

    assert rubric.rubric_id == "staggered-did-methodology-evaluation"
    assert rubric.positive_weight_total == pytest.approx(10.0)
    assert rubric.criteria == (
        WeightedRubricCriterion(
            section_id="factual-accuracy",
            criterion_id="twfe-variance-weighted-decomposition",
            weight=10.0,
            requirement=("States TWFE coefficient is a variance-weighted average of treatment effects."),
        ),
        WeightedRubricCriterion(
            section_id="factual-accuracy",
            criterion_id="negative-example",
            weight=-5.0,
            requirement="Penalizes a specific error when MET.",
        ),
    )


def test_parse_weighted_rubric_rejects_missing_positive_weight_denominator() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_weighted_rubric(
            json.dumps(
                {
                    "id": "only-negative",
                    "sections": [
                        {
                            "id": "factual-accuracy",
                            "title": "Factual Accuracy",
                            "criteria": [
                                {
                                    "id": "negative-example",
                                    "weight": -5,
                                    "requirement": "Penalizes a specific error when MET.",
                                }
                            ],
                        }
                    ],
                }
            )
        )


def test_parse_weighted_rubric_rejects_duplicate_criterion_ids() -> None:
    with pytest.raises(ValueError, match="duplicate criterion"):
        parse_weighted_rubric(
            json.dumps(
                {
                    "id": "duplicate",
                    "sections": [
                        {
                            "id": "factual-accuracy",
                            "title": "Factual Accuracy",
                            "criteria": [
                                {
                                    "id": "same",
                                    "weight": 10,
                                    "requirement": "Positive criterion.",
                                },
                                {
                                    "id": "same",
                                    "weight": 5,
                                    "requirement": "Duplicate criterion.",
                                },
                            ],
                        }
                    ],
                }
            )
        )


def test_parse_weighted_rubric_rejects_zero_weight_criteria() -> None:
    with pytest.raises(ValueError, match="zero"):
        parse_weighted_rubric(
            json.dumps(
                {
                    "id": "zero-weight",
                    "sections": [
                        {
                            "id": "factual-accuracy",
                            "title": "Factual Accuracy",
                            "criteria": [
                                {
                                    "id": "zero",
                                    "weight": 0,
                                    "requirement": "Zero weight is not a real criterion.",
                                }
                            ],
                        }
                    ],
                }
            )
        )


def test_score_weighted_rubric_decisions_applies_signed_weights_and_clamps() -> None:
    rubric = _weighted_rubric_with_positive_and_negative_criteria()

    score = score_weighted_rubric_decisions(
        rubric=rubric,
        decisions=(
            WeightedRubricCriterionDecision(
                criterion_id="positive",
                met=True,
                justification="Positive requirement is satisfied.",
            ),
            WeightedRubricCriterionDecision(
                criterion_id="negative",
                met=True,
                justification="Negative requirement is also present.",
            ),
        ),
    )

    assert score.raw_score == pytest.approx(5.0)
    assert score.normalized_score == pytest.approx(0.5)
    assert score.criteria[0].contribution == pytest.approx(10.0)
    assert score.criteria[1].contribution == pytest.approx(-5.0)


def test_score_weighted_rubric_decisions_clamps_negative_raw_score_to_zero() -> None:
    rubric = _weighted_rubric_with_positive_and_negative_criteria()

    score = score_weighted_rubric_decisions(
        rubric=rubric,
        decisions=(
            WeightedRubricCriterionDecision(
                criterion_id="positive",
                met=False,
                justification="Positive requirement is missing.",
            ),
            WeightedRubricCriterionDecision(
                criterion_id="negative",
                met=True,
                justification="Negative requirement is present.",
            ),
        ),
    )

    assert score.raw_score == pytest.approx(-5.0)
    assert score.normalized_score == 0.0


def test_score_weighted_rubric_decisions_caps_score_at_one_and_ignores_unmet_negative() -> None:
    rubric = _weighted_rubric_with_positive_and_negative_criteria()

    score = score_weighted_rubric_decisions(
        rubric=rubric,
        decisions=(
            WeightedRubricCriterionDecision(
                criterion_id="positive",
                met=True,
                justification="Positive requirement is satisfied.",
            ),
            WeightedRubricCriterionDecision(
                criterion_id="negative",
                met=False,
                justification="Negative requirement is absent.",
            ),
        ),
    )

    assert score.raw_score == pytest.approx(rubric.positive_weight_total)
    assert score.normalized_score == 1.0
    assert score.criteria[1].contribution == 0.0


def test_score_weighted_rubric_decisions_rejects_missing_decision() -> None:
    rubric = _weighted_rubric_with_positive_and_negative_criteria()

    with pytest.raises(ValueError, match="missing"):
        score_weighted_rubric_decisions(
            rubric=rubric,
            decisions=(
                WeightedRubricCriterionDecision(
                    criterion_id="positive",
                    met=True,
                    justification="Positive requirement is satisfied.",
                ),
            ),
        )


@pytest.mark.anyio("asyncio")
async def test_weighted_rubric_judge_calls_one_ungrounded_structured_request_per_criterion() -> None:
    provider = _RecordingRubricJudgeProvider(
        verdicts=[
            {"verdict": "MET", "justification": "Positive requirement is satisfied."},
            {"verdict": "UNMET", "justification": "Negative issue is absent."},
        ]
    )
    service = BenchmarkWeightedRubricScoringService(
        llm_provider=provider,
        config=BenchmarkWeightedRubricScoringConfig(provider="vertex", model="rubric-model"),
    )

    score = await service.score(
        question="Which answer should be blue?",
        rubric_answer=_rubric_answer_json(),
        generated_answer="The answer is blue.",
    )

    assert len(provider.requests) == 2
    assert all(request.grounded is False for request in provider.requests)
    assert all(not request.tools for request in provider.requests)
    assert all(request.output_mode == "structured" for request in provider.requests)
    assert all(request.use_case == "benchmark_weighted_rubric_criterion_judge" for request in provider.requests)
    assert all(request.max_output_tokens is None for request in provider.requests)
    assert score.normalized_score == pytest.approx(1.0)


@pytest.mark.anyio("asyncio")
async def test_weighted_rubric_judge_returns_score_detail_json_object() -> None:
    provider = _RecordingRubricJudgeProvider(
        verdicts=[
            {"verdict": "MET", "justification": "Positive requirement is satisfied."},
            {"verdict": "MET", "justification": "Negative issue is present."},
        ]
    )
    service = BenchmarkWeightedRubricScoringService(
        llm_provider=provider,
        config=BenchmarkWeightedRubricScoringConfig(provider="vertex", model="rubric-model"),
    )

    score = await service.score(
        question="Which answer should be blue?",
        rubric_answer=_rubric_answer_json(),
        generated_answer="The answer is blue with an issue.",
    )

    assert score.raw_score == pytest.approx(5.0)
    assert score.normalized_score == pytest.approx(0.5)
    assert score.score_detail["scoring_version"] == BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION
    assert score.score_detail["rubric_id"] == "signed-weight-example"
    assert score.score_detail["raw_score"] == pytest.approx(5.0)
    assert score.score_detail["normalized_score"] == pytest.approx(0.5)
    criteria = score.score_detail["criteria"]
    assert isinstance(criteria, list)
    assert criteria[1]["verdict"] == "MET"
    assert criteria[1]["weight"] == pytest.approx(-5.0)
    assert criteria[1]["contribution"] == pytest.approx(-5.0)


@pytest.mark.anyio("asyncio")
async def test_weighted_rubric_judge_fails_when_structured_output_is_missing() -> None:
    provider = _RecordingRubricJudgeProvider(verdicts=[None])
    service = BenchmarkWeightedRubricScoringService(
        llm_provider=provider,
        config=BenchmarkWeightedRubricScoringConfig(provider="vertex", model="rubric-model"),
    )

    with pytest.raises(RuntimeError, match="weighted rubric judge did not return structured output"):
        await service.score(
            question="Which answer should be blue?",
            rubric_answer=_rubric_answer_json(),
            generated_answer="The answer is blue.",
        )


def _weighted_rubric_with_positive_and_negative_criteria() -> WeightedRubric:
    return parse_weighted_rubric(_rubric_answer_json())


def _rubric_answer_json() -> str:
    return json.dumps(
        {
            "id": "signed-weight-example",
            "sections": [
                {
                    "id": "factual-accuracy",
                    "title": "Factual Accuracy",
                    "criteria": [
                        {
                            "id": "positive",
                            "weight": 10,
                            "requirement": "Reward this when MET.",
                        },
                        {
                            "id": "negative",
                            "weight": -5,
                            "requirement": "Penalize this when MET.",
                        },
                    ],
                }
            ],
        }
    )


class _RecordingRubricJudgeProvider:
    def __init__(self, *, verdicts: list[dict[str, str] | None]) -> None:
        self._verdicts = verdicts
        self.requests: list[LlmRequest] = []

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if not self._verdicts:
            raise RuntimeError("missing rubric verdict")
        postprocessed = self._verdicts.pop(0)
        return LlmResponse(
            id=f"rubric-response-{len(self.requests)}",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(),
                    ),
                ),
            ),
            usage=LlmUsage(),
            postprocessed=postprocessed,
        )

    async def aclose(self) -> None:
        return None


def _rubric_judge_payload(request: LlmRequest) -> dict[str, object]:
    user_prompt = request.messages[1].content[0].text
    _, payload_json = user_prompt.split("Payload:\n", 1)
    return json.loads(payload_json)
