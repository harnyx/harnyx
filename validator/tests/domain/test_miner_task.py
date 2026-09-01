from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    FastScoreEvidence,
    FastScoreExpectedComponent,
    Response,
    ScoreBreakdown,
)
from harnyx_validator.domain.evaluation import MinerTaskRun

_NOW = datetime.now(UTC)


def test_successful_run_requires_response() -> None:
    with pytest.raises(ValidationError, match="response"):
        MinerTaskRun(
            session_id=uuid4(),
            uid=1,
            artifact_id=uuid4(),
            task_id=uuid4(),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=0.75,
                    total_score=0.75,
                    scoring_version="v1",
                )
            ),
            completed_at=_NOW,
        )


def test_failed_run_rejects_response() -> None:
    with pytest.raises(ValidationError, match="must not include a response"):
        MinerTaskRun(
            session_id=uuid4(),
            uid=1,
            artifact_id=uuid4(),
            task_id=uuid4(),
            response=Response(text="This should not be present."),
            details=EvaluationDetails(
                error=EvaluationError(code="sandbox_failed", message="sandbox failed"),
            ),
            completed_at=_NOW,
        )


def test_successful_run_accepts_elapsed_ms() -> None:
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=1,
        artifact_id=uuid4(),
        task_id=uuid4(),
        response=Response(text="The miner answer."),
        details=EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=0.75,
                total_score=0.75,
                scoring_version="v1",
            ),
            elapsed_ms=1500.0,
        ),
        completed_at=_NOW,
    )

    assert run.details.elapsed_ms == pytest.approx(1500.0)


def test_evaluation_details_reject_negative_elapsed_ms() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=0.75,
                total_score=0.75,
                scoring_version="v1",
            ),
            elapsed_ms=-1.0,
        )


def test_score_breakdown_rejects_total_that_differs_from_comparison() -> None:
    with pytest.raises(ValidationError, match="comparison_score"):
        ScoreBreakdown(
            comparison_score=0.8,
            total_score=0.9,
            scoring_version="v1",
        )


def test_fast_score_evidence_rejects_inconsistent_metrics() -> None:
    with pytest.raises(ValidationError, match="precision"):
        FastScoreEvidence(
            expected_components=(FastScoreExpectedComponent(component_id="answer", is_correct=True),),
            excessive_components=(),
            precision=0.5,
            recall=1.0,
        )


def test_score_breakdown_rejects_score_that_differs_from_fast_f1() -> None:
    evidence = FastScoreEvidence(
        expected_components=(FastScoreExpectedComponent(component_id="answer", is_correct=True),),
        excessive_components=(),
        precision=1.0,
        recall=1.0,
    )

    with pytest.raises(ValidationError, match="fast score evidence"):
        ScoreBreakdown(
            comparison_score=0.5,
            total_score=0.5,
            scoring_version="deepsearchqa-f1-v1",
            fast_score_evidence=evidence,
        )


def test_score_breakdown_omits_absent_fast_evidence_from_legacy_payloads() -> None:
    breakdown = ScoreBreakdown(
        comparison_score=0.8,
        total_score=0.8,
        scoring_version="v1",
    )

    assert breakdown.fast_score_evidence is None
    assert "fast_score_evidence" not in breakdown.model_dump(mode="json")


def test_evaluation_details_normalizes_legacy_similarity_score_payload() -> None:
    details = EvaluationDetails.model_validate(
        {
            "score_breakdown": {
                "comparison_score": 0.8,
                "similarity_score": 0.9,
                "total_score": 0.9,
                "scoring_version": "v1",
            },
            "total_tool_usage": {},
        }
    )

    assert details.score_breakdown is not None
    assert details.score_breakdown.comparison_score == pytest.approx(0.9)
    assert details.score_breakdown.total_score == pytest.approx(0.9)
