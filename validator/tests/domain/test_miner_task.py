from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from caster_commons.domain.miner_task import EvaluationDetails, EvaluationError, Response, ScoreBreakdown
from caster_validator.domain.evaluation import MinerTaskRun

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
                    comparison_score=1.0,
                    similarity_score=0.5,
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
                comparison_score=1.0,
                similarity_score=0.5,
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
                comparison_score=1.0,
                similarity_score=0.5,
                total_score=0.75,
                scoring_version="v1",
            ),
            elapsed_ms=-1.0,
        )
