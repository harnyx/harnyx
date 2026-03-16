from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from harnyx_commons.domain.miner_task import EvaluationDetails, EvaluationError, Response, ScoreBreakdown
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_validator.domain.evaluation import MinerTaskRun


def test_miner_task_run_rejects_failed_state_without_error() -> None:
    with pytest.raises(
        ValueError,
        match="evaluation details must include exactly one of score_breakdown or error",
    ):
        MinerTaskRun(
            session_id=uuid4(),
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            response=None,
            details=EvaluationDetails.model_construct(
                score_breakdown=None,
                total_tool_usage=ToolUsageSummary.zero(),
                error=None,
            ),
            completed_at=datetime.now(UTC),
        )


def test_miner_task_run_accepts_failed_state_with_error() -> None:
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=7,
        artifact_id=uuid4(),
        task_id=uuid4(),
        response=None,
        details=EvaluationDetails(
            error=EvaluationError(code="sandbox_failed", message="sandbox failed"),
        ),
        completed_at=datetime.now(UTC),
    )

    assert run.response is None


def test_miner_task_run_accepts_successful_state_with_response() -> None:
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=7,
        artifact_id=uuid4(),
        task_id=uuid4(),
        response=Response(text="ok"),
        details=EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=0.5,
                similarity_score=0.5,
                total_score=0.5,
                scoring_version="v1",
            ),
        ),
        completed_at=datetime.now(UTC),
    )

    assert run.response == Response(text="ok")
