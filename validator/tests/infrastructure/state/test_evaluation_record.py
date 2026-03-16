from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import Session, SessionUsage
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission, TokenUsageSummary
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.state.evaluation_record import InMemoryEvaluationRecordStore


def _make_submission(
    *,
    batch_id: UUID | None = None,
    artifact_id: UUID | None = None,
    task_id: UUID | None = None,
    score: float = 1.0,
) -> MinerTaskRunSubmission:
    task = MinerTask(
        task_id=task_id or uuid4(),
        query=Query(text="Claim text"),
        reference_answer=ReferenceAnswer(text="Reference text"),
    )
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=7,
        artifact_id=artifact_id or uuid4(),
        task_id=task.task_id,
        response=Response(text="ok"),
        details=EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=score,
                similarity_score=score,
                total_score=score,
                scoring_version="v1",
            ),
            total_tool_usage=ToolUsageSummary.zero(),
        ),
        completed_at=datetime.now(UTC),
    )

    issued_at = datetime.now(UTC)
    session = Session(
        session_id=run.session_id,
        uid=run.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=0.1,
        usage=SessionUsage(total_cost_usd=0.0),
    )

    return MinerTaskRunSubmission(
        batch_id=batch_id or uuid4(),
        validator_uid=4,
        run=run,
        score=score,
        usage=TokenUsageSummary.empty(),
        session=session,
    )


def test_in_memory_store_records_miner_task_run_submissions() -> None:
    store = InMemoryEvaluationRecordStore()
    submission = _make_submission()

    store.record(submission)

    records = store.records()
    assert records == (submission,)


def test_in_memory_store_duplicate_identical_pair_is_a_noop() -> None:
    store = InMemoryEvaluationRecordStore()
    submission = _make_submission()

    store.record(submission)
    store.record(submission)

    assert store.records() == (submission,)


def test_in_memory_store_rejects_conflicting_duplicate_pair() -> None:
    store = InMemoryEvaluationRecordStore()
    batch_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    submission = _make_submission(batch_id=batch_id, artifact_id=artifact_id, task_id=task_id, score=1.0)
    conflicting = _make_submission(batch_id=batch_id, artifact_id=artifact_id, task_id=task_id, score=0.0)

    store.record(submission)

    with pytest.raises(
        RuntimeError,
        match="batch already recorded a different result for artifact/task pair",
    ):
        store.record(conflicting)
