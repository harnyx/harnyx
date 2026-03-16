from __future__ import annotations

from dataclasses import replace
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
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.state.run_progress import InMemoryRunProgress


def _make_batch(*, batch_id: UUID | None = None, query_text: str = "example") -> MinerTaskBatchSpec:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text=query_text),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="abc", size_bytes=1)
    return MinerTaskBatchSpec(
        batch_id=batch_id or uuid4(),
        cutoff_at="2025-01-01T00:00:00Z",
        created_at="2025-01-01T00:00:00Z",
        tasks=(task,),
        artifacts=(artifact,),
    )


def _make_multi_batch(*, batch_id: UUID | None = None) -> MinerTaskBatchSpec:
    tasks = (
        MinerTask(
            task_id=uuid4(),
            query=Query(text="example one"),
            reference_answer=ReferenceAnswer(text="reference one"),
        ),
        MinerTask(
            task_id=uuid4(),
            query=Query(text="example two"),
            reference_answer=ReferenceAnswer(text="reference two"),
        ),
    )
    artifacts = (
        ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="abc", size_bytes=1),
        ScriptArtifactSpec(uid=8, artifact_id=uuid4(), content_hash="def", size_bytes=2),
    )
    return MinerTaskBatchSpec(
        batch_id=batch_id or uuid4(),
        cutoff_at="2025-01-01T00:00:00Z",
        created_at="2025-01-01T00:00:00Z",
        tasks=tasks,
        artifacts=artifacts,
    )


def _make_submission(batch: MinerTaskBatchSpec, *, score: float = 1.0) -> MinerTaskRunSubmission:
    task = batch.tasks[0]
    artifact = batch.artifacts[0]
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=artifact.uid,
        artifact_id=artifact.artifact_id,
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
        budget_usd=task.budget_usd,
        usage=SessionUsage(total_cost_usd=0.0),
    )
    return MinerTaskRunSubmission(
        batch_id=batch.batch_id,
        validator_uid=4,
        run=run,
        score=score,
        usage=TokenUsageSummary.empty(),
        session=session,
    )


def test_run_progress_recorded_pairs_returns_exact_finished_pairs() -> None:
    progress = InMemoryRunProgress()
    batch = _make_multi_batch()
    first_submission = _make_submission(batch)
    second_run = first_submission.run.model_copy(
        update={
            "artifact_id": batch.artifacts[1].artifact_id,
            "task_id": batch.tasks[1].task_id,
        }
    )
    second_submission = first_submission.model_copy(
        update={
            "run": second_run,
            "session": replace(
                first_submission.session,
                session_id=second_run.session_id,
                task_id=second_run.task_id,
            ),
        }
    )

    progress.register(batch)
    progress.record(first_submission)
    progress.record(second_submission)

    assert progress.recorded_pairs(batch.batch_id) == frozenset(
        {
            (batch.artifacts[0].artifact_id, batch.tasks[0].task_id),
            (batch.artifacts[1].artifact_id, batch.tasks[1].task_id),
        }
    )


def test_run_progress_register_is_idempotent_for_exact_replay() -> None:
    progress = InMemoryRunProgress()
    batch = _make_batch()

    progress.register(batch)
    progress.register(batch)

    snapshot = progress.snapshot(batch.batch_id)
    assert snapshot["total"] == 1
    assert snapshot["completed"] == 0
    assert snapshot["remaining"] == 1
    assert snapshot["tasks"] == batch.tasks


def test_run_progress_register_rejects_conflicting_replay() -> None:
    progress = InMemoryRunProgress()
    batch_id = uuid4()
    batch = _make_batch(batch_id=batch_id, query_text="original")
    conflicting = _make_batch(batch_id=batch_id, query_text="different")

    progress.register(batch)

    with pytest.raises(RuntimeError, match="batch_id already exists with different contents"):
        progress.register(conflicting)


def test_run_progress_record_is_idempotent_for_duplicate_pair() -> None:
    progress = InMemoryRunProgress()
    batch = _make_batch()
    submission = _make_submission(batch)

    progress.register(batch)
    progress.record(submission)
    progress.record(submission)

    snapshot = progress.snapshot(batch.batch_id)
    assert snapshot["total"] == 1
    assert snapshot["completed"] == 1
    assert snapshot["remaining"] == 0
    assert snapshot["miner_task_runs"] == (submission,)


def test_run_progress_record_rejects_conflicting_duplicate_pair() -> None:
    progress = InMemoryRunProgress()
    batch = _make_batch()
    submission = _make_submission(batch, score=1.0)
    conflicting = _make_submission(batch, score=0.0)
    conflicting = conflicting.model_copy(
        update={
            "run": conflicting.run.model_copy(
                update={
                    "artifact_id": submission.run.artifact_id,
                    "task_id": submission.run.task_id,
                }
            )
        }
    )

    progress.register(batch)
    progress.record(submission)

    with pytest.raises(
        RuntimeError,
        match="batch already recorded a different result for artifact/task pair",
    ):
        progress.record(conflicting)
