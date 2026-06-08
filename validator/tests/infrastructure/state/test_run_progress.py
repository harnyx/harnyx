from __future__ import annotations

import os
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
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
from harnyx_validator.infrastructure.state.run_progress import (
    FileBackedRunProgress,
    ProgressCursorBeforeRestoreFloorError,
)


def _progress(tmp_path: Path) -> FileBackedRunProgress:
    return FileBackedRunProgress(storage_root=tmp_path / "run-progress")


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


def _make_failed_submission(
    batch: MinerTaskBatchSpec,
    *,
    error_code: str,
) -> MinerTaskRunSubmission:
    task = batch.tasks[0]
    artifact = batch.artifacts[0]
    issued_at = datetime.now(UTC)
    session_id = uuid4()
    return MinerTaskRunSubmission(
        batch_id=batch.batch_id,
        validator_uid=4,
        run=MinerTaskRun(
            session_id=session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=None,
            details=EvaluationDetails(
                error=EvaluationError(code=error_code, message="terminal timeout"),
                total_tool_usage=ToolUsageSummary.zero(),
                elapsed_ms=5000.0,
            ),
            completed_at=issued_at + timedelta(seconds=5),
        ),
        score=0.0,
        usage=TokenUsageSummary.empty(),
        session=Session(
            session_id=session_id,
            uid=artifact.uid,
            task_id=task.task_id,
            issued_at=issued_at,
            expires_at=issued_at + timedelta(minutes=5),
            budget_usd=task.budget_usd,
            usage=SessionUsage(total_cost_usd=0.0),
        ),
    )


def _set_tree_mtime(path: Path, when: datetime) -> None:
    timestamp = when.timestamp()
    os.utime(path, (timestamp, timestamp))
    if path.is_dir():
        for child in path.rglob("*"):
            os.utime(child, (timestamp, timestamp))


def _make_distinct_multi_submissions(
    batch: MinerTaskBatchSpec,
) -> tuple[MinerTaskRunSubmission, ...]:
    template = _make_submission(batch)

    def build_submission(*, artifact_index: int, task_index: int, score: float) -> MinerTaskRunSubmission:
        details = template.run.details.model_copy(
            update={
                "score_breakdown": ScoreBreakdown(
                    comparison_score=score,
                    total_score=score,
                    scoring_version="v1",
                )
            }
        )
        run = template.run.model_copy(
            update={
                "session_id": uuid4(),
                "uid": batch.artifacts[artifact_index].uid,
                "artifact_id": batch.artifacts[artifact_index].artifact_id,
                "task_id": batch.tasks[task_index].task_id,
                "details": details,
            }
        )
        session = replace(
            template.session,
            session_id=run.session_id,
            uid=run.uid,
            task_id=run.task_id,
        )
        return template.model_copy(update={"run": run, "session": session, "score": score})

    return (
        build_submission(artifact_index=1, task_index=1, score=0.4),
        build_submission(artifact_index=0, task_index=1, score=0.3),
        build_submission(artifact_index=1, task_index=0, score=0.2),
        build_submission(artifact_index=0, task_index=0, score=0.1),
    )


def test_run_progress_recorded_pairs_returns_exact_finished_pairs(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
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


def test_run_progress_register_is_idempotent_for_exact_replay(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()

    progress.register(batch)
    progress.register(batch)

    summary = progress.summary(batch.batch_id)
    assert summary["total"] == 1
    assert summary["completed"] == 0
    assert summary["remaining"] == 1
    assert summary["latest_sequence"] == 0


def test_run_progress_register_rejects_conflicting_replay(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch_id = uuid4()
    batch = _make_batch(batch_id=batch_id, query_text="original")
    conflicting = _make_batch(batch_id=batch_id, query_text="different")

    progress.register(batch)

    with pytest.raises(RuntimeError, match="batch_id already exists with different contents"):
        progress.register(conflicting)


def test_run_progress_record_is_idempotent_for_duplicate_pair(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    submission = _make_submission(batch)

    progress.register(batch)
    progress.record(submission)
    progress.record(submission)

    summary = progress.summary(batch.batch_id)
    page = progress.completed_run_page(batch.batch_id, after_sequence=0, limit=10)
    assert summary["total"] == 1
    assert summary["completed"] == 1
    assert summary["remaining"] == 0
    assert summary["latest_sequence"] == 1
    assert page["items"] == ({"sequence": 1, "kind": "completed_run", "submission": submission, "attempt": None},)


def test_run_progress_restore_completed_runs_records_terminal_timeout_failed_submission(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    failed_submission = _make_failed_submission(batch, error_code="timeout_miner_owned")

    progress.restore_completed_runs(batch, submissions=(failed_submission,))

    summary = progress.summary(batch.batch_id)
    page = progress.completed_run_page(batch.batch_id, after_sequence=0, limit=10)
    assert summary["total"] == 1
    assert summary["completed"] == 1
    assert summary["remaining"] == 0
    assert summary["latest_sequence"] == 1
    assert page["items"] == (
        {"sequence": 1, "kind": "completed_run", "submission": failed_submission, "attempt": None},
    )
    assert progress.recorded_pairs(batch.batch_id) == {
        (failed_submission.run.artifact_id, failed_submission.run.task_id),
    }


def test_run_progress_restore_floor_continues_without_replaying_restored_runs(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    submission = _make_submission(batch)

    progress.register(batch)
    progress.restore_progress_floor(batch.batch_id, 7)
    progress.restore_completed_runs(batch, submissions=(submission,))

    summary = progress.summary(batch.batch_id)
    page = progress.completed_run_page(batch.batch_id, after_sequence=7, limit=10)
    assert summary["completed"] == 1
    assert summary["latest_sequence"] == 7
    assert page["items"] == ()
    assert progress.recorded_pairs(batch.batch_id) == {
        (submission.run.artifact_id, submission.run.task_id),
    }


def test_run_progress_first_new_detail_after_restore_floor_is_dense(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_multi_batch()
    restored, new_submission = _make_distinct_multi_submissions(batch)[:2]

    progress.register(batch)
    progress.restore_progress_floor(batch.batch_id, 7)
    progress.restore_completed_runs(batch, submissions=(restored,))
    progress.record(new_submission)

    page = progress.completed_run_page(batch.batch_id, after_sequence=7, limit=10)
    assert [item["sequence"] for item in page["items"]] == [8]


def test_run_progress_page_below_restore_floor_fails_explicitly(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()

    progress.register(batch)
    progress.restore_progress_floor(batch.batch_id, 7)

    with pytest.raises(ProgressCursorBeforeRestoreFloorError, match="restored platform detail floor"):
        progress.completed_run_page(batch.batch_id, after_sequence=0, limit=10)


def test_run_progress_record_rejects_conflicting_duplicate_pair(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
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


def test_run_progress_restore_provider_evidence_is_monotonic_for_duplicate_replay(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    session_id = uuid4()

    progress.register(batch)
    progress.register_task_session(
        batch_id=batch.batch_id,
        session_id=session_id,
    )
    progress.record_provider_call(session_id=session_id, provider="openai", model="gpt-4o")
    progress.record_provider_call(session_id=session_id, provider="openai", model="gpt-4o")
    progress.record_provider_failure(
        session_id=session_id,
        provider="openai",
        model="gpt-4o",
        reason="rate limited",
    )
    progress.record_provider_call(session_id=session_id, provider="desearch", model="search_web")
    progress.record_provider_call(session_id=session_id, provider="desearch", model="search_web")
    progress.record_provider_call(session_id=session_id, provider="desearch", model="search_web")

    progress.restore_completed_runs(
        batch,
        submissions=(),
        provider_evidence=(
            {
                "provider": "openai",
                "model": "gpt-4o",
                "total_calls": 1,
                "failed_calls": 0,
            },
            {
                "provider": "anthropic",
                "model": "claude-sonnet",
                "total_calls": 4,
                "failed_calls": 1,
            },
        ),
    )

    assert progress.provider_evidence(batch.batch_id) == (
        {
            "provider": "anthropic",
            "model": "claude-sonnet",
            "total_calls": 4,
            "failed_calls": 1,
        },
        {
            "provider": "desearch",
            "model": "search_web",
            "total_calls": 3,
            "failed_calls": 0,
        },
        {
            "provider": "openai",
            "model": "gpt-4o",
            "total_calls": 2,
            "failed_calls": 1,
            "failure_reason": "rate limited",
        },
    )


def test_run_progress_includes_failure_reason_in_consumed_provider_failures(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    session_id = uuid4()

    progress.register(batch)
    progress.register_task_session(batch_id=batch.batch_id, session_id=session_id)
    progress.record_provider_call(session_id=session_id, provider="desearch", model="search_web")
    progress.record_provider_failure(
        session_id=session_id,
        provider="desearch",
        model="search_web",
        reason="http_402: subscription usage cap exceeded",
    )

    expected = (
        {
            "provider": "desearch",
            "model": "search_web",
            "total_calls": 1,
            "failed_calls": 1,
            "failure_reason": "http_402: subscription usage cap exceeded",
        },
    )
    assert progress.consume_provider_failures(session_id) == expected
    assert progress.provider_evidence(batch.batch_id) == expected


def test_run_progress_rejected_restore_does_not_mutate_provider_evidence(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    other_batch = _make_batch()
    session_id = uuid4()

    progress.register(batch)
    progress.register_task_session(
        batch_id=batch.batch_id,
        session_id=session_id,
    )
    progress.record_provider_call(session_id=session_id, provider="openai", model="gpt-4o")
    progress.record_provider_failure(
        session_id=session_id,
        provider="openai",
        model="gpt-4o",
        reason="rate limited",
    )
    before = progress.provider_evidence(batch.batch_id)

    with pytest.raises(RuntimeError, match="restored submission batch_id mismatch"):
        progress.restore_completed_runs(
            batch,
            submissions=(_make_submission(other_batch),),
            provider_evidence=(
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet",
                    "total_calls": 4,
                    "failed_calls": 1,
                },
            ),
        )

    assert progress.provider_evidence(batch.batch_id) == before


def test_run_progress_rejected_restore_does_not_advance_sequence_cursor(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    submission = _make_submission(batch, score=1.0)
    conflicting = _make_submission(batch, score=0.0).model_copy(
        update={
            "run": submission.run.model_copy(),
            "session": replace(
                submission.session,
                session_id=submission.run.session_id,
                task_id=submission.run.task_id,
            ),
        }
    )

    with pytest.raises(
        RuntimeError,
        match="batch already recorded a different result for artifact/task pair",
    ):
        progress.restore_completed_runs(batch, submissions=(submission, conflicting))

    progress.restore_completed_runs(batch, submissions=(submission,))

    summary = progress.summary(batch.batch_id)
    page = progress.completed_run_page(batch.batch_id, after_sequence=0, limit=10)
    assert summary["completed"] == 1
    assert summary["latest_sequence"] == 1
    assert page["items"] == ({"sequence": 1, "kind": "completed_run", "submission": submission, "attempt": None},)


def test_run_progress_page_preserves_record_sequence_without_global_task_order(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_multi_batch()
    submissions = _make_distinct_multi_submissions(batch)

    progress.register(batch)
    for submission in submissions:
        progress.record(submission)

    page = progress.completed_run_page(batch.batch_id, after_sequence=0, limit=10)

    assert tuple(
        (item["sequence"], item["submission"].run.artifact_id, item["submission"].run.task_id)
        for item in page["items"]
    ) == (
        (1, batch.artifacts[1].artifact_id, batch.tasks[1].task_id),
        (2, batch.artifacts[0].artifact_id, batch.tasks[1].task_id),
        (3, batch.artifacts[1].artifact_id, batch.tasks[0].task_id),
        (4, batch.artifacts[0].artifact_id, batch.tasks[0].task_id),
    )


def test_run_progress_page_returns_cursor_window_without_rescanning_prefix(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_multi_batch()
    submissions = _make_distinct_multi_submissions(batch)

    progress.register(batch)
    for submission in submissions:
        progress.record(submission)

    page = progress.completed_run_page(batch.batch_id, after_sequence=2, limit=1)

    assert page["latest_sequence"] == 4
    assert page["next_after_sequence"] == 3
    assert page["has_more"] is True
    assert page["items"] == (
        {"sequence": 3, "kind": "completed_run", "submission": submissions[2], "attempt": None},
    )


def test_run_progress_page_rejects_missing_dense_sequence(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_multi_batch()
    submissions = _make_distinct_multi_submissions(batch)

    progress.register(batch)
    for submission in submissions:
        progress.record(submission)

    del progress.detail_by_sequence_by_batch[batch.batch_id][2]
    del progress.pair_by_sequence_by_batch[batch.batch_id][2]

    with pytest.raises(RuntimeError, match="progress sequence points at missing detail"):
        progress.completed_run_page(batch.batch_id, after_sequence=0, limit=10)


def test_run_progress_discard_batch_removes_indexes_and_blob_dir(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    submission = _make_submission(batch)
    session_id = uuid4()

    progress.register(batch)
    progress.register_task_session(batch_id=batch.batch_id, session_id=session_id)
    progress.record_provider_call(session_id=session_id, provider="desearch", model="search_web")
    progress.record_provider_failure(
        session_id=session_id,
        provider="desearch",
        model="search_web",
        reason="rate limited",
    )
    progress.record(submission)

    batch_dir = progress.storage_root / str(batch.batch_id)
    assert batch_dir.exists()
    assert progress.completed_run_page(batch.batch_id, after_sequence=0, limit=10)["items"]

    assert progress.discard_batch(batch.batch_id) is True

    assert not batch_dir.exists()
    assert progress.recorded_pairs(batch.batch_id) == frozenset()
    assert progress.provider_evidence(batch.batch_id) == ()
    assert progress.consume_provider_failures(session_id) == ()
    assert progress.summary(batch.batch_id) == {
        "batch_id": batch.batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "latest_sequence": 0,
        "provider_evidence": (),
    }


def test_run_progress_prunes_only_stale_unindexed_uuid_dirs(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=1)
    stale = now - timedelta(hours=2)
    fresh = now
    stale_batch_id = uuid4()
    fresh_batch_id = uuid4()
    non_uuid_dir = progress.storage_root / "not-a-batch"
    uuid_file = progress.storage_root / str(uuid4())
    stale_dir = progress.storage_root / str(stale_batch_id)
    fresh_dir = progress.storage_root / str(fresh_batch_id)

    batch = _make_batch()
    progress.register(batch)
    progress.record(_make_submission(batch))
    active_dir = progress.storage_root / str(batch.batch_id)

    for directory in (stale_dir, fresh_dir, non_uuid_dir):
        directory.mkdir(parents=True)
        (directory / "runs-000001.blob").write_bytes(b"data")
    uuid_file.write_text("not a directory", encoding="utf-8")

    _set_tree_mtime(stale_dir, stale)
    _set_tree_mtime(fresh_dir, fresh)
    _set_tree_mtime(non_uuid_dir, stale)
    _set_tree_mtime(active_dir, stale)
    _set_tree_mtime(uuid_file, stale)

    removed = progress.prune_stale_batch_dirs_older_than(cutoff)

    assert removed == (stale_batch_id,)
    assert not stale_dir.exists()
    assert fresh_dir.exists()
    assert active_dir.exists()
    assert non_uuid_dir.exists()
    assert uuid_file.exists()


def test_run_progress_completed_page_and_discard_are_lock_coordinated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress(tmp_path)
    batch = _make_batch()
    submission = _make_submission(batch)
    read_started = threading.Event()
    release_read = threading.Event()
    discard_finished = threading.Event()
    page_result: dict[str, object] = {}
    original_read_many = progress.blob_store.read_many

    def blocking_read_many(refs):
        read_started.set()
        assert release_read.wait(timeout=1.0)
        return original_read_many(refs)

    progress.register(batch)
    progress.record(submission)
    monkeypatch.setattr(progress.blob_store, "read_many", blocking_read_many)

    page_thread = threading.Thread(
        target=lambda: page_result.update(
            progress.completed_run_page(batch.batch_id, after_sequence=0, limit=10)
        )
    )
    discard_thread = threading.Thread(
        target=lambda: (progress.discard_batch(batch.batch_id), discard_finished.set())
    )

    page_thread.start()
    assert read_started.wait(timeout=1.0)
    discard_thread.start()
    assert not discard_finished.wait(timeout=0.05)

    release_read.set()
    page_thread.join(timeout=1.0)
    discard_thread.join(timeout=1.0)

    assert not page_thread.is_alive()
    assert not discard_thread.is_alive()
    assert page_result["items"] == (
        {"sequence": 1, "kind": "completed_run", "submission": submission, "attempt": None},
    )
    assert not (progress.storage_root / str(batch.batch_id)).exists()
