from __future__ import annotations

import gc
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

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
from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
from harnyx_validator.runtime import resource_usage


def _progress(tmp_path: Path) -> FileBackedRunProgress:
    return FileBackedRunProgress(storage_root=tmp_path / "run-progress")


def _large_log_batch(*, submission_count: int) -> MinerTaskBatchSpec:
    tasks = tuple(
        MinerTask(
            task_id=uuid4(),
            query=Query(text=f"query {index}"),
            reference_answer=ReferenceAnswer(text=f"reference {index}"),
        )
        for index in range(submission_count)
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="abc", size_bytes=1)
    return MinerTaskBatchSpec(
        batch_id=uuid4(),
        cutoff_at="2025-01-01T00:00:00Z",
        created_at="2025-01-01T00:00:00Z",
        tasks=tasks,
        artifacts=(artifact,),
    )


def _make_submission_with_execution_log_bytes(
    batch: MinerTaskBatchSpec,
    *,
    index: int,
    log_bytes: int,
) -> MinerTaskRunSubmission:
    task = batch.tasks[index]
    artifact = batch.artifacts[0]
    session_id = uuid4()
    completed_at = datetime.now(UTC)
    payload = f"{index:04d}-" + ("x" * max(0, log_bytes - 5))
    receipt = ToolCall(
        receipt_id=f"receipt-{index}",
        session_id=session_id,
        uid=artifact.uid,
        tool="fetch_page",
        issued_at=completed_at,
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash=f"request-{index}",
            response_hash=f"response-{index}",
            response_payload={"content": payload},
        ),
    )
    run = MinerTaskRun(
        session_id=session_id,
        uid=artifact.uid,
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        response=Response(text="ok"),
        details=EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=1.0,
                total_score=1.0,
                scoring_version="v1",
            ),
            total_tool_usage=ToolUsageSummary.zero(),
        ),
        completed_at=completed_at,
    )
    session = Session(
        session_id=session_id,
        uid=artifact.uid,
        task_id=task.task_id,
        issued_at=completed_at,
        expires_at=completed_at + timedelta(minutes=5),
        budget_usd=task.budget_usd,
        usage=SessionUsage(total_cost_usd=0.0),
    )
    return MinerTaskRunSubmission(
        batch_id=batch.batch_id,
        validator_uid=4,
        run=run,
        score=1.0,
        execution_log=(receipt,),
        usage=TokenUsageSummary.empty(),
        session=session,
    )


def _make_attempt_with_execution_log_bytes(
    batch: MinerTaskBatchSpec,
    *,
    index: int,
    log_bytes: int,
) -> MinerTaskAttemptAuditRecord:
    task = batch.tasks[index]
    artifact = batch.artifacts[0]
    session_id = uuid4()
    started_at = datetime.now(UTC)
    payload = f"{index:04d}-" + ("x" * max(0, log_bytes - 5))
    receipt = ToolCall(
        receipt_id=f"attempt-receipt-{index}",
        session_id=session_id,
        uid=artifact.uid,
        tool="fetch_page",
        issued_at=started_at,
        outcome=ToolCallOutcome.TIMEOUT,
        details=ToolCallDetails(
            request_hash=f"attempt-request-{index}",
            response_hash=f"attempt-response-{index}",
            response_payload={"content": payload},
        ),
    )
    return MinerTaskAttemptAuditRecord(
        validator_session_id=session_id,
        batch_id=batch.batch_id,
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        attempt_number=1,
        uid=artifact.uid,
        miner_hotkey_ss58="seed-miner-hotkey",
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=1),
        status=MinerTaskAttemptStatus.FAILED,
        error_code="tool_execution_timeout",
        error_summary_code="timeout_miner_owned",
        retry_decision=MinerTaskAttemptRetryDecision.WILL_RETRY,
        terminal_effect=None,
        max_attempts=2,
        execution_log=(receipt,),
    )


def _estimated_execution_log_payload_bytes(submission: MinerTaskRunSubmission) -> int:
    total = 0
    for receipt in submission.execution_log:
        response_payload = receipt.details.response_payload
        if isinstance(response_payload, dict):
            content = response_payload.get("content")
            if isinstance(content, str):
                total += len(content.encode("utf-8"))
    return total


def _estimated_attempt_execution_log_payload_bytes(attempt: MinerTaskAttemptAuditRecord) -> int:
    total = 0
    for receipt in attempt.execution_log:
        response_payload = receipt.details.response_payload
        if isinstance(response_payload, dict):
            content = response_payload.get("content")
            if isinstance(content, str):
                total += len(content.encode("utf-8"))
    return total


def _sum_blob_bytes(storage_root: Path | None) -> int:
    if storage_root is None or not storage_root.exists():
        return 0
    return sum(path.stat().st_size for path in storage_root.rglob("*.blob"))


def _measure_large_log_retention(
    progress: FileBackedRunProgress,
    *,
    batch: MinerTaskBatchSpec,
    submission_count: int,
    log_bytes_per_submission: int,
) -> dict[str, int]:
    gc.collect()
    tracemalloc.start()
    before_heap, _before_peak = tracemalloc.get_traced_memory()
    before_rss = resource_usage._read_process_rss_bytes()
    total_log_bytes = 0
    progress.register(batch)
    for index in range(submission_count):
        submission = _make_submission_with_execution_log_bytes(
            batch,
            index=index,
            log_bytes=log_bytes_per_submission,
        )
        total_log_bytes += _estimated_execution_log_payload_bytes(submission)
        progress.record(submission)
        del submission
        gc.collect()
    after_heap, _after_peak = tracemalloc.get_traced_memory()
    after_rss = resource_usage._read_process_rss_bytes()
    tracemalloc.stop()
    storage_root = getattr(progress, "storage_root", None)
    return {
        "total_log_bytes": total_log_bytes,
        "retained_heap_delta": after_heap - before_heap,
        "rss_delta": after_rss - before_rss,
        "blob_bytes": _sum_blob_bytes(storage_root),
    }


def _measure_large_attempt_log_retention(
    progress: FileBackedRunProgress,
    *,
    batch: MinerTaskBatchSpec,
    submission_count: int,
    log_bytes_per_submission: int,
) -> dict[str, int]:
    gc.collect()
    tracemalloc.start()
    before_heap, _before_peak = tracemalloc.get_traced_memory()
    before_rss = resource_usage._read_process_rss_bytes()
    total_log_bytes = 0
    progress.register(batch)
    for index in range(submission_count):
        attempt = _make_attempt_with_execution_log_bytes(
            batch,
            index=index,
            log_bytes=log_bytes_per_submission,
        )
        total_log_bytes += _estimated_attempt_execution_log_payload_bytes(attempt)
        progress.record_terminated_attempt(attempt)
        del attempt
        gc.collect()
    after_heap, _after_peak = tracemalloc.get_traced_memory()
    after_rss = resource_usage._read_process_rss_bytes()
    tracemalloc.stop()
    storage_root = getattr(progress, "storage_root", None)
    return {
        "total_log_bytes": total_log_bytes,
        "retained_heap_delta": after_heap - before_heap,
        "rss_delta": after_rss - before_rss,
        "blob_bytes": _sum_blob_bytes(storage_root),
    }


def test_run_progress_retained_memory_is_sublinear_in_execution_log_bytes(tmp_path: Path) -> None:
    progress = _progress(tmp_path)
    batch = _large_log_batch(submission_count=16)
    measurement = _measure_large_log_retention(
        progress,
        batch=batch,
        submission_count=16,
        log_bytes_per_submission=1 * 1024 * 1024,
    )
    print(f"run_progress_memory_measurement={measurement!r}")

    assert measurement["retained_heap_delta"] < max(
        8 * 1024 * 1024,
        measurement["total_log_bytes"] // 5,
    ), measurement
    if measurement["blob_bytes"] == 0:
        pytest.fail(f"full execution logs were not spooled to blob storage: {measurement!r}")
    assert measurement["blob_bytes"] >= measurement["total_log_bytes"], measurement
    assert measurement["rss_delta"] < max(
        64 * 1024 * 1024,
        measurement["total_log_bytes"],
    ), measurement
    assert progress.discard_batch(batch.batch_id) is True
    assert _sum_blob_bytes(progress.storage_root) == 0


def test_run_progress_retained_memory_is_sublinear_in_attempt_execution_log_bytes(
    tmp_path: Path,
) -> None:
    progress = _progress(tmp_path)
    batch = _large_log_batch(submission_count=16)
    measurement = _measure_large_attempt_log_retention(
        progress,
        batch=batch,
        submission_count=16,
        log_bytes_per_submission=1 * 1024 * 1024,
    )
    print(f"run_progress_attempt_memory_measurement={measurement!r}")

    assert measurement["retained_heap_delta"] < max(
        8 * 1024 * 1024,
        measurement["total_log_bytes"] // 5,
    ), measurement
    if measurement["blob_bytes"] == 0:
        pytest.fail(f"attempt execution logs were not spooled to blob storage: {measurement!r}")
    assert measurement["blob_bytes"] >= measurement["total_log_bytes"], measurement
    assert measurement["rss_delta"] < max(
        64 * 1024 * 1024,
        measurement["total_log_bytes"],
    ), measurement
    assert progress.discard_batch(batch.batch_id) is True
    assert _sum_blob_bytes(progress.storage_root) == 0
