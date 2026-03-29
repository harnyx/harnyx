from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

import harnyx_validator.runtime.evaluation_worker as worker_mod
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import Session, SessionStatus
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.application.services.evaluation_runner import (
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.runtime.evaluation_worker import EvaluationWorker


def _sample_batch() -> MinerTaskBatchSpec:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="example"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="abc", size_bytes=1)
    return MinerTaskBatchSpec(
        batch_id=uuid4(),
        cutoff_at="2025-01-01T00:00:00Z",
        created_at="2025-01-01T00:00:00Z",
        tasks=(task,),
        artifacts=(artifact,),
    )


def _all_pairs(batch: MinerTaskBatchSpec) -> frozenset[tuple[UUID, UUID]]:
    return frozenset((artifact.artifact_id, task.task_id) for artifact in batch.artifacts for task in batch.tasks)


class ProgressSpy:
    def __init__(self) -> None:
        self._recorded_by_batch: dict[UUID, set[tuple[UUID, UUID]]] = {}

    def register(self, batch: MinerTaskBatchSpec) -> None:
        self._recorded_by_batch.setdefault(batch.batch_id, set())

    def record(self, _result) -> None:
        return None

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        submissions,
        _provider_evidence=(),
    ) -> None:
        self.register(batch)
        bucket = self._recorded_by_batch.setdefault(batch.batch_id, set())
        for submission in submissions:
            bucket.add((submission.run.artifact_id, submission.run.task_id))

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        return frozenset(self._recorded_by_batch.get(batch_id, set()))

    def set_recorded_pairs(
        self,
        batch: MinerTaskBatchSpec,
        pairs: Iterable[tuple[UUID, UUID]],
    ) -> None:
        self._recorded_by_batch[batch.batch_id] = set(pairs)


class FakeBatchService:
    """Fake batch service for testing."""

    def __init__(self, *, on_process=None, error: Exception | None = None) -> None:
        self.processed: list[MinerTaskBatchSpec] = []
        self.processed_event = threading.Event()
        self._on_process = on_process
        self._error = error

    def process(self, batch: MinerTaskBatchSpec) -> None:
        self.processed.append(batch)
        if self._on_process is not None:
            self._on_process(batch)
        self.processed_event.set()
        if self._error is not None:
            raise self._error

    async def process_async(self, batch: MinerTaskBatchSpec) -> None:
        self.process(batch)


class HookedBatchInbox(InMemoryBatchInbox):
    def __init__(self, *, on_get=None) -> None:
        super().__init__()
        self._on_get = on_get

    def get(
        self,
        *,
        timeout: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> MinerTaskBatchSpec | None:
        batch = super().get(timeout=timeout, stop_event=stop_event)
        if batch is not None and self._on_get is not None:
            self._on_get(batch)
        return batch


class PersistentFailureBatchService:
    def __init__(self) -> None:
        self.processed: list[MinerTaskBatchSpec] = []
        self.processed_event = threading.Event()

    async def process_async(self, batch: MinerTaskBatchSpec) -> None:
        self.processed.append(batch)
        self.processed_event.set()
        raise RuntimeError("worker boom")


def _completed_submission(batch: MinerTaskBatchSpec) -> MinerTaskRunSubmission:
    artifact = batch.artifacts[0]
    task = batch.tasks[0]
    issued_at = datetime.now(UTC)
    completed_at = issued_at + timedelta(seconds=5)
    session = Session(
        session_id=uuid4(),
        uid=artifact.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=completed_at + timedelta(minutes=5),
        budget_usd=task.budget_usd,
        status=SessionStatus.COMPLETED,
    )
    details = EvaluationDetails(
        score_breakdown=ScoreBreakdown(
            comparison_score=1.0,
            similarity_score=1.0,
            total_score=1.0,
            scoring_version="v1",
        ),
        total_tool_usage=ToolUsageSummary.zero(),
    )
    return MinerTaskRunSubmission(
        batch_id=batch.batch_id,
        validator_uid=1,
        run=MinerTaskRun(
            session_id=session.session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=Response(text="completed"),
            details=details,
            completed_at=completed_at,
        ),
        score=1.0,
        usage=TokenUsageSummary.empty(),
        session=session,
    )


def _validator_batch_failed_error(
    batch: MinerTaskBatchSpec,
    *,
    error_code: str,
    error_message: str,
    task_id: UUID | None,
    exception_type: str | None = None,
    occurred_at: datetime | None = None,
) -> ValidatorBatchFailedError:
    artifact = batch.artifacts[0]
    detail = ValidatorBatchFailureDetail(
        error_code=error_code,
        error_message=error_message,
        occurred_at=occurred_at or datetime.now(UTC),
        artifact_id=artifact.artifact_id,
        task_id=task_id,
        uid=artifact.uid,
        exception_type=exception_type,
    )
    return ValidatorBatchFailedError(
        error_code=error_code,
        message=error_message,
        failure_detail=detail,
    )


@pytest.mark.anyio
async def test_evaluation_worker_drains_inbox():
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    fake_service = FakeBatchService(
        on_process=lambda current_batch: progress.set_recorded_pairs(current_batch, _all_pairs(current_batch)),
    )

    worker = EvaluationWorker(
        batch_service=fake_service,
        batch_inbox=inbox,
        status_provider=status,
        batch_tracker=accept_batch,
    )
    batch = _sample_batch()
    accept_batch.execute(batch)

    worker.start()
    assert await asyncio.to_thread(fake_service.processed_event.wait, timeout=1.0)
    await worker.stop(timeout=1.0)

    assert fake_service.processed
    assert status.state.queued_batches == 0


@pytest.mark.anyio
async def test_evaluation_worker_skips_batch_completed_by_restore_after_dequeue() -> None:
    restore_applied = threading.Event()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch: AcceptEvaluationBatch | None = None

    def _restore_after_get(batch: MinerTaskBatchSpec) -> None:
        if accept_batch is None:
            raise AssertionError("accept_batch should be initialized before restore callback")
        accept_batch.execute(batch, restore_runs=(_completed_submission(batch),))
        restore_applied.set()

    inbox = HookedBatchInbox(on_get=_restore_after_get)
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    fake_service = FakeBatchService()
    worker = EvaluationWorker(
        batch_service=fake_service,
        batch_inbox=inbox,
        status_provider=status,
        batch_tracker=accept_batch,
    )
    batch = _sample_batch()
    accept_batch.execute(batch)

    worker.start()
    try:
        assert await asyncio.to_thread(restore_applied.wait, timeout=1.0)
        await asyncio.sleep(0.05)
    finally:
        await worker.stop(timeout=1.0)

    assert fake_service.processed == []
    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert status.state.queued_batches == 0


@pytest.mark.anyio
async def test_evaluation_worker_does_not_retry_incomplete_batch_after_service_boundary_exception() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _sample_batch()
    accept_batch.execute(batch)
    fake_service = PersistentFailureBatchService()

    worker = EvaluationWorker(
        batch_service=fake_service,
        batch_inbox=inbox,
        status_provider=status,
        batch_tracker=accept_batch,
    )

    worker.start()
    try:
        assert await asyncio.to_thread(fake_service.processed_event.wait, timeout=1.0)
        await asyncio.sleep(0.05)
    finally:
        await worker.stop(timeout=1.0)

    assert len(fake_service.processed) == 1
    assert accept_batch.lifecycle_for(batch.batch_id) == "failed"
    assert status.state.last_error == "unexpected_batch_failure"
    assert len(inbox) == 0


@pytest.mark.anyio
async def test_evaluation_worker_marks_batch_failed_when_unexpected_error_happens_after_progress() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _sample_batch()
    accept_batch.execute(batch)
    fake_service = FakeBatchService(
        on_process=lambda current_batch: progress.set_recorded_pairs(current_batch, _all_pairs(current_batch)),
        error=RuntimeError("worker boom"),
    )

    worker = EvaluationWorker(
        batch_service=fake_service,
        batch_inbox=inbox,
        status_provider=status,
        batch_tracker=accept_batch,
    )

    worker.start()
    assert await asyncio.to_thread(fake_service.processed_event.wait, timeout=1.0)
    await worker.stop(timeout=1.0)

    accept_batch.execute(batch)

    assert len(inbox) == 0
    assert accept_batch.lifecycle_for(batch.batch_id) == "failed"
    assert status.state.last_error == "unexpected_batch_failure"


@pytest.mark.anyio
async def test_evaluation_worker_sends_failed_batch_exception_to_sentry(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(worker_mod, "capture_exception", captured.append)

    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _sample_batch()
    accept_batch.execute(batch)
    fake_service = PersistentFailureBatchService()

    worker = EvaluationWorker(
        batch_service=fake_service,
        batch_inbox=inbox,
        status_provider=status,
        batch_tracker=accept_batch,
    )

    worker.start()
    try:
        assert await asyncio.to_thread(fake_service.processed_event.wait, timeout=1.0)
        await asyncio.sleep(0.05)
    finally:
        await worker.stop(timeout=1.0)

    assert [str(exc) for exc in captured] == ["worker boom"]
    assert len(fake_service.processed) == 1
    assert len(inbox) == 0


@pytest.mark.anyio
async def test_evaluation_worker_sends_batch_failure_scope_to_sentry(monkeypatch) -> None:
    captured: list[tuple[BaseException, dict[str, object]]] = []

    def _capture_exception(exc: BaseException, **kwargs: object) -> None:
        captured.append((exc, kwargs))

    monkeypatch.setattr(worker_mod, "capture_exception", _capture_exception)

    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _sample_batch()
    accept_batch.execute(batch)
    occurred_at = datetime.now(UTC)
    error = _validator_batch_failed_error(
        batch,
        error_code="provider_batch_failure",
        error_message=(
            "provider failure threshold reached (provider=desearch model=search_web failed_calls=10 total_calls=10)"
        ),
        task_id=batch.tasks[0].task_id,
        exception_type="SandboxInvocationError",
        occurred_at=occurred_at,
    )
    fake_service = FakeBatchService(error=error)

    worker = EvaluationWorker(
        batch_service=fake_service,
        batch_inbox=inbox,
        status_provider=status,
        batch_tracker=accept_batch,
    )

    worker.start()
    try:
        assert await asyncio.to_thread(fake_service.processed_event.wait, timeout=1.0)
        await asyncio.sleep(0.05)
    finally:
        await worker.stop(timeout=1.0)

    assert len(captured) == 1
    exc, payload = captured[0]
    assert exc is error
    assert payload["tags"] == {
        "error_code": "provider_batch_failure",
        "failure_kind": "provider",
        "provider": "desearch",
        "model": "search_web",
    }
    assert payload["context_name"] == "validator_batch"
    assert payload["context"] == {
        "batch_id": str(batch.batch_id),
        "error_code": "provider_batch_failure",
        "error_message": (
            "provider failure threshold reached (provider=desearch model=search_web failed_calls=10 total_calls=10)"
        ),
        "occurred_at": occurred_at.isoformat(),
        "artifact_id": str(batch.artifacts[0].artifact_id),
        "task_id": str(batch.tasks[0].task_id),
        "uid": batch.artifacts[0].uid,
        "exception_type": "SandboxInvocationError",
        "provider": "desearch",
        "model": "search_web",
        "failed_calls": 10,
        "total_calls": 10,
    }
    assert payload["fingerprint"] == [
        "validator-batch",
        "provider_batch_failure",
        "desearch",
        "search_web",
    ]


def test_batch_failure_capture_payload_uses_base_grouping_for_non_provider_errors() -> None:
    batch = _sample_batch()
    occurred_at = datetime.now(UTC)
    error = _validator_batch_failed_error(
        batch,
        error_code="artifact_breaker_tripped",
        error_message="validator artifact breaker tripped across 3 artifacts",
        task_id=None,
        exception_type="SandboxInvocationError",
        occurred_at=occurred_at,
    )

    payload = worker_mod._batch_failure_capture_payload(batch_id=batch.batch_id, exc=error)

    assert payload.tags == {
        "error_code": "artifact_breaker_tripped",
        "failure_kind": "artifact",
    }
    assert payload.context_name == "validator_batch"
    assert payload.context == {
        "batch_id": str(batch.batch_id),
        "error_code": "artifact_breaker_tripped",
        "error_message": "validator artifact breaker tripped across 3 artifacts",
        "occurred_at": occurred_at.isoformat(),
        "artifact_id": str(batch.artifacts[0].artifact_id),
        "uid": batch.artifacts[0].uid,
        "exception_type": "SandboxInvocationError",
    }
    assert payload.fingerprint == [
        "validator-batch",
        "artifact_breaker_tripped",
    ]
