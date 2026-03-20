from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable
from uuid import UUID, uuid4

import pytest

import harnyx_validator.runtime.evaluation_worker as worker_mod
from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec
from harnyx_validator.application.status import StatusProvider
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


@pytest.mark.anyio
async def test_evaluation_worker_drains_inbox():
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
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
    assert await asyncio.to_thread(fake_service.processed_event.wait, timeout=1.0)
    await worker.stop(timeout=1.0)

    assert fake_service.processed
    assert status.state.queued_batches == 0


@pytest.mark.anyio
async def test_evaluation_worker_marks_batch_retryable_when_pairs_remain() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _sample_batch()
    accept_batch.execute(batch)
    fake_service = FakeBatchService(error=RuntimeError("worker boom"))

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

    assert len(inbox) == 1
    assert status.state.last_error == "worker boom"


@pytest.mark.anyio
async def test_evaluation_worker_marks_batch_completed_when_all_pairs_are_recorded() -> None:
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
    assert status.state.last_error == "worker boom"


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
    fake_service = FakeBatchService(error=RuntimeError("worker boom"))

    worker = EvaluationWorker(
        batch_service=fake_service,
        batch_inbox=inbox,
        status_provider=status,
        batch_tracker=accept_batch,
    )

    worker.start()
    assert await asyncio.to_thread(fake_service.processed_event.wait, timeout=1.0)
    await worker.stop(timeout=1.0)

    assert [str(exc) for exc in captured] == ["worker boom"]
