from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID, uuid4

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox


class ProgressSpy:
    def __init__(self) -> None:
        self.register_attempts: list[MinerTaskBatchSpec] = []
        self.registered: list[MinerTaskBatchSpec] = []
        self._batches_by_id: dict[UUID, MinerTaskBatchSpec] = {}
        self._recorded_by_batch: dict[UUID, set[tuple[UUID, UUID]]] = {}

    def register(self, batch: MinerTaskBatchSpec) -> None:
        self.register_attempts.append(batch)
        existing = self._batches_by_id.get(batch.batch_id)
        if existing is not None:
            if existing != batch:
                raise RuntimeError("batch_id already exists with different contents")
            return
        self._batches_by_id[batch.batch_id] = batch
        self.registered.append(batch)
        self._recorded_by_batch.setdefault(batch.batch_id, set())

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        return frozenset(self._recorded_by_batch.get(batch_id, set()))

    def set_recorded_pairs(
        self,
        batch: MinerTaskBatchSpec,
        pairs: Iterable[tuple[UUID, UUID]],
    ) -> None:
        self._recorded_by_batch[batch.batch_id] = set(pairs)


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


def _all_pairs(batch: MinerTaskBatchSpec) -> frozenset[tuple[UUID, UUID]]:
    return frozenset((artifact.artifact_id, task.task_id) for artifact in batch.artifacts for task in batch.tasks)


def test_accept_batch_ignores_exact_duplicate_replay_while_queued() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    accept_batch.execute(batch)

    assert len(inbox) == 1
    assert status.state.queued_batches == 1
    assert progress.register_attempts == [batch]
    assert progress.registered == [batch]


def test_accept_batch_ignores_exact_duplicate_replay_while_processing() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    accept_batch.execute(batch)

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert progress.register_attempts == [batch]
    assert progress.registered == [batch]


def test_accept_batch_ignores_exact_duplicate_replay_while_completed() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    progress.set_recorded_pairs(batch, _all_pairs(batch))
    accept_batch.mark_completed(batch.batch_id)
    accept_batch.execute(batch)

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert progress.register_attempts == [batch]
    assert progress.registered == [batch]


def test_accept_batch_duplicate_processing_replay_becomes_completed_when_all_pairs_are_recorded() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    progress.set_recorded_pairs(batch, _all_pairs(batch))
    accept_batch.execute(batch)

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert progress.register_attempts == [batch]


def test_accept_batch_complete_if_recorded_leaves_processing_when_pairs_are_missing() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)

    completed = accept_batch.mark_completed_if_recorded(batch.batch_id)

    assert completed is False
    assert accept_batch.lifecycle_for(batch.batch_id) == "processing"
    assert len(inbox) == 0
    assert status.state.queued_batches == 0


def test_accept_batch_complete_if_recorded_marks_completed_when_all_pairs_are_recorded() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    progress.set_recorded_pairs(batch, _all_pairs(batch))

    completed = accept_batch.mark_completed_if_recorded(batch.batch_id)

    assert completed is True
    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert len(inbox) == 0
    assert status.state.queued_batches == 0


def test_accept_batch_rejects_conflicting_replay() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch_id = uuid4()
    batch = _make_batch(batch_id=batch_id, query_text="original")
    conflicting = _make_batch(batch_id=batch_id, query_text="different")

    accept_batch.execute(batch)

    with pytest.raises(RuntimeError, match="batch_id already exists with different contents"):
        accept_batch.execute(conflicting)

    assert len(inbox) == 1
    assert status.state.queued_batches == 1
    assert progress.register_attempts == [batch]
    assert progress.registered == [batch]
