"""Use case for accepting platform-supplied batches into the inbox."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Literal
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox

BatchLifecycle = Literal["queued", "processing", "completed"]


@dataclass(frozen=True, slots=True)
class _AcceptedBatchState:
    batch: MinerTaskBatchSpec
    lifecycle: BatchLifecycle


@dataclass(slots=True)
class AcceptEvaluationBatch:
    """Store the provided batch for later execution."""

    inbox: InMemoryBatchInbox
    status: StatusProvider | None
    progress: ProgressRecorder
    _accepted_batches: dict[UUID, _AcceptedBatchState] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def execute(self, batch: MinerTaskBatchSpec) -> None:
        with self._lock:
            state = self._accepted_batch_state(batch.batch_id)
            if state is None:
                self.progress.register(batch)
                self._queue_new_batch(batch)
                return
            if state.batch != batch:
                raise RuntimeError("batch_id already exists with different contents")
            if state.lifecycle == "processing" and self._all_pairs_recorded(state.batch):
                self._accepted_batches[batch.batch_id] = replace(state, lifecycle="completed")
            return

    def mark_processing(self, batch_id: UUID) -> None:
        self._set_lifecycle(batch_id, "processing")

    def mark_completed(self, batch_id: UUID) -> None:
        with self._lock:
            state = self._require_state(batch_id)
            if not self._all_pairs_recorded(state.batch):
                raise RuntimeError("cannot mark batch completed before all pairs are recorded")
            self._accepted_batches[batch_id] = replace(state, lifecycle="completed")

    def mark_completed_if_recorded(self, batch_id: UUID) -> bool:
        with self._lock:
            state = self._require_state(batch_id)
            if state.lifecycle != "processing":
                raise RuntimeError("can only complete a batch from processing")
            if not self._all_pairs_recorded(state.batch):
                return False
            self._accepted_batches[batch_id] = replace(state, lifecycle="completed")
            return True

    def lifecycle_for(self, batch_id: UUID) -> BatchLifecycle | None:
        with self._lock:
            state = self._accepted_batches.get(batch_id)
            if state is None:
                return None
            return state.lifecycle

    def _queue_new_batch(self, batch: MinerTaskBatchSpec) -> None:
        self._accepted_batches[batch.batch_id] = _AcceptedBatchState(batch=batch, lifecycle="queued")
        self.inbox.put(batch)
        self._update_status_queue_length()

    def _accepted_batch_state(self, batch_id: UUID) -> _AcceptedBatchState | None:
        return self._accepted_batches.get(batch_id)

    def _set_lifecycle(self, batch_id: UUID, lifecycle: BatchLifecycle) -> None:
        with self._lock:
            state = self._require_state(batch_id)
            self._accepted_batches[batch_id] = replace(state, lifecycle=lifecycle)

    def _require_state(self, batch_id: UUID) -> _AcceptedBatchState:
        state = self._accepted_batches.get(batch_id)
        if state is None:
            raise RuntimeError(f"batch_id {batch_id} was not accepted before lifecycle transition")
        return state

    def _all_pairs_recorded(self, batch: MinerTaskBatchSpec) -> bool:
        expected_pairs = frozenset(
            (artifact.artifact_id, task.task_id)
            for artifact in batch.artifacts
            for task in batch.tasks
        )
        return expected_pairs.issubset(self.progress.recorded_pairs(batch.batch_id))

    def _update_status_queue_length(self) -> None:
        if self.status is not None:
            self.status.state.queued_batches = len(self.inbox)


__all__ = ["AcceptEvaluationBatch", "BatchLifecycle"]
