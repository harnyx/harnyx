"""Use case for accepting platform-supplied batches into the inbox."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import Lock
from typing import Literal
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission
from harnyx_validator.application.ports.progress import (
    ConsumedAttemptNumber,
    ProgressRecorder,
    ProviderFailureEvidence,
    TerminatedMinerTaskAttemptOrdinal,
)
from harnyx_validator.application.services.evaluation_runner import ValidatorBatchFailureDetail
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox

BatchLifecycle = Literal["restoring", "queued", "processing", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class _AcceptedBatchState:
    batch: MinerTaskBatchSpec
    lifecycle: BatchLifecycle
    error_code: str | None = None
    failure_detail: ValidatorBatchFailureDetail | None = None
    terminal_at: datetime | None = None


@dataclass(slots=True)
class AcceptEvaluationBatch:
    """Store the provided batch for later execution."""

    inbox: InMemoryBatchInbox
    status: StatusProvider | None
    progress: ProgressRecorder
    _accepted_batches: dict[UUID, _AcceptedBatchState] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def execute(
        self,
        batch: MinerTaskBatchSpec,
        *,
        restore_runs: Sequence[MinerTaskRunSubmission] = (),
        restore_provider_evidence: Sequence[ProviderFailureEvidence] = (),
    ) -> None:
        """Compatibility path for immediate local restore and queueing."""
        with self._lock:
            state = self._accepted_batch_state(batch.batch_id)
            if state is not None:
                if state.batch != batch:
                    raise RuntimeError("batch_id already exists with different contents")

            self.progress.restore_completed_runs(
                batch,
                restore_runs,
                restore_provider_evidence,
            )
            if state is not None:
                if state.lifecycle == "queued" and self._all_pairs_recorded(state.batch):
                    self._accepted_batches[batch.batch_id] = replace(
                        state,
                        lifecycle="completed",
                        terminal_at=_utcnow(),
                    )
                    self.inbox.discard(batch.batch_id)
                    self._update_status_queue_length()
                return
            if self._all_pairs_recorded(batch):
                self._accepted_batches[batch.batch_id] = _AcceptedBatchState(
                    batch=batch,
                    lifecycle="completed",
                    terminal_at=_utcnow(),
                )
                return
            self._queue_new_batch(batch)

    def register_for_restore(self, batch: MinerTaskBatchSpec) -> bool:
        with self._lock:
            state = self._accepted_batch_state(batch.batch_id)
            if state is not None:
                if state.batch != batch:
                    raise RuntimeError("batch_id already exists with different contents")
                return False
            self.progress.register(batch)
            if self._all_pairs_recorded(batch):
                self._accepted_batches[batch.batch_id] = _AcceptedBatchState(
                    batch=batch,
                    lifecycle="completed",
                    terminal_at=_utcnow(),
                )
                return False
            self._accepted_batches[batch.batch_id] = _AcceptedBatchState(
                batch=batch,
                lifecycle="restoring",
                terminal_at=None,
            )
            return True

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        runs: Sequence[MinerTaskRunSubmission],
        provider_evidence: Sequence[ProviderFailureEvidence] = (),
    ) -> None:
        with self._lock:
            state = self._require_state(batch.batch_id)
            if state.batch != batch:
                raise RuntimeError("batch_id already exists with different contents")
            self.progress.restore_completed_runs(batch, runs, provider_evidence)

    def restore_attempt_number_high_waters(
        self,
        batch_id: UUID,
        *,
        terminated: Sequence[TerminatedMinerTaskAttemptOrdinal],
        consumed: Sequence[ConsumedAttemptNumber],
    ) -> None:
        with self._lock:
            self._require_state(batch_id)
            self.progress.restore_attempt_number_high_waters(batch_id, terminated, consumed)

    def restore_progress_floor(self, batch_id: UUID, sequence: int) -> None:
        with self._lock:
            self._require_state(batch_id)
            self.progress.restore_progress_floor(batch_id, sequence)

    def queue_after_restore(self, batch_id: UUID) -> None:
        with self._lock:
            state = self._require_state(batch_id)
            if state.lifecycle == "queued" and self._all_pairs_recorded(state.batch):
                self._accepted_batches[batch_id] = replace(
                    state,
                    lifecycle="completed",
                    error_code=None,
                    failure_detail=None,
                    terminal_at=_utcnow(),
                )
                self.inbox.discard(batch_id)
                self._update_status_queue_length()
                return
            if state.lifecycle in {"queued", "processing", "completed", "failed"}:
                return
            if self._all_pairs_recorded(state.batch):
                self._accepted_batches[batch_id] = replace(
                    state,
                    lifecycle="completed",
                    error_code=None,
                    failure_detail=None,
                    terminal_at=_utcnow(),
                )
                self.inbox.discard(batch_id)
                self._update_status_queue_length()
                return
            self._accepted_batches[batch_id] = replace(
                state,
                lifecycle="queued",
                error_code=None,
                failure_detail=None,
                terminal_at=None,
            )
            self.inbox.put(state.batch)
            self._update_status_queue_length()

    def batch_for(self, batch_id: UUID) -> MinerTaskBatchSpec:
        with self._lock:
            return self._require_state(batch_id).batch

    def begin_processing(self, batch_id: UUID) -> bool:
        with self._lock:
            state = self._require_state(batch_id)
            if state.lifecycle == "completed":
                return False
            if state.lifecycle != "queued":
                raise RuntimeError(f"batch_id {batch_id} cannot begin processing from {state.lifecycle}")
            if self._all_pairs_recorded(state.batch):
                self._accepted_batches[batch_id] = replace(
                    state,
                    lifecycle="completed",
                    error_code=None,
                    terminal_at=_utcnow(),
                )
                return False
            self._accepted_batches[batch_id] = replace(
                state,
                lifecycle="processing",
                error_code=None,
                failure_detail=None,
                terminal_at=None,
            )
            return True

    def mark_processing(self, batch_id: UUID) -> None:
        self._set_lifecycle(batch_id, "processing")

    def mark_completed(
        self,
        batch_id: UUID,
        *,
        terminal_at: datetime | None = None,
    ) -> None:
        with self._lock:
            state = self._require_state(batch_id)
            if not self._all_pairs_recorded(state.batch):
                raise RuntimeError("cannot mark batch completed before all pairs are recorded")
            self._accepted_batches[batch_id] = replace(
                state,
                lifecycle="completed",
                error_code=None,
                failure_detail=None,
                terminal_at=terminal_at or _utcnow(),
            )

    def mark_failed(
        self,
        batch_id: UUID,
        *,
        error_code: str,
        failure_detail: ValidatorBatchFailureDetail | None,
        terminal_at: datetime | None = None,
    ) -> None:
        with self._lock:
            state = self._require_state(batch_id)
            self._accepted_batches[batch_id] = replace(
                state,
                lifecycle="failed",
                error_code=error_code,
                failure_detail=failure_detail,
                terminal_at=terminal_at or _utcnow(),
            )

    def lifecycle_for(self, batch_id: UUID) -> BatchLifecycle | None:
        with self._lock:
            state = self._accepted_batches.get(batch_id)
            if state is None:
                return None
            return state.lifecycle

    def public_lifecycle_for(self, batch_id: UUID) -> BatchLifecycle | None:
        lifecycle = self.lifecycle_for(batch_id)
        if lifecycle == "restoring":
            return "queued"
        return lifecycle

    def error_code_for(self, batch_id: UUID) -> str | None:
        with self._lock:
            state = self._accepted_batches.get(batch_id)
            if state is None:
                return None
            return state.error_code

    def failure_detail_for(self, batch_id: UUID) -> ValidatorBatchFailureDetail | None:
        with self._lock:
            state = self._accepted_batches.get(batch_id)
            if state is None:
                return None
            return state.failure_detail

    def terminal_batches_older_than(self, cutoff: datetime) -> tuple[UUID, ...]:
        normalized_cutoff = _as_utc(cutoff)
        with self._lock:
            return tuple(
                batch_id
                for batch_id, state in self._accepted_batches.items()
                if _is_terminal(state)
                and state.terminal_at is not None
                and _as_utc(state.terminal_at) <= normalized_cutoff
            )

    def forget_terminal_batch(self, batch_id: UUID, *, older_than: datetime) -> bool:
        normalized_cutoff = _as_utc(older_than)
        with self._lock:
            state = self._accepted_batches.get(batch_id)
            if (
                state is None
                or not _is_terminal(state)
                or state.terminal_at is None
                or _as_utc(state.terminal_at) > normalized_cutoff
            ):
                return False
            self._accepted_batches.pop(batch_id)
            self.inbox.discard(batch_id)
            self._update_status_queue_length()
            return True

    def prune_terminal_batch(
        self,
        batch_id: UUID,
        *,
        older_than: datetime,
        cleanup: Callable[[UUID], object],
    ) -> bool:
        normalized_cutoff = _as_utc(older_than)
        with self._lock:
            state = self._accepted_batches.get(batch_id)
            if (
                state is None
                or not _is_terminal(state)
                or state.terminal_at is None
                or _as_utc(state.terminal_at) > normalized_cutoff
            ):
                return False
            cleanup(batch_id)
            self._accepted_batches.pop(batch_id)
            self.inbox.discard(batch_id)
            self._update_status_queue_length()
            return True

    def _queue_new_batch(self, batch: MinerTaskBatchSpec) -> None:
        self._accepted_batches[batch.batch_id] = _AcceptedBatchState(batch=batch, lifecycle="queued")
        self.inbox.put(batch)
        self._update_status_queue_length()

    def _accepted_batch_state(self, batch_id: UUID) -> _AcceptedBatchState | None:
        return self._accepted_batches.get(batch_id)

    def _set_lifecycle(self, batch_id: UUID, lifecycle: BatchLifecycle) -> None:
        with self._lock:
            state = self._require_state(batch_id)
            self._accepted_batches[batch_id] = replace(
                state,
                lifecycle=lifecycle,
                error_code=None,
                failure_detail=None,
                terminal_at=None,
            )

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


def _is_terminal(state: _AcceptedBatchState) -> bool:
    return state.lifecycle in {"completed", "failed"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["AcceptEvaluationBatch", "BatchLifecycle"]
