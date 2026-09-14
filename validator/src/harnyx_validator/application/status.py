"""Simple status snapshot provider for the validator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import TypedDict
from uuid import UUID


@dataclass
class InMemoryStatus:
    last_batch_id: UUID | None = None
    last_started_at: datetime | None = None
    last_completed_at: datetime | None = None
    running: bool = False
    last_error: str | None = None
    queued_batches: int = 0
    last_weight_submission_at: datetime | None = None
    last_weight_error: str | None = None
    platform_registration_ready: bool = False
    platform_registration_error: str | None = None
    platform_registration_last_succeeded_at: datetime | None = None
    platform_registration_last_refresh_error: str | None = None
    platform_registration_refresh_failure_count: int = 0
    auth_ready: bool = False
    auth_error: str | None = None


class StatusSnapshot(TypedDict):
    rating_worker_ready: bool
    status: str
    last_batch_id: str | None
    last_started_at: str | None
    last_completed_at: str | None
    running: bool
    queued_batches: int
    last_error: str | None
    last_weight_submission_at: str | None
    last_weight_error: str | None


@dataclass(frozen=True, slots=True)
class BatchActivitySnapshot:
    last_activity_at: datetime | None = None
    last_activity_stage: str | None = None
    active_artifact_count: int = 0
    active_task_session_count: int = 0


@dataclass(slots=True)
class _MutableBatchActivity:
    last_activity_at: datetime | None = None
    last_activity_stage: str | None = None
    active_artifact_ids: set[UUID] = field(default_factory=set)
    active_task_session_count: int = 0


class BatchActivityTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._activity_by_batch: dict[UUID, _MutableBatchActivity] = {}

    def mark_batch_started(self, batch_id: UUID) -> None:
        self._mark(batch_id, "batch_started")

    def mark_batch_finished(self, batch_id: UUID) -> None:
        with self._lock:
            self._activity_by_batch.pop(batch_id, None)

    def mark_artifact_started(self, batch_id: UUID, artifact_id: UUID) -> None:
        with self._lock:
            activity = self._activity_by_batch.setdefault(batch_id, _MutableBatchActivity())
            activity.active_artifact_ids.add(artifact_id)
            self._record_activity(activity, "artifact_started")

    def mark_artifact_stage(self, batch_id: UUID, stage: str) -> None:
        self._mark(batch_id, stage)

    def mark_artifact_finished(self, batch_id: UUID, artifact_id: UUID) -> None:
        with self._lock:
            activity = self._activity_by_batch.setdefault(batch_id, _MutableBatchActivity())
            activity.active_artifact_ids.discard(artifact_id)
            self._record_activity(activity, "artifact_finished")

    def mark_task_session_started(self, batch_id: UUID) -> None:
        with self._lock:
            activity = self._activity_by_batch.setdefault(batch_id, _MutableBatchActivity())
            activity.active_task_session_count += 1
            self._record_activity(activity, "task_session_started")

    def mark_task_session_finished(self, batch_id: UUID) -> None:
        with self._lock:
            activity = self._activity_by_batch.setdefault(batch_id, _MutableBatchActivity())
            activity.active_task_session_count = max(0, activity.active_task_session_count - 1)
            self._record_activity(activity, "task_session_finished")

    def snapshot(self, batch_id: UUID) -> BatchActivitySnapshot:
        with self._lock:
            activity = self._activity_by_batch.get(batch_id)
            if activity is None:
                return BatchActivitySnapshot()
            return BatchActivitySnapshot(
                last_activity_at=activity.last_activity_at,
                last_activity_stage=activity.last_activity_stage,
                active_artifact_count=len(activity.active_artifact_ids),
                active_task_session_count=activity.active_task_session_count,
            )

    def _mark(self, batch_id: UUID, stage: str) -> None:
        with self._lock:
            activity = self._activity_by_batch.setdefault(batch_id, _MutableBatchActivity())
            self._record_activity(activity, stage)

    @staticmethod
    def _record_activity(activity: _MutableBatchActivity, stage: str) -> None:
        activity.last_activity_at = datetime.now(UTC)
        activity.last_activity_stage = stage


@dataclass(slots=True)
class StatusProvider:
    """Tracks lightweight runtime status for RPC inspection."""

    state: InMemoryStatus = field(default_factory=InMemoryStatus)
    rating_worker_readiness: Callable[[], bool] = field(default=lambda: False)

    def snapshot(self) -> StatusSnapshot:
        if self.state.running:
            status_value = "running"
        elif self.state.last_error:
            status_value = "error"
        else:
            status_value = "idle"
        return {
            "rating_worker_ready": self.rating_worker_readiness(),
            "status": status_value,
            "last_batch_id": str(self.state.last_batch_id) if self.state.last_batch_id else None,
            "last_started_at": self._iso(self.state.last_started_at),
            "last_completed_at": self._iso(self.state.last_completed_at),
            "running": self.state.running,
            "queued_batches": self.state.queued_batches,
            "last_error": self.state.last_error,
            "last_weight_submission_at": self._iso(self.state.last_weight_submission_at),
            "last_weight_error": self.state.last_weight_error,
        }

    def mark_platform_registration_succeeded(self) -> None:
        self.state.platform_registration_ready = True
        self.state.platform_registration_error = None
        self.state.platform_registration_last_succeeded_at = datetime.now(UTC)
        self.state.platform_registration_last_refresh_error = None
        self.state.platform_registration_refresh_failure_count = 0

    def mark_platform_registration_failed(self, error: str) -> None:
        self.state.platform_registration_ready = False
        self.state.platform_registration_error = error

    def mark_platform_registration_refresh_failed(self, error: str) -> None:
        self.state.platform_registration_last_refresh_error = error
        self.state.platform_registration_refresh_failure_count += 1
        if not self.state.platform_registration_ready:
            self.state.platform_registration_error = error

    def platform_registration_ready(self) -> bool:
        return self.state.platform_registration_ready

    def platform_registration_error(self) -> str | None:
        return self.state.platform_registration_error

    def mark_auth_ready(self) -> None:
        self.state.auth_ready = True
        self.state.auth_error = None

    def mark_auth_unavailable(self, error: str) -> None:
        self.state.auth_ready = False
        self.state.auth_error = error

    def auth_ready(self) -> bool:
        return self.state.auth_ready

    def auth_error(self) -> str | None:
        return self.state.auth_error

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None


__all__ = [
    "BatchActivitySnapshot",
    "BatchActivityTracker",
    "StatusProvider",
    "InMemoryStatus",
    "StatusSnapshot",
]
