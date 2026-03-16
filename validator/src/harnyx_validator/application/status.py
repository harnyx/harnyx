"""Simple status snapshot provider for the validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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


class StatusSnapshot(TypedDict):
    status: str
    last_batch_id: str | None
    last_started_at: str | None
    last_completed_at: str | None
    running: bool
    queued_batches: int
    last_error: str | None
    last_weight_submission_at: str | None
    last_weight_error: str | None


@dataclass(slots=True)
class StatusProvider:
    """Tracks lightweight runtime status for RPC inspection."""

    state: InMemoryStatus = field(default_factory=InMemoryStatus)

    def snapshot(self) -> StatusSnapshot:
        if self.state.running:
            status_value = "running"
        elif self.state.last_error:
            status_value = "error"
        else:
            status_value = "idle"
        return {
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

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None


__all__ = ["StatusProvider", "InMemoryStatus", "StatusSnapshot"]
