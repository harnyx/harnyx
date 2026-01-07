"""In-memory tracker for per-run evaluation progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from caster_validator.application.dto.evaluation import EvaluationCloseout


@dataclass(slots=True)
class InMemoryRunProgress:
    expected: dict[UUID, int] = field(default_factory=dict)
    closeouts_by_run: dict[UUID, list[EvaluationCloseout]] = field(default_factory=dict)

    def register(self, run_id: UUID, *, uids: tuple[int, ...], claims_count: int) -> None:
        total = len(uids) * claims_count
        self.expected[run_id] = total

    def record(self, closeout: EvaluationCloseout) -> None:
        bucket = self.closeouts_by_run.setdefault(closeout.run_id, [])
        bucket.append(closeout)

    def snapshot(self, run_id: UUID) -> dict[str, object]:
        closeouts = tuple(self.closeouts_by_run.get(run_id, ()))
        total = int(self.expected.get(run_id, 0))
        completed = len(closeouts)
        remaining = max(0, total - completed)
        return {
            "run_id": run_id,
            "total": total,
            "completed": completed,
            "remaining": remaining,
            "closeouts": closeouts,
        }


__all__ = ["InMemoryRunProgress"]
