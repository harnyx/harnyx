"""Port for recording evaluation progress snapshots."""

from __future__ import annotations

from typing import Protocol

from caster_validator.application.dto.evaluation import EvaluationCloseout


class ProgressRecorder(Protocol):
    def record(self, closeout: EvaluationCloseout) -> None:
        ...


__all__ = ["ProgressRecorder"]

