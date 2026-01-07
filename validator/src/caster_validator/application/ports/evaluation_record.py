"""Port describing durable evaluation record persistence."""

from __future__ import annotations

from typing import Protocol

from caster_validator.application.dto.evaluation import EvaluationCloseout


class EvaluationRecordPort(Protocol):
    """Persists evaluation closeouts to an external store."""

    def record(self, closeout: EvaluationCloseout) -> None:
        """Persist the supplied evaluation closeout payload."""


__all__ = ["EvaluationRecordPort"]
