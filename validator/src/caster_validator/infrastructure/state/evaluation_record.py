"""In-memory evaluation record store."""

from __future__ import annotations

from threading import Lock

from caster_validator.application.dto.evaluation import EvaluationCloseout
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort


class InMemoryEvaluationRecordStore(EvaluationRecordPort):
    """Stores evaluation closeouts in memory."""

    def __init__(self) -> None:
        self._records: list[EvaluationCloseout] = []
        self._lock = Lock()

    def record(self, closeout: EvaluationCloseout) -> None:
        with self._lock:
            self._records.append(closeout)

    def records(self) -> tuple[EvaluationCloseout, ...]:
        with self._lock:
            return tuple(self._records)


__all__ = ["InMemoryEvaluationRecordStore"]
