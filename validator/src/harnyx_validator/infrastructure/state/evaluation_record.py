"""In-memory miner-task run record store."""

from __future__ import annotations

from threading import Lock
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort


class InMemoryEvaluationRecordStore(EvaluationRecordPort):
    """Stores miner-task run submissions in memory."""

    def __init__(self) -> None:
        self._records_by_pair: dict[tuple[UUID, UUID, UUID], MinerTaskRunSubmission] = {}
        self._lock = Lock()

    def record(self, result: MinerTaskRunSubmission) -> None:
        key = (result.batch_id, result.run.artifact_id, result.run.task_id)
        with self._lock:
            existing = self._records_by_pair.get(key)
            if existing is not None:
                if existing != result:
                    raise RuntimeError(
                        "batch already recorded a different result for artifact/task pair"
                    )
                return
            self._records_by_pair[key] = result

    def records(self) -> tuple[MinerTaskRunSubmission, ...]:
        with self._lock:
            return tuple(self._records_by_pair.values())


__all__ = ["InMemoryEvaluationRecordStore"]
