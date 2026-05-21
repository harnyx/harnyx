"""Compact in-memory miner-task run record store."""

from __future__ import annotations

from threading import Lock
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort


class CompactEvaluationRecordStore(EvaluationRecordPort):
    """Stores miner-task run submissions without full execution logs."""

    def __init__(self) -> None:
        self._records_by_pair: dict[tuple[UUID, UUID, UUID], MinerTaskRunSubmission] = {}
        self._lock = Lock()

    def record(self, result: MinerTaskRunSubmission) -> None:
        key = (result.batch_id, result.run.artifact_id, result.run.task_id)
        compact_result = _compact_submission_for_evaluation_record(result)
        with self._lock:
            existing = self._records_by_pair.get(key)
            if existing is not None:
                if existing != compact_result:
                    raise RuntimeError(
                        "batch already recorded a different result for artifact/task pair"
                    )
                return
            self._records_by_pair[key] = compact_result

    def records(self) -> tuple[MinerTaskRunSubmission, ...]:
        with self._lock:
            return tuple(self._records_by_pair.values())


def _compact_submission_for_evaluation_record(
    result: MinerTaskRunSubmission,
) -> MinerTaskRunSubmission:
    return result.model_copy(update={"execution_log": ()})


__all__ = ["CompactEvaluationRecordStore"]
