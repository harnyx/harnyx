"""Port describing durable miner-task run persistence."""

from __future__ import annotations

from typing import Protocol

from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission


class EvaluationRecordPort(Protocol):
    """Persists miner-task run submissions to an external store.

    The owning record store must be idempotent per `(batch_id, artifact_id, task_id)`
    and fail loudly when a duplicate pair attempts to persist different contents.
    """

    def record(self, result: MinerTaskRunSubmission) -> None:
        """Persist the supplied miner-task run submission payload."""


__all__ = ["EvaluationRecordPort"]
