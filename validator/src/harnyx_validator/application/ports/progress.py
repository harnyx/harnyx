"""Port for recording miner-task batch progress snapshots."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission


class ProgressRecorder(Protocol):
    def register(self, batch: MinerTaskBatchSpec) -> None:
        ...

    def record(self, result: MinerTaskRunSubmission) -> None:
        ...

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        ...


__all__ = ["ProgressRecorder"]
