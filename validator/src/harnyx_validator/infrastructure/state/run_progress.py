"""In-memory tracker for per-batch miner-task progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
from uuid import UUID

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission


class RunProgressSnapshot(TypedDict):
    batch_id: UUID
    total: int
    completed: int
    remaining: int
    tasks: tuple[MinerTask, ...]
    miner_task_runs: tuple[MinerTaskRunSubmission, ...]


@dataclass(slots=True)
class InMemoryRunProgress:
    batches_by_id: dict[UUID, MinerTaskBatchSpec] = field(default_factory=dict)
    expected_by_batch: dict[UUID, int] = field(default_factory=dict)
    tasks_by_batch: dict[UUID, tuple[MinerTask, ...]] = field(default_factory=dict)
    results_by_batch: dict[
        UUID,
        dict[tuple[UUID, UUID], MinerTaskRunSubmission],
    ] = field(default_factory=dict)

    def register(self, batch: MinerTaskBatchSpec) -> None:
        existing = self.batches_by_id.get(batch.batch_id)
        if existing is not None:
            if existing != batch:
                raise RuntimeError("batch_id already exists with different contents")
            return

        self.batches_by_id[batch.batch_id] = batch
        self.expected_by_batch[batch.batch_id] = len(batch.tasks) * len(batch.artifacts)
        self.tasks_by_batch[batch.batch_id] = batch.tasks

    def record(self, result: MinerTaskRunSubmission) -> None:
        pair = (result.run.artifact_id, result.run.task_id)
        bucket = self.results_by_batch.setdefault(result.batch_id, {})
        existing = bucket.get(pair)
        if existing is not None:
            if existing != result:
                raise RuntimeError(
                    "batch already recorded a different result for artifact/task pair"
                )
            return
        bucket[pair] = result

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        bucket = self.results_by_batch.get(batch_id, {})
        return frozenset(bucket)

    def snapshot(self, batch_id: UUID) -> RunProgressSnapshot:
        results = tuple(self.results_by_batch.get(batch_id, {}).values())
        total = int(self.expected_by_batch.get(batch_id, 0))
        completed = len(results)
        remaining = max(0, total - completed)
        return {
            "batch_id": batch_id,
            "total": total,
            "completed": completed,
            "remaining": remaining,
            "tasks": self.tasks_by_batch.get(batch_id, ()),
            "miner_task_runs": results,
        }


__all__ = ["InMemoryRunProgress", "RunProgressSnapshot"]
