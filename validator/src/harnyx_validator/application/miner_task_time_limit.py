"""Execution-time policy for Platform-assigned miner tasks."""

from __future__ import annotations

from hashlib import sha256
from uuid import UUID

from harnyx_commons.domain.miner_task import MinerTask

_FAST_LIMIT_NAMESPACE = b"harnyx-fast-task-execution-limit-v1\0"
_FAST_LIMIT_BUCKETS_SECONDS = (150.0, 180.0, 210.0)
_ORDINARY_LIMIT_NAMESPACE = b"harnyx-ordinary-task-execution-limit-v1\0"
_ORDINARY_LIMIT_BUCKETS_SECONDS = (180.0, 210.0, 240.0)


def assigned_miner_task_execution_time_limit_seconds(*, batch_id: UUID, task: MinerTask) -> float:
    """Return the stable execution limit for one assigned batch-task occurrence."""
    namespace, buckets = (
        (_FAST_LIMIT_NAMESPACE, _FAST_LIMIT_BUCKETS_SECONDS)
        if task.query.fast
        else (_ORDINARY_LIMIT_NAMESPACE, _ORDINARY_LIMIT_BUCKETS_SECONDS)
    )

    digest = sha256(namespace + batch_id.bytes + task.task_id.bytes).digest()
    bucket_index = int.from_bytes(digest, byteorder="big", signed=False) % len(buckets)
    return buckets[bucket_index]


__all__ = ["assigned_miner_task_execution_time_limit_seconds"]
