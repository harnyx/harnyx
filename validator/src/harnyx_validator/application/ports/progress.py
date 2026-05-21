"""Port for recording miner-task batch progress snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict
from uuid import UUID

from harnyx_commons.miner_task_failure_policy import ProviderFailureEvidence
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission


class RunProgressSummary(TypedDict):
    batch_id: UUID
    total: int
    completed: int
    remaining: int
    latest_sequence: int
    provider_evidence: tuple[ProviderFailureEvidence, ...]


class SequencedRun(TypedDict):
    sequence: int
    submission: MinerTaskRunSubmission


class RunProgressPage(TypedDict):
    batch_id: UUID
    after_sequence: int
    limit: int
    latest_sequence: int
    next_after_sequence: int
    has_more: bool
    items: tuple[SequencedRun, ...]


class ProgressRecorder(Protocol):
    def register(self, batch: MinerTaskBatchSpec) -> None:
        ...

    def record(self, result: MinerTaskRunSubmission) -> None:
        ...

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        submissions: Sequence[MinerTaskRunSubmission],
        provider_evidence: Sequence[ProviderFailureEvidence] = (),
    ) -> None:
        ...

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        ...

    def summary(self, batch_id: UUID) -> RunProgressSummary:
        ...

    def completed_run_page(
        self,
        batch_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> RunProgressPage:
        ...

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        ...

    def record_provider_call(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
    ) -> None:
        ...

    def record_provider_failure(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        ...

    def consume_provider_failures(self, session_id: UUID) -> tuple[ProviderFailureEvidence, ...]:
        ...

    def clear_task_session(self, session_id: UUID) -> None:
        ...


__all__ = [
    "ProgressRecorder",
    "ProviderFailureEvidence",
    "RunProgressPage",
    "RunProgressSummary",
    "SequencedRun",
]
