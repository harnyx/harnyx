"""DTOs related to weight updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission, TokenUsageSummary


@dataclass(frozen=True)
class WeightUpdateRequest:
    """Payload describing a batch of recorded miner-task runs."""

    run_id: UUID
    runs: Sequence[MinerTaskRunSubmission]


@dataclass(frozen=True)
class WeightUpdateResult:
    """Normalized weight vector derived from miner-task runs."""

    run_id: UUID
    weights: Mapping[int, float]
    usage_by_uid: Mapping[int, TokenUsageSummary]
    ordering: tuple[int, ...]
