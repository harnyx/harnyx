"""Port definitions for interacting with the platform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission
from harnyx_validator.application.ports.progress import ProviderFailureEvidence


class PlatformPort(Protocol):
    """Abstract platform client capable of champion lookup and logging."""

    def get_miner_task_batch(self, batch_id: UUID) -> MinerTaskBatchSpec:
        """Retrieve a platform-composed miner-task batch."""
        ...

    def fetch_artifact(self, batch_id: UUID, artifact_id: UUID) -> bytes:
        """Download the python agent artifact for a given candidate in the batch."""
        ...

    def get_champion_weights(self) -> ChampionWeights:
        """Return platform-computed champion weights."""
        ...

    def get_restore_metadata(self, batch_id: UUID) -> RestoreMetadata:
        """Return restore metadata for the validator-owned batch delivery."""
        ...

    def get_restore_runs_page(
        self,
        *,
        batch: MinerTaskBatchSpec,
        snapshot_received_at: datetime,
        cursor: int,
        limit: int,
    ) -> RestoreRunsPage:
        """Return one restore page converted to validator-domain submissions."""
        ...


@dataclass(frozen=True)
class ChampionWeights:
    champion_uid: int | None
    weights: dict[int, float]


@dataclass(frozen=True)
class RestoreMetadata:
    batch_id: UUID
    snapshot_received_at: datetime
    total_restore_runs: int
    page_limit: int
    provider_model_evidence: tuple[ProviderFailureEvidence, ...] = ()


@dataclass(frozen=True)
class RestoreRunsPage:
    batch_id: UUID
    snapshot_received_at: datetime
    cursor: int
    limit: int
    next_cursor: int | None
    items: tuple[MinerTaskRunSubmission, ...] = ()


__all__ = ["PlatformPort", "ChampionWeights", "RestoreMetadata", "RestoreRunsPage"]
