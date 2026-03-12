"""Port definitions for interacting with the platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from caster_validator.application.dto.evaluation import MinerTaskBatchSpec


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


@dataclass(frozen=True)
class ChampionWeights:
    champion_uid: int | None
    weights: dict[int, float]


__all__ = ["PlatformPort", "ChampionWeights"]
