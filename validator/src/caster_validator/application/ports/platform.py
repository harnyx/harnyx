"""Port definitions for interacting with the external evaluation platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from caster_validator.application.dto.evaluation import EvaluationBatchSpec


class PlatformPort(Protocol):
    """Abstract platform client capable of champion lookup and logging."""

    def get_evaluation_batch(self, run_id: UUID) -> EvaluationBatchSpec:
        """Retrieve a platform-composed evaluation batch."""

    def fetch_artifact(self, run_id: UUID, uid: int) -> bytes:
        """Download the python agent artifact for a given uid in the batch."""

    def get_champion_weights(self) -> ChampionWeights:
        """Return platform-computed champion weights and top3."""


@dataclass(frozen=True)
class ChampionWeights:
    final_top: tuple[int | None, int | None, int | None]
    weights: dict[int, float]


__all__ = ["PlatformPort", "ChampionWeights"]
