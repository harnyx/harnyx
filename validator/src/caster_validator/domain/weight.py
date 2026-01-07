"""Weight submission record."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class WeightSubmission:
    """Normalized weight vector pushed on-chain."""

    run_id: UUID
    submitted_at: datetime
    weights: Mapping[int, float]
    tx_hash: str

    def __post_init__(self) -> None:
        if not self.tx_hash.strip():
            raise ValueError("tx_hash must not be empty")
        for uid, weight in self.weights.items():
            if weight <= 0.0:
                raise ValueError(f"weights[{uid}] must be positive")
        total = self.total_weight
        if not 0.99 <= total <= 1.01:
            raise ValueError("weight vector must be normalized to sum to 1.0 ± 0.01")

    @property
    def total_weight(self) -> float:
        return float(sum(self.weights.values()))


__all__ = ["WeightSubmission"]

