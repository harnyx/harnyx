"""Port definitions for interacting with the Bittensor subtensor network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CommitmentRecord:
    """Miner commitment metadata stored on-chain."""

    block: int
    data: str


@dataclass(frozen=True, slots=True)
class MetagraphSnapshot:
    """Lightweight snapshot of the active subnet metagraph."""

    uids: Sequence[int]
    hotkeys: Sequence[str]


@dataclass(frozen=True, slots=True)
class ValidatorNodeInfo:
    """Information about the validator's on-chain identity."""

    uid: int
    version_key: int | None


class SubtensorClientPort(Protocol):
    """Abstract client responsible for subtensor interactions."""

    def connect(self) -> None:
        """Ensure the underlying client is ready for use (idempotent)."""

    def close(self) -> None:
        """Release any held resources."""

    def fetch_metagraph(self) -> MetagraphSnapshot:
        """Return the active subnet metagraph."""

    def fetch_commitment(self, uid: int) -> CommitmentRecord | None:
        """Return the latest commitment for ``uid`` when available."""

    def publish_commitment(self, data: str, *, blocks_until_reveal: int = 1) -> CommitmentRecord:
        """Publish a new commitment for the validator."""

    def current_block(self) -> int:
        """Return the current chain block height."""

    def last_update_block(self, uid: int) -> int | None:
        """Return the block height of the most recent weight update for ``uid``."""

    def validator_info(self) -> ValidatorNodeInfo:
        """Return validator node metadata (UID, version key, etc.)."""

    def submit_weights(self, weights: Mapping[int, float]) -> str:
        """Submit normalized weights and return the transaction reference/hash."""

    def fetch_weight(self, uid: int) -> float:
        """Return the current on-chain weight for ``uid`` (0.0 when absent)."""

    def tempo(self, netuid: int) -> int:
        """Return the epoch tempo (blocks per epoch) for ``netuid``."""

    def get_next_epoch_start_block(
        self,
        netuid: int,
        *,
        reference_block: int | None = None,
    ) -> int:
        """Return the block height where the next epoch starts for ``netuid``."""


__all__ = [
    "CommitmentRecord",
    "MetagraphSnapshot",
    "SubtensorClientPort",
    "ValidatorNodeInfo",
]
