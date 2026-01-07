"""Utilities for gating evaluation cycles by on-chain cadence."""

from __future__ import annotations

from typing import Protocol

from caster_validator.application.ports.subtensor import CommitmentRecord


class _SubtensorForGate(Protocol):
    def current_block(self) -> int:
        ...

    def last_update_block(self, uid: int) -> int | None:
        ...

    def fetch_commitment(self, uid: int) -> CommitmentRecord | None:
        ...

    def tempo(self, netuid: int) -> int:
        ...

    def get_next_epoch_start_block(
        self,
        netuid: int,
        *,
        reference_block: int | None = None,
    ) -> int:
        ...


def is_submission_window_open(
    subtensor: _SubtensorForGate,
    uid: int,
    *,
    min_blocks: int,
) -> bool:
    """Return ``True`` when the validator can safely submit new weights."""

    now = subtensor.current_block()
    last_raw = subtensor.last_update_block(uid)

    # If the validator has never submitted weights, allow the first submission immediately.
    if last_raw is None:
        return True

    last = int(last_raw)
    return (now - last) >= max(0, min_blocks)


def seconds_until_window(
    subtensor: _SubtensorForGate,
    uid: int,
    *,
    min_blocks: int,
    block_time_seconds: int,
    jitter_seconds: int = 0,
) -> float:
    """Estimate how long to wait before another submission window opens."""

    now = subtensor.current_block()
    last = subtensor.last_update_block(uid) or 0
    blocks_remaining = max(0, min_blocks - (now - last))
    jitter = (uid % (max(1, jitter_seconds + 1))) if jitter_seconds > 0 else 0
    return max(0.0, blocks_remaining * float(block_time_seconds) - float(jitter))


def submission_window_index(now_block: int, min_blocks: int) -> int:
    """Return the submission cadence bucket derived from ``min_blocks``."""

    if min_blocks <= 0:
        return 0
    return max(0, now_block) // min_blocks


def commitment_marker(uid: int, epoch: int, *, prefix: str = "caster:weights:v1") -> str:
    """Build the canonical commitment payload for ``uid`` and ``epoch``."""

    return f"{prefix}:uid={uid}:epoch={epoch}"


def _epoch_cycle_length(tempo: int) -> int:
    return max(1, tempo + 1)


def chain_epoch_window(
    subtensor: _SubtensorForGate,
    netuid: int,
    *,
    reference_block: int | None = None,
) -> tuple[int, int]:
    """Return (current_epoch_start, next_epoch_start) for ``netuid``."""

    tempo = subtensor.tempo(netuid)
    window = _epoch_cycle_length(tempo)
    next_start = subtensor.get_next_epoch_start_block(
        netuid,
        reference_block=reference_block,
    )
    current_start = next_start - window
    return current_start, next_start


def chain_epoch_index(
    *,
    at_block: int,
    netuid: int,
    tempo: int,
) -> int:
    """Return the chain epoch index at ``at_block``."""

    window = _epoch_cycle_length(tempo)
    return max(0, (at_block + netuid + 1) // window)


def current_chain_epoch_index(
    subtensor: _SubtensorForGate,
    netuid: int,
) -> int:
    """Convenience helper to read the current chain epoch index."""

    now_block = subtensor.current_block()
    tempo = subtensor.tempo(netuid)
    return chain_epoch_index(at_block=now_block, netuid=netuid, tempo=tempo)


def is_current_epoch_committed(
    subtensor: _SubtensorForGate,
    uid: int,
    *,
    netuid: int,
) -> bool:
    """Return ``True`` when the epoch commitment marker matches the latest submission."""

    record = subtensor.fetch_commitment(uid)
    if record is None:
        return False
    block = record.block
    start, end = chain_epoch_window(subtensor, netuid)
    return start <= int(block) < end


__all__ = [
    "chain_epoch_index",
    "chain_epoch_window",
    "commitment_marker",
    "current_chain_epoch_index",
    "is_current_epoch_committed",
    "is_submission_window_open",
    "submission_window_index",
    "seconds_until_window",
]
