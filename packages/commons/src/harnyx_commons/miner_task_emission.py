"""Miner-task champion emission policies."""

from __future__ import annotations

from math import isfinite

OWNER_UID = 0
MAX_MINER_EMISSION_FRACTION = 0.0


def compose_champion_weights(champion_uid: int | None) -> dict[int, float]:
    if champion_uid is None:
        return {}
    return {champion_uid: 1.0}


def apply_miner_emission_cap(weights: dict[int, float], batch_score: float) -> dict[int, float]:
    if not isfinite(batch_score) or batch_score < 0.0 or batch_score > 1.0:
        raise ValueError("miner task batch score must be between 0.0 and 1.0")

    base = {uid: weight for uid, weight in weights.items() if uid != OWNER_UID}
    if not base:
        raise ValueError("miner weights are empty")
    total = float(sum(base.values()))
    if total <= 0.0:
        raise ValueError("miner weights must have positive miner total")

    miner_fraction = batch_score * MAX_MINER_EMISSION_FRACTION
    scaled: dict[int, float] = {
        uid: float(weight) / total * miner_fraction for uid, weight in base.items()
    }
    scaled[OWNER_UID] = 1.0 - miner_fraction
    return scaled


def owner_fallback_weights() -> dict[int, float]:
    return {OWNER_UID: 1.0}


__all__ = [
    "MAX_MINER_EMISSION_FRACTION",
    "OWNER_UID",
    "apply_miner_emission_cap",
    "compose_champion_weights",
    "owner_fallback_weights",
]
