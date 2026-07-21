"""Miner-task champion emission policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, fsum, isfinite
from typing import TypeVar
from uuid import UUID

from harnyx_commons.miner_task_similarity import EligibleSimilarityClassification

OWNER_UID = 0
DEFAULT_MINER_PARTICIPATION_EMISSION = 0.004
TOTAL_EMISSION_FRACTION = 1.0
_TOTAL_WEIGHT_EPSILON = 1e-12
_ParticipantKey = TypeVar("_ParticipantKey")


class ParticipantEmissionTotalWeightError(ValueError):
    """Raised when participant emission would exceed total weight."""


@dataclass(frozen=True, slots=True)
class ParticipantEmissionScore:
    participant_key: str
    score: float
    artifact_id: UUID | None = None
    classification: EligibleSimilarityClassification | None = None

    def __post_init__(self) -> None:
        if (self.artifact_id is None) != (self.classification is None):
            raise ValueError("participant artifact and classification must be set together")


def compose_champion_weights(champion_uid: int | None) -> dict[int, float]:
    if champion_uid is None:
        return {}
    return {champion_uid: 1.0}


def apply_miner_emission_cap(
    weights: dict[int, float],
    batch_score: float,
    *,
    max_miner_emission_fraction: float,
) -> dict[int, float]:
    if not isfinite(batch_score) or batch_score < 0.0 or batch_score > 1.0:
        raise ValueError("miner task batch score must be between 0.0 and 1.0")
    if (
        not isfinite(max_miner_emission_fraction)
        or max_miner_emission_fraction < 0.0
        or max_miner_emission_fraction > 1.0
    ):
        raise ValueError("max miner emission fraction must be between 0.0 and 1.0")

    base = {uid: weight for uid, weight in weights.items() if uid != OWNER_UID}
    if not base:
        raise ValueError("miner weights are empty")
    total = float(sum(base.values()))
    if total <= 0.0:
        raise ValueError("miner weights must have positive miner total")

    miner_fraction = batch_score * max_miner_emission_fraction
    scaled: dict[int, float] = {uid: float(weight) / total * miner_fraction for uid, weight in base.items()}
    scaled[OWNER_UID] = 1.0 - miner_fraction
    return scaled


def participant_emission_fraction(
    participant_count: int,
    *,
    miner_participation_emission: float,
) -> float:
    if participant_count < 0:
        raise ValueError("participant count must be non-negative")
    if (
        not isfinite(miner_participation_emission)
        or miner_participation_emission < 0.0
        or miner_participation_emission > 1.0
    ):
        raise ValueError("miner participation emission must be between 0.0 and 1.0")
    if miner_participation_emission == 0.0 or participant_count == 0:
        return 0.0
    payable_count = min(
        participant_count,
        floor((TOTAL_EMISSION_FRACTION + _TOTAL_WEIGHT_EPSILON) / miner_participation_emission),
    )
    return min(TOTAL_EMISSION_FRACTION, payable_count * miner_participation_emission)


def compose_participant_emission_weights(
    registered_participant_uids: tuple[int, ...],
    *,
    miner_participation_emission: float = DEFAULT_MINER_PARTICIPATION_EMISSION,
) -> dict[int, float]:
    _validate_miner_participation_emission(miner_participation_emission)
    distinct_uids = tuple(dict.fromkeys(uid for uid in registered_participant_uids if uid != OWNER_UID))
    return _capped_allocations_in_order(tuple((uid, miner_participation_emission) for uid in distinct_uids))


def compose_flat_participant_emission_allocations(
    participant_keys: Sequence[str],
    *,
    miner_participation_emission: float = DEFAULT_MINER_PARTICIPATION_EMISSION,
) -> dict[str, float]:
    _validate_miner_participation_emission(miner_participation_emission)
    distinct_keys: dict[str, None] = {}
    for participant_key in participant_keys:
        if not participant_key:
            raise ValueError("participant key must be non-empty")
        distinct_keys.setdefault(participant_key, None)
    return _capped_allocations_in_order(
        tuple((participant_key, miner_participation_emission) for participant_key in distinct_keys)
    )


def compose_tiered_participant_emission_allocations(
    participant_scores: Sequence[ParticipantEmissionScore],
    *,
    miner_participation_emission: float = DEFAULT_MINER_PARTICIPATION_EMISSION,
) -> dict[str, float]:
    _validate_miner_participation_emission(miner_participation_emission)

    ordered = tuple(
        sorted(
            select_participant_emission_scores(participant_scores),
            key=lambda participant: (-participant.score, participant.participant_key),
        )
    )
    if not ordered:
        return {}

    top_floor = _score_floor(ordered, fraction=0.10)
    middle_floor = _score_floor(ordered, fraction=0.50)
    ordered_allocations: list[tuple[str, float]] = []
    for participant in ordered:
        participant_key = participant.participant_key
        score = participant.score
        if score <= 0.0:
            continue
        if score >= top_floor:
            multiplier = 2.0
        elif score >= middle_floor:
            multiplier = 1.0
        else:
            continue
        ordered_allocations.append(
            (
                participant_key,
                miner_participation_emission
                * multiplier
                * participant_emission_novelty_multiplier(participant.classification),
            )
        )

    return _capped_allocations_in_order(tuple(ordered_allocations))


def compose_emission_weights(*components: dict[int, float]) -> dict[int, float]:
    weights: dict[int, float] = {}
    for component in components:
        for uid, weight in component.items():
            if uid == OWNER_UID:
                continue
            weights[uid] = weights.get(uid, 0.0) + weight

    miner_fraction = fsum(weights.values())
    if _exceeds_total_emission(miner_fraction):
        raise ValueError("emission exceeds total weight")
    # Assigning miner emission to owner UID burns it; owner is not a miner payout recipient.
    weights[OWNER_UID] = TOTAL_EMISSION_FRACTION - min(TOTAL_EMISSION_FRACTION, miner_fraction)
    return weights


def select_participant_emission_scores(
    participant_scores: Sequence[ParticipantEmissionScore],
) -> tuple[ParticipantEmissionScore, ...]:
    selected: dict[str, ParticipantEmissionScore] = {}
    for participant in participant_scores:
        if not participant.participant_key:
            raise ValueError("participant key must be non-empty")
        if not isfinite(participant.score) or participant.score < 0.0 or participant.score > 1.0:
            raise ValueError("participant score must be between 0.0 and 1.0")
        existing = selected.get(participant.participant_key)
        if existing is None or _participant_selection_key(participant) > _participant_selection_key(existing):
            selected[participant.participant_key] = participant
    return tuple(selected[participant_key] for participant_key in sorted(selected))


def participant_emission_novelty_multiplier(
    classification: EligibleSimilarityClassification | None,
) -> float:
    if classification is None or classification == "novel":
        return 1.0
    if classification == "near_duplicate":
        return 0.5
    raise ValueError(f"unsupported participant similarity classification: {classification}")


def _participant_selection_key(
    participant: ParticipantEmissionScore,
) -> tuple[float, int, str]:
    novelty_rank = {
        None: 0,
        "near_duplicate": 1,
        "novel": 2,
    }[participant.classification]
    artifact_key = "" if participant.artifact_id is None else str(participant.artifact_id)
    return participant.score, novelty_rank, artifact_key


def owner_fallback_weights() -> dict[int, float]:
    return {OWNER_UID: 1.0}


def _score_floor(
    ordered_scores: tuple[ParticipantEmissionScore, ...],
    *,
    fraction: float,
) -> float:
    cutoff_count = ceil(len(ordered_scores) * fraction)
    index = max(0, cutoff_count - 1)
    return ordered_scores[index].score


def _validate_miner_participation_emission(miner_participation_emission: float) -> None:
    if (
        not isfinite(miner_participation_emission)
        or miner_participation_emission < 0.0
        or miner_participation_emission > 1.0
    ):
        raise ValueError("miner participation emission must be between 0.0 and 1.0")


def _capped_allocations_in_order(
    weighted_participants: Sequence[tuple[_ParticipantKey, float]],
) -> dict[_ParticipantKey, float]:
    allocations: dict[_ParticipantKey, float] = {}
    miner_fraction = 0.0
    for participant_key, participant_emission in weighted_participants:
        if participant_emission <= 0.0:
            continue
        next_fraction = miner_fraction + participant_emission
        if _exceeds_total_emission(next_fraction):
            break
        allocations[participant_key] = participant_emission
        miner_fraction = min(TOTAL_EMISSION_FRACTION, next_fraction)
    return allocations


def _exceeds_total_emission(miner_fraction: float) -> bool:
    return miner_fraction > TOTAL_EMISSION_FRACTION + _TOTAL_WEIGHT_EPSILON


__all__ = [
    "DEFAULT_MINER_PARTICIPATION_EMISSION",
    "OWNER_UID",
    "ParticipantEmissionScore",
    "ParticipantEmissionTotalWeightError",
    "apply_miner_emission_cap",
    "compose_champion_weights",
    "compose_emission_weights",
    "compose_flat_participant_emission_allocations",
    "compose_participant_emission_weights",
    "compose_tiered_participant_emission_allocations",
    "owner_fallback_weights",
    "participant_emission_fraction",
    "participant_emission_novelty_multiplier",
    "select_participant_emission_scores",
]
