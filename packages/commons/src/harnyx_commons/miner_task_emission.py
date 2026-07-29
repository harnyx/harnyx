"""Miner-task champion emission policies."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor, fsum, isfinite
from typing import Generic, Literal, TypeVar
from uuid import UUID

from harnyx_commons.miner_task_similarity import EligibleSimilarityClassification

OWNER_UID = 0
DEFAULT_MINER_PARTICIPATION_EMISSION = 0.004
DEFAULT_SUCCESSFUL_MINER_PARTICIPATION_EMISSION = 0.008
TOTAL_EMISSION_FRACTION = 1.0
_TOTAL_WEIGHT_EPSILON = 1e-12
_ParticipantKey = TypeVar("_ParticipantKey")
NoveltyDistributionWeight = Literal[1, 3, 5]
ParticipationStageMultiplier = Literal[1, 2, 5]
NoveltyMultiplier = Literal[1, 3, 5]
ParticipantDistributionWeight = Literal[1, 2, 3, 5, 6, 10, 15, 25]
_ParticipantTierWeight = Literal[1, 2]


class ParticipantEmissionTotalWeightError(ValueError):
    """Raised when participant emission would exceed total weight."""


@dataclass(frozen=True, slots=True)
class ParticipantEmissionScore:
    participant_key: str
    score: float
    artifact_id: UUID | None = None
    classification: EligibleSimilarityClassification | None = None

    def __post_init__(self) -> None:
        if self.classification is not None and self.artifact_id is None:
            raise ValueError("participant classification requires an artifact")


@dataclass(frozen=True, slots=True)
class ParticipantEmissionArtifactWeight:
    participation_stage_multiplier: ParticipationStageMultiplier
    novelty_multiplier: NoveltyMultiplier
    participant_distribution_weight: ParticipantDistributionWeight


@dataclass(frozen=True, slots=True)
class PrioritizedEmissionComposition:
    weights: dict[int, float]
    accepted_main_additions: dict[int, float]
    dropped_main_additions: dict[int, float]
    accepted_general_participation: dict[int, float]
    dropped_general_participation: dict[int, float]


@dataclass(frozen=True, slots=True)
class PrioritizedEmissionAdmission(Generic[_ParticipantKey]):
    accepted_main_additions: dict[_ParticipantKey, float]
    dropped_main_additions: dict[_ParticipantKey, float]
    accepted_general_participation: dict[_ParticipantKey, float]
    dropped_general_participation: dict[_ParticipantKey, float]


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
    base = {uid: weight for uid, weight in weights.items() if uid != OWNER_UID}
    if not base:
        raise ValueError("miner weights are empty")
    total = float(sum(base.values()))
    if total <= 0.0:
        raise ValueError("miner weights must have positive miner total")

    miner_fraction = champion_emission_fraction(
        batch_score,
        max_miner_emission_fraction=max_miner_emission_fraction,
    )
    scaled: dict[int, float] = {uid: float(weight) / total * miner_fraction for uid, weight in base.items()}
    scaled[OWNER_UID] = 1.0 - miner_fraction
    return scaled


def champion_emission_fraction(
    batch_score: float,
    *,
    max_miner_emission_fraction: float,
) -> float:
    if not isfinite(batch_score) or batch_score < 0.0 or batch_score > 1.0:
        raise ValueError("miner task batch score must be between 0.0 and 1.0")
    if (
        not isfinite(max_miner_emission_fraction)
        or max_miner_emission_fraction < 0.0
        or max_miner_emission_fraction > 1.0
    ):
        raise ValueError("max miner emission fraction must be between 0.0 and 1.0")
    return batch_score * max_miner_emission_fraction


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


def compose_base_participant_emission_allocations(
    participant_keys: Sequence[str],
    *,
    miner_participation_emission: float = DEFAULT_MINER_PARTICIPATION_EMISSION,
) -> dict[str, float]:
    """Return one uncapped base allocation per distinct participant in input order."""

    _validate_miner_participation_emission(miner_participation_emission)
    distinct_keys: dict[str, None] = {}
    for participant_key in participant_keys:
        if not participant_key:
            raise ValueError("participant key must be non-empty")
        distinct_keys.setdefault(participant_key, None)
    return {participant_key: miner_participation_emission for participant_key in distinct_keys}


def compose_tiered_participant_emission_allocations(
    participant_scores: Sequence[ParticipantEmissionScore],
    *,
    miner_participation_emission: float = DEFAULT_SUCCESSFUL_MINER_PARTICIPATION_EMISSION,
) -> dict[str, float]:
    _validate_miner_participation_emission(miner_participation_emission)

    ordered_allocations = tuple(
        (
            participant.participant_key,
            miner_participation_emission
            * tier_weight
            * _participant_fixed_reward_ratio(participant.classification),
        )
        for participant, tier_weight in _tiered_participant_weights(participant_scores)
    )
    return _capped_allocations_in_order(ordered_allocations)


def compose_novelty_distribution_weights(
    participant_scores: Sequence[ParticipantEmissionScore],
    *,
    main_participant_artifact_ids: Collection[UUID] = (),
) -> dict[str, NoveltyDistributionWeight]:
    """Return exclusive main, top-10% or top-50% weights for novel participants."""

    main_artifact_ids = set(main_participant_artifact_ids)
    selected = select_participant_emission_scores(participant_scores)
    weights: dict[str, NoveltyDistributionWeight] = {
        participant.participant_key: 3 if tier_weight == 2 else 1
        for participant, tier_weight in _tiered_participant_weights(participant_scores)
        if participant.classification == "novel"
    }
    for participant in selected:
        if participant.artifact_id in main_artifact_ids and participant.classification == "novel":
            weights[participant.participant_key] = 5
    return weights


def compose_novelty_emission_allocations(
    novelty_distribution_weights: Mapping[_ParticipantKey, float],
    *,
    remaining_emission_fraction: float,
) -> dict[_ParticipantKey, float]:
    """Divide the post-assignment emission remainder in proportion to novelty weights."""

    return _compose_proportional_emission_allocations(
        novelty_distribution_weights,
        emission_fraction=remaining_emission_fraction,
    )


def compose_artifact_participant_distribution_weights(
    participant_scores: Sequence[ParticipantEmissionScore],
    *,
    main_participant_artifact_ids: Collection[UUID] = (),
) -> dict[UUID, ParticipantEmissionArtifactWeight]:
    """Return v7 participant distribution weights keyed by eligible artifact."""

    main_artifact_ids = set(main_participant_artifact_ids)
    selected_by_artifact: dict[UUID, ParticipantEmissionScore] = {}
    for participant in participant_scores:
        if not participant.participant_key:
            raise ValueError("participant key must be non-empty")
        if participant.artifact_id is None:
            raise ValueError("artifact-weighted participant emission requires an artifact")
        if participant.classification is None:
            raise ValueError("artifact-weighted participant emission requires a classification")
        if not isfinite(participant.score) or participant.score < 0.0 or participant.score > 1.0:
            raise ValueError("participant score must be between 0.0 and 1.0")
        if participant.artifact_id in selected_by_artifact:
            raise ValueError("participant artifact ids must be unique")
        selected_by_artifact[participant.artifact_id] = participant

    stage_multipliers = _artifact_participation_stage_multipliers(
        tuple(selected_by_artifact.values()),
        main_participant_artifact_ids=main_artifact_ids,
    )
    weights: dict[UUID, ParticipantEmissionArtifactWeight] = {}
    for artifact_id, stage_multiplier in stage_multipliers.items():
        participant = selected_by_artifact[artifact_id]
        classification = participant.classification
        if classification is None:
            raise ValueError("artifact-weighted participant emission requires a classification")
        novelty_multiplier = _novelty_multiplier(classification)
        weights[artifact_id] = ParticipantEmissionArtifactWeight(
            participation_stage_multiplier=stage_multiplier,
            novelty_multiplier=novelty_multiplier,
            participant_distribution_weight=stage_multiplier * novelty_multiplier,
        )
    return weights


def compose_weighted_participant_emission_allocations(
    participant_distribution_weights: Mapping[_ParticipantKey, float],
    *,
    emission_fraction: float,
) -> dict[_ParticipantKey, float]:
    """Divide an emission pool in proportion to participant distribution weights."""

    return _compose_proportional_emission_allocations(
        participant_distribution_weights,
        emission_fraction=emission_fraction,
    )


def compose_equal_participant_emission_allocations(
    participant_keys: Sequence[_ParticipantKey],
    *,
    remaining_emission_fraction: float,
) -> dict[_ParticipantKey, float]:
    """Divide the available emission equally among distinct participants."""

    distinct_keys: dict[_ParticipantKey, float] = {}
    for participant_key in participant_keys:
        if not participant_key:
            raise ValueError("participant key must be non-empty")
        distinct_keys.setdefault(participant_key, 1.0)
    return _compose_proportional_emission_allocations(
        distinct_keys,
        emission_fraction=remaining_emission_fraction,
    )


def _compose_proportional_emission_allocations(
    distribution_weights: Mapping[_ParticipantKey, float],
    *,
    emission_fraction: float,
) -> dict[_ParticipantKey, float]:
    if (
        not isfinite(emission_fraction)
        or emission_fraction < 0.0
        or _exceeds_total_emission(emission_fraction)
    ):
        raise ValueError("emission fraction must be between 0.0 and 1.0")
    for participant_key, weight in distribution_weights.items():
        if not participant_key:
            raise ValueError("participant key must be non-empty")
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("distribution weight must be positive")
    total_weight = fsum(distribution_weights.values())
    if emission_fraction == 0.0 or total_weight == 0.0:
        return {}

    allocations: dict[_ParticipantKey, float] = {}
    allocated = 0.0
    final_index = len(distribution_weights) - 1
    for index, (participant_key, weight) in enumerate(distribution_weights.items()):
        if index == final_index:
            allocation = emission_fraction - allocated
        else:
            allocation = emission_fraction * weight / total_weight
            allocated += allocation
        allocations[participant_key] = allocation
    return allocations


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


def compose_prioritized_emission(
    champion: dict[int, float],
    main_additions: dict[int, float],
    general_participation: dict[int, float],
) -> PrioritizedEmissionComposition:
    """Fill champion, main and general emission capacity in policy order."""

    weights = {uid: weight for uid, weight in champion.items() if uid != OWNER_UID and weight > 0.0}
    used = fsum(weights.values())
    if _exceeds_total_emission(used):
        raise ValueError("champion emission exceeds total weight")
    admission = admit_prioritized_emission(
        {uid: weight for uid, weight in main_additions.items() if uid != OWNER_UID},
        {uid: weight for uid, weight in general_participation.items() if uid != OWNER_UID},
        reserved_fraction=used,
    )
    for component in (
        admission.accepted_main_additions,
        admission.accepted_general_participation,
    ):
        for uid, weight in component.items():
            weights[uid] = weights.get(uid, 0.0) + weight
    emitted_fraction = fsum(weights.values())
    weights[OWNER_UID] = TOTAL_EMISSION_FRACTION - min(TOTAL_EMISSION_FRACTION, emitted_fraction)
    return PrioritizedEmissionComposition(
        weights=weights,
        accepted_main_additions=admission.accepted_main_additions,
        dropped_main_additions=admission.dropped_main_additions,
        accepted_general_participation=admission.accepted_general_participation,
        dropped_general_participation=admission.dropped_general_participation,
    )


def admit_prioritized_emission(
    main_additions: dict[_ParticipantKey, float],
    general_participation: dict[_ParticipantKey, float],
    *,
    reserved_fraction: float,
) -> PrioritizedEmissionAdmission[_ParticipantKey]:
    """Reserve capacity in champion, main and general policy order before UID projection."""

    if not isfinite(reserved_fraction) or reserved_fraction < 0.0 or _exceeds_total_emission(
        reserved_fraction
    ):
        raise ValueError("reserved emission fraction must be between 0.0 and 1.0")
    accepted_main, dropped_main, used = _admit_prioritized_component(
        main_additions,
        used=reserved_fraction,
    )
    accepted_general, dropped_general, _ = _admit_prioritized_component(
        general_participation,
        used=used,
    )
    return PrioritizedEmissionAdmission(
        accepted_main_additions=accepted_main,
        dropped_main_additions=dropped_main,
        accepted_general_participation=accepted_general,
        dropped_general_participation=dropped_general,
    )


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


def _participant_fixed_reward_ratio(
    classification: EligibleSimilarityClassification | None,
) -> float:
    if classification is None:
        return 1.0
    if classification in {"near_duplicate", "novel"}:
        return 0.25
    raise ValueError(f"unsupported participant similarity classification: {classification}")


def _participant_selection_key(
    participant: ParticipantEmissionScore,
) -> tuple[float, int, str]:
    novelty_rank = {
        None: 0,
        "near_duplicate": 1,
        "notable_change": 2,
        "novel": 3,
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


def _tiered_participant_weights(
    participant_scores: Sequence[ParticipantEmissionScore],
) -> tuple[tuple[ParticipantEmissionScore, _ParticipantTierWeight], ...]:
    ordered = tuple(
        sorted(
            select_participant_emission_scores(participant_scores),
            key=lambda participant: (-participant.score, participant.participant_key),
        )
    )
    if not ordered:
        return ()

    top_floor = _score_floor(ordered, fraction=0.10)
    middle_floor = _score_floor(ordered, fraction=0.50)
    weighted: list[tuple[ParticipantEmissionScore, _ParticipantTierWeight]] = []
    for participant in ordered:
        if participant.score <= 0.0:
            continue
        if participant.score >= top_floor:
            tier_weight = 2
        elif participant.score >= middle_floor:
            tier_weight = 1
        else:
            continue
        weighted.append((participant, tier_weight))
    return tuple(weighted)


def _artifact_participation_stage_multipliers(
    participant_scores: Sequence[ParticipantEmissionScore],
    *,
    main_participant_artifact_ids: set[UUID],
) -> dict[UUID, ParticipationStageMultiplier]:
    ordered = tuple(
        sorted(
            participant_scores,
            key=lambda participant: (
                -participant.score,
                str(participant.artifact_id),
            ),
        )
    )
    if not ordered:
        return {}

    top_floor = _score_floor(ordered, fraction=0.10)
    middle_floor = _score_floor(ordered, fraction=0.50)
    multipliers: dict[UUID, ParticipationStageMultiplier] = {}
    for participant in ordered:
        artifact_id = participant.artifact_id
        if artifact_id is None:
            raise ValueError("artifact-weighted participant emission requires an artifact")
        if artifact_id in main_participant_artifact_ids:
            multipliers[artifact_id] = 5
        elif participant.score <= 0.0:
            continue
        elif participant.score >= top_floor:
            multipliers[artifact_id] = 2
        elif participant.score >= middle_floor:
            multipliers[artifact_id] = 1
    return multipliers


def _novelty_multiplier(
    classification: EligibleSimilarityClassification,
) -> NoveltyMultiplier:
    if classification == "near_duplicate":
        return 1
    if classification == "notable_change":
        return 3
    return 5


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


def _admit_prioritized_component(
    component: dict[_ParticipantKey, float],
    *,
    used: float,
) -> tuple[dict[_ParticipantKey, float], dict[_ParticipantKey, float], float]:
    accepted: dict[_ParticipantKey, float] = {}
    dropped: dict[_ParticipantKey, float] = {}
    capacity_exhausted = False
    for participant_key, weight in component.items():
        if weight <= 0.0:
            continue
        if capacity_exhausted or _exceeds_total_emission(used + weight):
            dropped[participant_key] = weight
            capacity_exhausted = True
            continue
        accepted[participant_key] = weight
        used = min(TOTAL_EMISSION_FRACTION, used + weight)
    return accepted, dropped, used


__all__ = [
    "DEFAULT_MINER_PARTICIPATION_EMISSION",
    "DEFAULT_SUCCESSFUL_MINER_PARTICIPATION_EMISSION",
    "NoveltyDistributionWeight",
    "NoveltyMultiplier",
    "OWNER_UID",
    "ParticipantEmissionArtifactWeight",
    "ParticipantDistributionWeight",
    "ParticipantEmissionScore",
    "ParticipantEmissionTotalWeightError",
    "ParticipationStageMultiplier",
    "PrioritizedEmissionComposition",
    "PrioritizedEmissionAdmission",
    "apply_miner_emission_cap",
    "admit_prioritized_emission",
    "champion_emission_fraction",
    "compose_artifact_participant_distribution_weights",
    "compose_champion_weights",
    "compose_base_participant_emission_allocations",
    "compose_emission_weights",
    "compose_equal_participant_emission_allocations",
    "compose_flat_participant_emission_allocations",
    "compose_novelty_distribution_weights",
    "compose_novelty_emission_allocations",
    "compose_participant_emission_weights",
    "compose_prioritized_emission",
    "compose_tiered_participant_emission_allocations",
    "compose_weighted_participant_emission_allocations",
    "owner_fallback_weights",
    "participant_emission_fraction",
    "select_participant_emission_scores",
]
