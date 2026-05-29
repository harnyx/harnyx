"""Miner-task participant and champion selection policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from math import isfinite
from typing import Protocol, TypeVar
from uuid import UUID

from harnyx_commons.miner_task_emission import compose_champion_weights
from harnyx_commons.miner_task_ranking import (
    ArtifactAggregateBundle,
    ArtifactRankingRow,
    RankingCascade,
    RankingCascadeTrace,
    aggregate_ranking_rows,
    ordered_challengers,
)

REQUIRED_SUCCESSFUL_VALIDATOR_COUNT = 1


class ValidatorRunInput(Protocol):
    validator_id: UUID


_RunT = TypeVar("_RunT", bound=ValidatorRunInput)


@dataclass(frozen=True, slots=True)
class SubmittedArtifactInput:
    uid: int
    artifact_id: UUID
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class CurrentChampionInput:
    uid: int
    artifact_id: UUID


@dataclass(frozen=True, slots=True)
class ChampionArtifactInput:
    artifact_id: UUID
    uid: int


@dataclass(frozen=True, slots=True)
class ChampionRunInput:
    validator_id: UUID
    artifact_id: UUID
    task_id: UUID
    score: float
    total_cost_usd: float
    elapsed_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ChampionSelection:
    champion_uid: int | None
    weights: dict[int, float]
    score: float = 1.0
    champion_artifact_id: UUID | None = None
    artifact_scores: dict[UUID, float] = field(default_factory=dict, compare=False)
    incumbent_artifact_id: UUID | None = field(default=None, compare=False)
    ranking_trace: RankingCascadeTrace | None = field(default=None, compare=False)
    similarity_fallback_artifact_ids: tuple[UUID, ...] = field(default=(), compare=False)


def selection_from_stored_champion_weights(
    *,
    final_top: Sequence[int],
    weights: Mapping[str, float],
    champion_artifact_id: UUID | None,
) -> ChampionSelection:
    champion_uid = int(final_top[0]) if final_top else None
    if champion_uid is None:
        return ChampionSelection(
            champion_uid=None,
            weights={},
            score=0.0,
            champion_artifact_id=champion_artifact_id,
        )
    return ChampionSelection(
        champion_uid=champion_uid,
        weights=compose_champion_weights(champion_uid),
        score=_stored_champion_score(champion_uid=champion_uid, weights=weights),
        champion_artifact_id=champion_artifact_id,
    )


def select_batch_artifacts(
    *,
    latest_by_uid: Mapping[int, SubmittedArtifactInput],
    previous_completed_cutoff: datetime,
    current_champion: CurrentChampionInput | None,
    incumbent: SubmittedArtifactInput | None,
) -> tuple[SubmittedArtifactInput, ...]:
    incumbent_records, incumbent_artifact_ids = _validated_incumbent(
        current_champion=current_champion,
        incumbent=incumbent,
    )
    challengers = tuple(
        record
        for record in latest_by_uid.values()
        if record.submitted_at > previous_completed_cutoff
        and record.artifact_id not in incumbent_artifact_ids
    )
    challengers_ordered = tuple(
        sorted(challengers, key=lambda record: (record.submitted_at, record.uid, str(record.artifact_id)))
    )
    return incumbent_records + challengers_ordered


def has_required_successful_validators(successful_validator_ids: Sequence[UUID]) -> bool:
    return len(successful_validator_ids) >= REQUIRED_SUCCESSFUL_VALIDATOR_COUNT


def eligible_validator_ids(successful_validator_ids: Sequence[UUID]) -> frozenset[UUID]:
    return frozenset(successful_validator_ids)


def filter_successful_validator_runs(
    runs: Sequence[_RunT],
    *,
    successful_validator_ids: Sequence[UUID],
) -> tuple[_RunT, ...]:
    allowed_validator_ids = eligible_validator_ids(successful_validator_ids)
    return tuple(run for run in runs if run.validator_id in allowed_validator_ids)


def select_champion(
    *,
    task_ids: Sequence[UUID],
    artifacts: Sequence[ChampionArtifactInput],
    runs: Sequence[ChampionRunInput],
    current_champion_artifact_id: UUID | None,
    cascade: RankingCascade,
) -> ChampionSelection | None:
    validated_runs, candidate_artifact_ids, artifact_uid_map = validate_champion_run_inputs(
        task_ids=task_ids,
        artifacts=artifacts,
        runs=runs,
    )
    aggregates = aggregate_ranking_rows(
        tuple(
            ArtifactRankingRow(
                validator_id=run.validator_id,
                artifact_id=run.artifact_id,
                task_id=run.task_id,
                score=run.score,
                total_cost_usd=run.total_cost_usd,
                elapsed_ms=run.elapsed_ms,
            )
            for run in validated_runs
        )
    )
    if not aggregates.vectors or not aggregates.totals:
        return None

    challengers_ordered = ordered_challengers(
        initial=current_champion_artifact_id,
        candidate_artifact_ids=candidate_artifact_ids,
    )
    ranking_trace = cascade.trace(
        initial=current_champion_artifact_id,
        challengers_ordered=challengers_ordered,
        aggregates=aggregates,
    )
    champion_artifact_id = ranking_trace.final_artifact_id
    if champion_artifact_id is None:
        return ChampionSelection(
            champion_uid=None,
            weights={},
            score=0.0,
            champion_artifact_id=None,
            incumbent_artifact_id=current_champion_artifact_id,
            ranking_trace=ranking_trace,
        )

    champion_uid = artifact_uid_map[champion_artifact_id]
    artifact_scores = _artifact_scores(
        artifact_ids=candidate_artifact_ids,
        task_count=len(task_ids),
        aggregates=aggregates,
    )
    return ChampionSelection(
        champion_uid=champion_uid,
        weights=compose_champion_weights(champion_uid),
        score=artifact_scores[champion_artifact_id],
        champion_artifact_id=champion_artifact_id,
        artifact_scores=artifact_scores,
        incumbent_artifact_id=current_champion_artifact_id,
        ranking_trace=ranking_trace,
        similarity_fallback_artifact_ids=_similarity_fallback_artifact_ids(
            ranking_trace=ranking_trace,
            incumbent_artifact_id=current_champion_artifact_id,
        ),
    )


def validate_champion_run_inputs(
    *,
    task_ids: Sequence[UUID],
    artifacts: Sequence[ChampionArtifactInput],
    runs: Sequence[ChampionRunInput],
) -> tuple[tuple[ChampionRunInput, ...], tuple[UUID, ...], dict[UUID, int]]:
    if not task_ids:
        raise ValueError("batch contains no tasks")

    candidate_artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    if len(set(candidate_artifact_ids)) != len(candidate_artifact_ids):
        raise ValueError("batch contains duplicate artifact ids")

    task_id_set = set(task_ids)
    expected_pairs = {
        (artifact_id, task_id)
        for artifact_id, task_id in product(candidate_artifact_ids, task_id_set)
    }

    artifact_uid_map = {artifact.artifact_id: artifact.uid for artifact in artifacts}
    records_by_validator: dict[UUID, list[ChampionRunInput]] = {}
    eligible_runs: list[ChampionRunInput] = []
    for run in runs:
        records_by_validator.setdefault(run.validator_id, []).append(run)

    for validator_id, validator_runs in records_by_validator.items():
        seen_pairs: set[tuple[UUID, UUID]] = set()
        for run in validator_runs:
            if run.artifact_id not in artifact_uid_map:
                raise ValueError(f"run referenced artifact outside batch: {run.artifact_id}")
            if run.task_id not in task_id_set:
                raise ValueError(f"run referenced task outside batch: {run.task_id}")

            pair = (run.artifact_id, run.task_id)
            if pair in seen_pairs:
                raise ValueError(
                    "duplicate run pair for validator "
                    f"{validator_id}: artifact={run.artifact_id} task={run.task_id}"
                )
            seen_pairs.add(pair)
        if seen_pairs != expected_pairs:
            missing_pairs = expected_pairs - seen_pairs
            raise ValueError(
                "validator has incomplete run coverage for batch "
                f"{validator_id}: missing_pairs={sorted(missing_pairs)}"
            )
        eligible_runs.extend(validator_runs)

    return tuple(eligible_runs), candidate_artifact_ids, artifact_uid_map


def _validated_incumbent(
    *,
    current_champion: CurrentChampionInput | None,
    incumbent: SubmittedArtifactInput | None,
) -> tuple[tuple[SubmittedArtifactInput, ...], set[UUID]]:
    if current_champion is None:
        return (), set()
    if incumbent is None:
        raise RuntimeError(f"incumbent script missing for cutoff: {current_champion.artifact_id}")
    if incumbent.uid != current_champion.uid:
        raise RuntimeError(
            f"incumbent script uid mismatch: champion uid={current_champion.uid} script uid={incumbent.uid}"
        )
    if incumbent.artifact_id != current_champion.artifact_id:
        raise RuntimeError(
            "incumbent script artifact mismatch: "
            f"champion artifact={current_champion.artifact_id} script artifact={incumbent.artifact_id}"
        )
    return (incumbent,), {incumbent.artifact_id}


def _champion_batch_score(
    *,
    champion_artifact_id: UUID,
    task_count: int,
    aggregates: ArtifactAggregateBundle,
) -> float:
    if task_count <= 0:
        raise ValueError("batch contains no tasks")
    score = float(aggregates.totals[champion_artifact_id]) / float(task_count)
    if not isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("champion batch score must be between 0.0 and 1.0")
    return score


def _artifact_scores(
    *,
    artifact_ids: Sequence[UUID],
    task_count: int,
    aggregates: ArtifactAggregateBundle,
) -> dict[UUID, float]:
    return {
        artifact_id: _champion_batch_score(
            champion_artifact_id=artifact_id,
            task_count=task_count,
            aggregates=aggregates,
        )
        for artifact_id in artifact_ids
        if artifact_id in aggregates.totals
    }


def _stored_champion_score(*, champion_uid: int, weights: Mapping[str, float]) -> float:
    if len(weights) != 1:
        return 1.0
    raw_score = weights.get(str(champion_uid))
    if raw_score is None:
        return 1.0
    score = float(raw_score)
    if not isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("stored champion score must be between 0.0 and 1.0")
    return score


def _similarity_fallback_artifact_ids(
    *,
    ranking_trace: RankingCascadeTrace,
    incumbent_artifact_id: UUID | None,
) -> tuple[UUID, ...]:
    if incumbent_artifact_id is None:
        return ()
    dethroners = tuple(
        artifact_id
        for artifact_id in ranking_trace.successful_dethroner_artifact_ids()
        if artifact_id != incumbent_artifact_id
    )
    return tuple(reversed(dethroners))


__all__ = [
    "ChampionArtifactInput",
    "ChampionRunInput",
    "ChampionSelection",
    "CurrentChampionInput",
    "REQUIRED_SUCCESSFUL_VALIDATOR_COUNT",
    "SubmittedArtifactInput",
    "eligible_validator_ids",
    "filter_successful_validator_runs",
    "has_required_successful_validators",
    "select_batch_artifacts",
    "select_champion",
    "selection_from_stored_champion_weights",
    "validate_champion_run_inputs",
]
