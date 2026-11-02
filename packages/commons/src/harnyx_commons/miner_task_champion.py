"""Miner-task participant and champion selection policies."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
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
    QualifyingSelection,
    RankingCascade,
    RankingCascadeTrace,
    aggregate_competition_components,
    aggregate_ranking_rows,
    ordered_challengers,
    select_qualifying_participants,
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
    miner_hotkey_ss58: str | None = None


@dataclass(frozen=True, slots=True)
class CurrentChampionInput:
    uid: int
    artifact_id: UUID
    miner_hotkey_ss58: str | None = None


@dataclass(frozen=True, slots=True)
class ChampionArtifactInput:
    artifact_id: UUID
    uid: int
    miner_hotkey_ss58: str | None = None


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
    champion_hotkey_ss58: str | None = None
    artifact_scores: dict[UUID, float] = field(default_factory=dict, compare=False)
    incumbent_artifact_id: UUID | None = field(default=None, compare=False)
    ranking_trace: RankingCascadeTrace | None = field(default=None, compare=False)
    similarity_fallback_artifact_ids: tuple[UUID, ...] = field(default=(), compare=False)


@dataclass(frozen=True, slots=True)
class ChampionSelections:
    overall: ChampionSelection
    f1: ChampionSelection


@dataclass(frozen=True, slots=True)
class QualifyingSelections:
    overall: QualifyingSelection
    f1: QualifyingSelection

    @property
    def participant_artifact_ids(self) -> tuple[UUID, ...]:
        incumbents = tuple(
            artifact
            for artifact in (self.overall.incumbent_artifact_id, self.f1.incumbent_artifact_id)
            if artifact is not None
        )
        return tuple(
            dict.fromkeys((*incumbents, *self.overall.participant_artifact_ids, *self.f1.participant_artifact_ids))
        )


def select_competition(
    *,
    task_ids: Sequence[UUID],
    deep_task_ids: Collection[UUID],
    artifacts: Sequence[ChampionArtifactInput],
    rows: Sequence[ArtifactRankingRow],
    validator_ids: Sequence[UUID],
    overall_incumbent: UUID | None,
    f1_incumbent: UUID | None,
    cascade: RankingCascade,
    qualifying: bool,
) -> ChampionSelections | QualifyingSelections:
    artifact_ids, identities = _validated_artifact_identities(artifacts)
    views = aggregate_competition_components(
        rows, task_ids=task_ids, deep_task_ids=deep_task_ids, artifact_ids=artifact_ids, validator_ids=validator_ids
    )
    if qualifying:
        selections = QualifyingSelections(
            overall=select_qualifying_participants(
                initial=overall_incumbent,
                challengers_ordered=ordered_challengers(initial=overall_incumbent, candidate_artifact_ids=artifact_ids),
                aggregates=views.overall,
                cascade=cascade,
            ),
            f1=select_qualifying_participants(
                initial=f1_incumbent,
                challengers_ordered=ordered_challengers(initial=f1_incumbent, candidate_artifact_ids=artifact_ids),
                aggregates=views.f1,
                cascade=cascade,
            ),
        )
        if selections.participant_artifact_ids:
            return selections
    overall = _select_champion_from_aggregates(
        task_count=len(task_ids),
        candidate_artifact_ids=artifact_ids,
        artifact_identity_map=identities,
        aggregates=views.overall,
        current_champion_artifact_id=overall_incumbent,
        cascade=cascade,
    )
    f1 = _select_champion_from_aggregates(
        task_count=len(task_ids),
        candidate_artifact_ids=artifact_ids,
        artifact_identity_map=identities,
        aggregates=views.f1,
        current_champion_artifact_id=f1_incumbent,
        cascade=cascade,
    )
    if overall is None or f1 is None:
        raise ValueError("competition requires a complete artifact surface")
    return ChampionSelections(overall=overall, f1=f1)


def selection_from_stored_champion_weights(
    *,
    final_top: Sequence[int],
    weights: Mapping[str, float],
    champion_artifact_id: UUID | None,
    champion_hotkey_ss58: str | None = None,
) -> ChampionSelection:
    champion_uid = int(final_top[0]) if final_top else None
    if champion_uid is None:
        return ChampionSelection(
            champion_uid=None,
            weights={},
            score=0.0,
            champion_artifact_id=champion_artifact_id,
            champion_hotkey_ss58=champion_hotkey_ss58,
        )
    return ChampionSelection(
        champion_uid=champion_uid,
        weights=compose_champion_weights(champion_uid),
        score=_stored_champion_score(champion_uid=champion_uid, weights=weights),
        champion_artifact_id=champion_artifact_id,
        champion_hotkey_ss58=champion_hotkey_ss58,
    )


def select_batch_artifacts(
    *,
    latest_by_hotkey: Mapping[str, SubmittedArtifactInput],
    previous_completed_cutoff: datetime,
    current_champion: CurrentChampionInput | None,
    incumbent: SubmittedArtifactInput | None,
    f1_champion: CurrentChampionInput | None = None,
    f1_incumbent: SubmittedArtifactInput | None = None,
) -> tuple[SubmittedArtifactInput, ...]:
    incumbent_records, incumbent_artifact_ids = _validated_incumbent(
        current_champion=current_champion,
        incumbent=incumbent,
    )
    f1_records, f1_artifact_ids = _validated_incumbent(current_champion=f1_champion, incumbent=f1_incumbent)
    incumbent_records = tuple({record.artifact_id: record for record in (*incumbent_records, *f1_records)}.values())
    incumbent_artifact_ids |= f1_artifact_ids
    challengers = tuple(
        record
        for record in latest_by_hotkey.values()
        if record.submitted_at > previous_completed_cutoff and record.artifact_id not in incumbent_artifact_ids
    )
    incumbent_hotkeys = {
        champion.miner_hotkey_ss58 for champion in (current_champion, f1_champion) if champion is not None
    }
    challengers_ordered = tuple(
        sorted(challengers, key=lambda record: _challenger_order_key(record, incumbent_hotkeys=incumbent_hotkeys))
    )
    return incumbent_records + challengers_ordered


def _challenger_order_key(
    record: SubmittedArtifactInput,
    *,
    incumbent_hotkeys: set[str | None],
) -> tuple[int, datetime, int, str]:
    incumbent_hotkey_priority = (
        0 if record.miner_hotkey_ss58 is not None and record.miner_hotkey_ss58 in incumbent_hotkeys else 1
    )
    return (
        incumbent_hotkey_priority,
        record.submitted_at,
        record.uid,
        str(record.artifact_id),
    )


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
    candidate_artifact_ids, artifact_identity_map, aggregates = _prepare_ranking_inputs(
        task_ids=task_ids,
        artifacts=artifacts,
        runs=runs,
    )
    return _select_champion_from_aggregates(
        task_count=len(task_ids),
        candidate_artifact_ids=candidate_artifact_ids,
        artifact_identity_map=artifact_identity_map,
        aggregates=aggregates,
        current_champion_artifact_id=current_champion_artifact_id,
        cascade=cascade,
    )


def select_qualifying(
    *,
    task_ids: Sequence[UUID],
    artifacts: Sequence[ChampionArtifactInput],
    runs: Sequence[ChampionRunInput],
    current_champion_artifact_id: UUID | None,
    cascade: RankingCascade,
) -> QualifyingSelection | ChampionSelection | None:
    candidate_artifact_ids, artifact_identity_map, aggregates = _prepare_ranking_inputs(
        task_ids=task_ids,
        artifacts=artifacts,
        runs=runs,
    )
    if not aggregates.vectors or not aggregates.totals:
        return None
    selection = select_qualifying_participants(
        initial=current_champion_artifact_id,
        challengers_ordered=ordered_challengers(
            initial=current_champion_artifact_id,
            candidate_artifact_ids=candidate_artifact_ids,
        ),
        aggregates=aggregates,
        cascade=cascade,
    )
    if selection.participant_artifact_ids:
        return selection
    # Preserve the existing persisted decision for championless all-zero completion.
    return _select_champion_from_aggregates(
        task_count=len(task_ids),
        candidate_artifact_ids=candidate_artifact_ids,
        artifact_identity_map=artifact_identity_map,
        aggregates=aggregates,
        current_champion_artifact_id=current_champion_artifact_id,
        cascade=cascade,
    )


def _prepare_ranking_inputs(
    *,
    task_ids: Sequence[UUID],
    artifacts: Sequence[ChampionArtifactInput],
    runs: Sequence[ChampionRunInput],
) -> tuple[tuple[UUID, ...], dict[UUID, _ArtifactIdentity], ArtifactAggregateBundle]:
    validated_runs, candidate_artifact_ids, artifact_identity_map = validate_champion_run_inputs(
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
    return candidate_artifact_ids, artifact_identity_map, aggregates


def _select_champion_from_aggregates(
    *,
    task_count: int,
    candidate_artifact_ids: tuple[UUID, ...],
    artifact_identity_map: dict[UUID, _ArtifactIdentity],
    aggregates: ArtifactAggregateBundle,
    current_champion_artifact_id: UUID | None,
    cascade: RankingCascade,
) -> ChampionSelection | None:
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
            artifact_scores=artifact_batch_scores(
                artifact_ids=candidate_artifact_ids, task_count=task_count, aggregates=aggregates
            )
            if aggregates.score_maximum is not None
            else {},
        )

    champion_identity = artifact_identity_map[champion_artifact_id]
    artifact_scores = artifact_batch_scores(
        artifact_ids=candidate_artifact_ids,
        task_count=task_count,
        aggregates=aggregates,
    )
    return ChampionSelection(
        champion_uid=champion_identity.uid,
        weights=compose_champion_weights(champion_identity.uid),
        score=artifact_scores[champion_artifact_id],
        champion_artifact_id=champion_artifact_id,
        champion_hotkey_ss58=champion_identity.hotkey,
        artifact_scores=artifact_scores,
        incumbent_artifact_id=current_champion_artifact_id,
        ranking_trace=ranking_trace,
        similarity_fallback_artifact_ids=_similarity_fallback_artifact_ids(
            ranking_trace=ranking_trace,
            incumbent_artifact_id=current_champion_artifact_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class _ArtifactIdentity:
    uid: int
    hotkey: str


def validate_champion_run_inputs(
    *,
    task_ids: Sequence[UUID],
    artifacts: Sequence[ChampionArtifactInput],
    runs: Sequence[ChampionRunInput],
) -> tuple[tuple[ChampionRunInput, ...], tuple[UUID, ...], dict[UUID, _ArtifactIdentity]]:
    if not task_ids:
        raise ValueError("batch contains no tasks")

    candidate_artifact_ids, artifact_identity_map = _validated_artifact_identities(artifacts)

    task_id_set = set(task_ids)
    expected_pairs = {(artifact_id, task_id) for artifact_id, task_id in product(candidate_artifact_ids, task_id_set)}

    records_by_validator: dict[UUID, list[ChampionRunInput]] = {}
    eligible_runs: list[ChampionRunInput] = []
    for run in runs:
        records_by_validator.setdefault(run.validator_id, []).append(run)

    for validator_id, validator_runs in records_by_validator.items():
        seen_pairs: set[tuple[UUID, UUID]] = set()
        for run in validator_runs:
            if run.artifact_id not in artifact_identity_map:
                raise ValueError(f"run referenced artifact outside batch: {run.artifact_id}")
            if run.task_id not in task_id_set:
                raise ValueError(f"run referenced task outside batch: {run.task_id}")

            pair = (run.artifact_id, run.task_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"duplicate run pair for validator {validator_id}: artifact={run.artifact_id} task={run.task_id}"
                )
            seen_pairs.add(pair)
        if seen_pairs != expected_pairs:
            missing_pairs = expected_pairs - seen_pairs
            raise ValueError(
                f"validator has incomplete run coverage for batch {validator_id}: missing_pairs={sorted(missing_pairs)}"
            )
        eligible_runs.extend(validator_runs)

    return tuple(eligible_runs), candidate_artifact_ids, artifact_identity_map


def _validated_artifact_identities(
    artifacts: Sequence[ChampionArtifactInput],
) -> tuple[tuple[UUID, ...], dict[UUID, _ArtifactIdentity]]:
    artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("batch contains duplicate artifact ids")
    identities: dict[UUID, _ArtifactIdentity] = {}
    for artifact in artifacts:
        if artifact.miner_hotkey_ss58 is None:
            raise ValueError(f"artifact missing miner hotkey: {artifact.artifact_id}")
        identities[artifact.artifact_id] = _ArtifactIdentity(uid=artifact.uid, hotkey=artifact.miner_hotkey_ss58)
    return artifact_ids, identities


def _validated_incumbent(
    *,
    current_champion: CurrentChampionInput | None,
    incumbent: SubmittedArtifactInput | None,
) -> tuple[tuple[SubmittedArtifactInput, ...], set[UUID]]:
    if current_champion is None:
        return (), set()
    if incumbent is None:
        raise RuntimeError(f"incumbent script missing for cutoff: {current_champion.artifact_id}")
    if current_champion.miner_hotkey_ss58 is None:
        raise RuntimeError("current champion missing miner hotkey")
    if incumbent.miner_hotkey_ss58 != current_champion.miner_hotkey_ss58:
        raise RuntimeError(
            "incumbent script hotkey mismatch: "
            f"champion hotkey={current_champion.miner_hotkey_ss58} script hotkey={incumbent.miner_hotkey_ss58}"
        )
    if incumbent.artifact_id != current_champion.artifact_id:
        raise RuntimeError(
            "incumbent script artifact mismatch: "
            f"champion artifact={current_champion.artifact_id} script artifact={incumbent.artifact_id}"
        )
    return (incumbent,), {incumbent.artifact_id}


def artifact_batch_score(
    *,
    artifact_id: UUID,
    task_count: int,
    aggregates: ArtifactAggregateBundle,
) -> float:
    if task_count <= 0:
        raise ValueError("batch contains no tasks")
    score = float(aggregates.totals[artifact_id]) / (
        aggregates.score_maximum if aggregates.score_maximum is not None else float(task_count)
    )
    if not isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("artifact batch score must be between 0.0 and 1.0")
    return score


def artifact_batch_scores(
    *,
    artifact_ids: Sequence[UUID],
    task_count: int,
    aggregates: ArtifactAggregateBundle,
) -> dict[UUID, float]:
    return {
        artifact_id: artifact_batch_score(
            artifact_id=artifact_id,
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
    "ChampionSelections",
    "QualifyingSelections",
    "select_competition",
    "select_qualifying",
    "ChampionArtifactInput",
    "ChampionRunInput",
    "ChampionSelection",
    "CurrentChampionInput",
    "REQUIRED_SUCCESSFUL_VALIDATOR_COUNT",
    "SubmittedArtifactInput",
    "artifact_batch_score",
    "artifact_batch_scores",
    "eligible_validator_ids",
    "filter_successful_validator_runs",
    "has_required_successful_validators",
    "select_batch_artifacts",
    "select_champion",
    "selection_from_stored_champion_weights",
    "validate_champion_run_inputs",
]
