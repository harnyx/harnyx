"""Shared artifact aggregation and champion-ranking helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from harnyx_commons.domain.miner_task import EvaluationDetails

ScoreVector = list[float]
_SCORE_PRECISION = 12

COST_REDUCTION_REQUIRED = 0.20
TIME_REDUCTION_REQUIRED = 0.20
TIME_REDUCTION_MIN_MS = 1000.0


@dataclass(frozen=True, slots=True)
class ArtifactRankingRow:
    validator_id: UUID
    artifact_id: UUID
    task_id: UUID
    score: float
    total_cost_usd: float
    elapsed_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ArtifactAggregateBundle:
    vectors: dict[UUID, ScoreVector]
    totals: dict[UUID, float]
    costs: dict[UUID, float]
    median_elapsed_ms: dict[UUID, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankingCascadeStep:
    incumbent_artifact_id: UUID | None
    challenger_artifact_id: UUID
    selected_artifact_id: UUID | None
    dethroned: bool


@dataclass(frozen=True, slots=True)
class RankingCascadeTrace:
    initial_artifact_id: UUID | None
    final_artifact_id: UUID | None
    steps: tuple[RankingCascadeStep, ...]

    def successful_dethroner_artifact_ids(self) -> tuple[UUID, ...]:
        return tuple(step.challenger_artifact_id for step in self.steps if step.dethroned)


@dataclass(frozen=True)
class CascadeConfig:
    """Static cascade configuration identical to platform defaults."""

    score_margin_required: float = 0.20


class RankingCascade:
    """Applies dethroning rules using challenger order and aggregate metrics."""

    def __init__(self, config: CascadeConfig) -> None:
        if not 0.0 < config.score_margin_required <= 1.0:
            raise ValueError("score_margin_required must be in (0.0, 1.0]")
        self._cfg = config

    def decide(
        self,
        *,
        initial: UUID | None,
        challengers_ordered: Iterable[UUID],
        aggregates: ArtifactAggregateBundle,
    ) -> UUID | None:
        return self.trace(
            initial=initial,
            challengers_ordered=challengers_ordered,
            aggregates=aggregates,
        ).final_artifact_id

    def trace(
        self,
        *,
        initial: UUID | None,
        challengers_ordered: Iterable[UUID],
        aggregates: ArtifactAggregateBundle,
    ) -> RankingCascadeTrace:
        current = initial if self._has_positive_total(initial, aggregates) else None
        steps: list[RankingCascadeStep] = []
        for artifact_id in challengers_ordered:
            if artifact_id not in aggregates.vectors:
                continue
            incumbent_before = current
            dethroned = False
            if current is None:
                if self._has_positive_total(artifact_id, aggregates):
                    current = artifact_id
                    incumbent_before = initial
                    dethroned = initial is not None and initial != artifact_id
                steps.append(
                    RankingCascadeStep(
                        incumbent_artifact_id=incumbent_before,
                        challenger_artifact_id=artifact_id,
                        selected_artifact_id=current,
                        dethroned=dethroned,
                    )
                )
                continue
            if self._can_dethrone(
                challenger_artifact_id=artifact_id,
                incumbent_artifact_id=current,
                aggregates=aggregates,
            ):
                current = artifact_id
                dethroned = True
            steps.append(
                RankingCascadeStep(
                    incumbent_artifact_id=incumbent_before,
                    challenger_artifact_id=artifact_id,
                    selected_artifact_id=current,
                    dethroned=dethroned,
                )
            )
        return RankingCascadeTrace(
            initial_artifact_id=initial,
            final_artifact_id=current,
            steps=tuple(steps),
        )

    def _can_dethrone(
        self,
        *,
        challenger_artifact_id: UUID,
        incumbent_artifact_id: UUID,
        aggregates: ArtifactAggregateBundle,
    ) -> bool:
        challenger_total = float(aggregates.totals.get(challenger_artifact_id, 0.0))
        incumbent_total = float(aggregates.totals.get(incumbent_artifact_id, 0.0))
        if challenger_total <= 0.0:
            return False
        if incumbent_total <= 0.0:
            return True
        margin = self._cfg.score_margin_required
        if challenger_total >= incumbent_total * (1.0 + margin):
            return True
        if challenger_total < incumbent_total and not math.isclose(
            challenger_total,
            incumbent_total,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return False
        return _is_meaningfully_lower(
            candidate_metric=aggregates.costs.get(challenger_artifact_id),
            incumbent_metric=aggregates.costs.get(incumbent_artifact_id),
            reduction_required=COST_REDUCTION_REQUIRED,
        ) or _is_meaningfully_faster(
            candidate_metric=aggregates.median_elapsed_ms.get(challenger_artifact_id),
            incumbent_metric=aggregates.median_elapsed_ms.get(incumbent_artifact_id),
            reduction_required=TIME_REDUCTION_REQUIRED,
            min_reduction_ms=TIME_REDUCTION_MIN_MS,
        )

    @staticmethod
    def _has_positive_total(artifact_id: UUID | None, aggregates: ArtifactAggregateBundle) -> bool:
        if artifact_id is None:
            return False
        return float(aggregates.totals.get(artifact_id, 0.0)) > 0.0


def aggregate_ranking_rows(
    rows: Sequence[ArtifactRankingRow],
) -> ArtifactAggregateBundle:
    if not rows:
        return ArtifactAggregateBundle(vectors={}, totals={}, costs={})

    task_ids = sorted({row.task_id for row in rows}, key=lambda task_id: task_id.hex)
    task_positions = {task_id: index for index, task_id in enumerate(task_ids)}
    vector_length = len(task_ids)

    vectors_by_validator: dict[UUID, dict[UUID, ScoreVector]] = {}
    costs_by_validator: dict[UUID, dict[UUID, float]] = {}
    elapsed_by_validator: dict[UUID, dict[UUID, float]] = {}
    elapsed_missing: set[tuple[UUID, UUID]] = set()
    pair_counts_by_validator: dict[UUID, int] = {}
    seen_pairs_by_validator: dict[UUID, set[tuple[UUID, UUID]]] = {}

    for row in rows:
        position = task_positions[row.task_id]
        validator_vectors = vectors_by_validator.setdefault(row.validator_id, {})
        vector = validator_vectors.setdefault(row.artifact_id, [0.0] * vector_length)

        seen_pairs = seen_pairs_by_validator.setdefault(row.validator_id, set())
        pair = (row.artifact_id, row.task_id)
        if pair in seen_pairs:
            raise ValueError(
                "duplicate run pair for validator "
                f"{row.validator_id}: artifact={row.artifact_id} task={row.task_id}"
            )
        seen_pairs.add(pair)

        vector[position] = _normalize_score(row.score)
        pair_counts_by_validator[row.validator_id] = pair_counts_by_validator.get(row.validator_id, 0) + 1

        validator_costs = costs_by_validator.setdefault(row.validator_id, {})
        validator_costs[row.artifact_id] = validator_costs.get(row.artifact_id, 0.0) + float(row.total_cost_usd)

        validator_elapsed = elapsed_by_validator.setdefault(row.validator_id, {})
        if row.elapsed_ms is None:
            elapsed_missing.add((row.validator_id, row.artifact_id))
        else:
            validator_elapsed[row.artifact_id] = validator_elapsed.get(row.artifact_id, 0.0) + float(row.elapsed_ms)

    validator_ids = sorted(vectors_by_validator, key=lambda validator_id: validator_id.hex)
    expected_artifact_ids = {
        artifact_id
        for validator_vectors in vectors_by_validator.values()
        for artifact_id in validator_vectors.keys()
    }
    expected_count_per_validator = len(expected_artifact_ids) * len(task_ids)

    for validator_id in validator_ids:
        present_artifact_ids = set(vectors_by_validator[validator_id])
        if present_artifact_ids != expected_artifact_ids:
            raise ValueError(f"incomplete runs for validator {validator_id}")
        if pair_counts_by_validator.get(validator_id, 0) != expected_count_per_validator:
            raise ValueError(f"incomplete runs for validator {validator_id}")

    aggregate_vectors: dict[UUID, ScoreVector] = {}
    totals_by_artifact: dict[UUID, float] = {}
    costs_by_artifact: dict[UUID, float] = {}
    median_elapsed_ms_by_artifact: dict[UUID, float] = {}
    for artifact_id in sorted(expected_artifact_ids, key=lambda value: value.hex):
        vectors_for_artifact = [vectors_by_validator[validator_id][artifact_id] for validator_id in validator_ids]
        aggregate_vector = [
            _normalize_score(statistics.median(vector[position] for vector in vectors_for_artifact))
            for position in range(vector_length)
        ]
        aggregate_vectors[artifact_id] = aggregate_vector
        totals_by_artifact[artifact_id] = _normalize_score(math.fsum(aggregate_vector))
        costs_by_artifact[artifact_id] = float(
            statistics.median(costs_by_validator[validator_id][artifact_id] for validator_id in validator_ids)
        )
        if any((validator_id, artifact_id) in elapsed_missing for validator_id in validator_ids):
            continue
        median_elapsed_ms_by_artifact[artifact_id] = float(
            statistics.median(elapsed_by_validator[validator_id][artifact_id] for validator_id in validator_ids)
        )

    return ArtifactAggregateBundle(
        vectors=aggregate_vectors,
        totals=totals_by_artifact,
        costs=costs_by_artifact,
        median_elapsed_ms=median_elapsed_ms_by_artifact,
    )


def ordered_challengers(
    *,
    initial: UUID | None,
    candidate_artifact_ids: Sequence[UUID],
) -> list[UUID]:
    incumbents = {initial} if initial is not None else set()
    return [artifact_id for artifact_id in candidate_artifact_ids if artifact_id not in incumbents]


def _normalize_score(value: float) -> float:
    return round(float(value), _SCORE_PRECISION)


def run_ranking_cost_usd(details: EvaluationDetails) -> float:
    return float(details.total_tool_usage.llm_cost + details.total_tool_usage.search_tool_cost)


def run_contribution_score(
    *,
    is_completed: bool,
    score: float | None,
    details: EvaluationDetails | None,
) -> float:
    if not is_completed:
        raise ValueError("only completed runs contribute to aggregates")
    if details is None:
        raise ValueError("completed runs must include details")
    if details.error is not None:
        if score != 0.0:
            raise ValueError("failed completed runs must contribute a zero score")
        return 0.0

    breakdown = details.score_breakdown
    if breakdown is None:
        raise ValueError("successful completed runs must include score breakdown details")
    if score is None:
        raise ValueError("successful completed runs must include a score")
    if not math.isclose(score, breakdown.total_score, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("completed run score must match details.score_breakdown.total_score")
    return float(breakdown.total_score)


def summarized_run_contribution_score(
    *,
    is_completed: bool,
    score: float | None,
    has_error: bool,
) -> float:
    if not is_completed:
        raise ValueError("only completed runs contribute to aggregates")
    if has_error:
        if score != 0.0:
            raise ValueError("failed completed runs must contribute a zero score")
        return 0.0
    if score is None:
        raise ValueError("successful completed runs must include a score")
    return float(score)


def _is_meaningfully_lower(
    *,
    candidate_metric: float | None,
    incumbent_metric: float | None,
    reduction_required: float,
) -> bool:
    if candidate_metric is None or incumbent_metric is None:
        return False
    if incumbent_metric <= 0.0:
        return False
    threshold = incumbent_metric * (1.0 - reduction_required)
    return candidate_metric <= threshold or math.isclose(
        candidate_metric,
        threshold,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _is_meaningfully_faster(
    *,
    candidate_metric: float | None,
    incumbent_metric: float | None,
    reduction_required: float,
    min_reduction_ms: float,
) -> bool:
    if candidate_metric is None or incumbent_metric is None:
        return False
    if not _is_meaningfully_lower(
        candidate_metric=candidate_metric,
        incumbent_metric=incumbent_metric,
        reduction_required=reduction_required,
    ):
        return False
    delta = incumbent_metric - candidate_metric
    return delta >= min_reduction_ms or math.isclose(
        delta,
        min_reduction_ms,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


__all__ = [
    "ArtifactAggregateBundle",
    "ArtifactRankingRow",
    "COST_REDUCTION_REQUIRED",
    "CascadeConfig",
    "RankingCascade",
    "RankingCascadeStep",
    "RankingCascadeTrace",
    "TIME_REDUCTION_MIN_MS",
    "TIME_REDUCTION_REQUIRED",
    "aggregate_ranking_rows",
    "ordered_challengers",
    "run_contribution_score",
    "run_ranking_cost_usd",
    "summarized_run_contribution_score",
]
