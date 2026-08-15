"""Shared artifact aggregation and champion-ranking helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from harnyx_commons.domain.miner_task import EvaluationDetails

ScoreVector = list[float]
_SCORE_PRECISION = 12

COST_REDUCTION_REQUIRED = 0.10
TIME_REDUCTION_REQUIRED = 0.10
TIME_REDUCTION_MIN_MS = 1000.0
_SCORE_MARGIN_FRACTION_OF_MAXIMUM = 0.10
_HISTORICAL_SCORE_MARGIN_REQUIRED = 0.20
_HISTORICAL_COST_REDUCTION_REQUIRED = 0.20
_HISTORICAL_TIME_REDUCTION_REQUIRED = 0.20


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


class RankingDecisionRule(StrEnum):
    POSITIVE_SCORE = "positive_score"
    SCORE_MARGIN = "score_margin"
    COST_REDUCTION = "cost_reduction"
    RUNTIME_REDUCTION = "runtime_reduction"


class RankingRuleStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class RankingRuleEvaluation:
    rule: RankingDecisionRule
    status: RankingRuleStatus
    required_relative_improvement: float | None = None
    required_absolute_improvement: float | None = None
    observed_relative_improvement: float | None = None
    observed_absolute_improvement: float | None = None


@dataclass(frozen=True, slots=True)
class RankingCascadeEvaluation:
    selected_rule: RankingDecisionRule | None
    score_non_regressing: bool | None
    rules: tuple[RankingRuleEvaluation, ...]
    cost_non_regressing: bool | None = None
    runtime_non_regressing: bool | None = None


@dataclass(frozen=True, slots=True)
class RankingCascadeStep:
    incumbent_artifact_id: UUID | None
    challenger_artifact_id: UUID
    selected_artifact_id: UUID | None
    dethroned: bool
    evaluation: RankingCascadeEvaluation | None = None


@dataclass(frozen=True, slots=True)
class RankingCascadeTrace:
    initial_artifact_id: UUID | None
    final_artifact_id: UUID | None
    steps: tuple[RankingCascadeStep, ...]

    def champion_lineage_artifact_ids(self) -> tuple[UUID, ...]:
        """Return champions in the order they entered the lineage."""

        lineage = () if self.initial_artifact_id is None else (self.initial_artifact_id,)
        selected_challengers = tuple(
            step.challenger_artifact_id
            for step in self.steps
            if step.selected_artifact_id == step.challenger_artifact_id
        )
        return tuple(dict.fromkeys((*lineage, *selected_challengers)))

    def successful_dethroner_artifact_ids(self) -> tuple[UUID, ...]:
        return tuple(step.challenger_artifact_id for step in self.steps if step.dethroned)

    def reverse_champion_lineage(self) -> tuple[UUID, ...]:
        """Return final champion first, followed by the champions it superseded."""

        if self.final_artifact_id is None:
            return ()
        return tuple(reversed(self.champion_lineage_artifact_ids()))


def main_participant_priority(
    *,
    main_trace: RankingCascadeTrace,
    qualifying_participant_order: Sequence[UUID],
) -> tuple[UUID, ...]:
    """Order main participants by final lineage, then reverse qualifying performance."""

    ordered = main_trace.reverse_champion_lineage() + tuple(reversed(qualifying_participant_order))
    return tuple(dict.fromkeys(ordered))


class _ScoreMarginKind(StrEnum):
    RELATIVE = "relative"
    FRACTION_OF_MAXIMUM = "fraction_of_maximum"


@dataclass(frozen=True, slots=True)
class _RankingPolicy:
    score_margin_kind: _ScoreMarginKind
    score_margin_required: float
    cost_reduction_required: float
    time_reduction_required: float


_CURRENT_POLICY = _RankingPolicy(
    score_margin_kind=_ScoreMarginKind.FRACTION_OF_MAXIMUM,
    score_margin_required=_SCORE_MARGIN_FRACTION_OF_MAXIMUM,
    cost_reduction_required=COST_REDUCTION_REQUIRED,
    time_reduction_required=TIME_REDUCTION_REQUIRED,
)
_HISTORICAL_POLICY = _RankingPolicy(
    score_margin_kind=_ScoreMarginKind.RELATIVE,
    score_margin_required=_HISTORICAL_SCORE_MARGIN_REQUIRED,
    cost_reduction_required=_HISTORICAL_COST_REDUCTION_REQUIRED,
    time_reduction_required=_HISTORICAL_TIME_REDUCTION_REQUIRED,
)


class RankingCascade:
    """Applies dethroning rules using challenger order and aggregate metrics."""

    def __init__(self) -> None:
        self._policy = _CURRENT_POLICY

    @classmethod
    def for_historical_batches(cls) -> RankingCascade:
        """Return the fixed policy used by miner-task batch data versions 1 through 7."""

        cascade = cls()
        cascade._policy = _HISTORICAL_POLICY
        return cascade

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
                evaluation = self._evaluate_positive_score(
                    challenger_artifact_id=artifact_id,
                    aggregates=aggregates,
                )
                if evaluation.selected_rule is not None:
                    current = artifact_id
                    incumbent_before = initial
                    dethroned = initial is not None and initial != artifact_id
                steps.append(
                    RankingCascadeStep(
                        incumbent_artifact_id=incumbent_before,
                        challenger_artifact_id=artifact_id,
                        selected_artifact_id=current,
                        dethroned=dethroned,
                        evaluation=evaluation,
                    )
                )
                continue
            evaluation = self._evaluate_dethrone(
                challenger_artifact_id=artifact_id,
                incumbent_artifact_id=current,
                aggregates=aggregates,
            )
            if evaluation.selected_rule is not None:
                current = artifact_id
                dethroned = True
            steps.append(
                RankingCascadeStep(
                    incumbent_artifact_id=incumbent_before,
                    challenger_artifact_id=artifact_id,
                    selected_artifact_id=current,
                    dethroned=dethroned,
                    evaluation=evaluation,
                )
            )
        return RankingCascadeTrace(
            initial_artifact_id=initial,
            final_artifact_id=current,
            steps=tuple(steps),
        )

    def _evaluate_positive_score(
        self,
        *,
        challenger_artifact_id: UUID,
        aggregates: ArtifactAggregateBundle,
    ) -> RankingCascadeEvaluation:
        passed = float(aggregates.totals.get(challenger_artifact_id, 0.0)) > 0.0
        return RankingCascadeEvaluation(
            selected_rule=RankingDecisionRule.POSITIVE_SCORE if passed else None,
            score_non_regressing=None,
            rules=(
                RankingRuleEvaluation(
                    rule=RankingDecisionRule.POSITIVE_SCORE,
                    status=RankingRuleStatus.PASSED if passed else RankingRuleStatus.FAILED,
                ),
                RankingRuleEvaluation(
                    rule=RankingDecisionRule.SCORE_MARGIN,
                    status=RankingRuleStatus.NOT_APPLICABLE,
                    required_relative_improvement=(
                        self._policy.score_margin_required
                        if self._policy.score_margin_kind is _ScoreMarginKind.RELATIVE
                        else None
                    ),
                    required_absolute_improvement=(
                        self._policy.score_margin_required
                        if self._policy.score_margin_kind is _ScoreMarginKind.FRACTION_OF_MAXIMUM
                        else None
                    ),
                ),
                RankingRuleEvaluation(
                    rule=RankingDecisionRule.COST_REDUCTION,
                    status=RankingRuleStatus.NOT_APPLICABLE,
                    required_relative_improvement=self._policy.cost_reduction_required,
                ),
                RankingRuleEvaluation(
                    rule=RankingDecisionRule.RUNTIME_REDUCTION,
                    status=RankingRuleStatus.NOT_APPLICABLE,
                    required_relative_improvement=self._policy.time_reduction_required,
                    required_absolute_improvement=TIME_REDUCTION_MIN_MS,
                ),
            ),
        )

    def _evaluate_dethrone(
        self,
        *,
        challenger_artifact_id: UUID,
        incumbent_artifact_id: UUID,
        aggregates: ArtifactAggregateBundle,
    ) -> RankingCascadeEvaluation:
        challenger_total = float(aggregates.totals.get(challenger_artifact_id, 0.0))
        incumbent_total = float(aggregates.totals.get(incumbent_artifact_id, 0.0))
        score_non_regressing = challenger_total > 0.0 and (
            challenger_total >= incumbent_total
            or math.isclose(
                challenger_total,
                incumbent_total,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
        score_rule = self._evaluate_score_margin(
            incumbent_artifact_id=incumbent_artifact_id,
            challenger_total=challenger_total,
            incumbent_total=incumbent_total,
            aggregates=aggregates,
        )

        challenger_cost = aggregates.costs.get(challenger_artifact_id)
        incumbent_cost = aggregates.costs.get(incumbent_artifact_id)
        challenger_runtime = aggregates.median_elapsed_ms.get(challenger_artifact_id)
        incumbent_runtime = aggregates.median_elapsed_ms.get(incumbent_artifact_id)
        cost_non_regressing = _lower_metric_non_regressing(
            challenger_metric=challenger_cost,
            incumbent_metric=incumbent_cost,
        )
        runtime_non_regressing = _lower_metric_non_regressing(
            challenger_metric=challenger_runtime,
            incumbent_metric=incumbent_runtime,
        )
        cost_rule = RankingRuleEvaluation(
            rule=RankingDecisionRule.COST_REDUCTION,
            status=_efficiency_rule_status(
                metric_available=challenger_cost is not None and incumbent_cost is not None,
                other_dimension_non_regressing=runtime_non_regressing,
                score_non_regressing=score_non_regressing,
                metric_passed=_is_meaningfully_lower(
                    candidate_metric=challenger_cost,
                    incumbent_metric=incumbent_cost,
                    reduction_required=self._policy.cost_reduction_required,
                ),
            ),
            required_relative_improvement=self._policy.cost_reduction_required,
            observed_relative_improvement=_relative_improvement(
                challenger_metric=challenger_cost,
                incumbent_metric=incumbent_cost,
                lower_is_better=True,
            ),
            observed_absolute_improvement=_absolute_reduction(
                challenger_metric=challenger_cost,
                incumbent_metric=incumbent_cost,
            ),
        )

        runtime_rule = RankingRuleEvaluation(
            rule=RankingDecisionRule.RUNTIME_REDUCTION,
            status=_efficiency_rule_status(
                metric_available=challenger_runtime is not None and incumbent_runtime is not None,
                other_dimension_non_regressing=cost_non_regressing,
                score_non_regressing=score_non_regressing,
                metric_passed=_is_meaningfully_faster(
                    candidate_metric=challenger_runtime,
                    incumbent_metric=incumbent_runtime,
                    reduction_required=self._policy.time_reduction_required,
                    min_reduction_ms=TIME_REDUCTION_MIN_MS,
                ),
            ),
            required_relative_improvement=self._policy.time_reduction_required,
            required_absolute_improvement=TIME_REDUCTION_MIN_MS,
            observed_relative_improvement=_relative_improvement(
                challenger_metric=challenger_runtime,
                incumbent_metric=incumbent_runtime,
                lower_is_better=True,
            ),
            observed_absolute_improvement=_absolute_reduction(
                challenger_metric=challenger_runtime,
                incumbent_metric=incumbent_runtime,
            ),
        )

        rules = (score_rule, cost_rule, runtime_rule)
        selected_rule = next(
            (rule.rule for rule in rules if rule.status is RankingRuleStatus.PASSED),
            None,
        )
        return RankingCascadeEvaluation(
            selected_rule=selected_rule,
            score_non_regressing=score_non_regressing,
            rules=rules,
            cost_non_regressing=cost_non_regressing,
            runtime_non_regressing=runtime_non_regressing,
        )

    def _evaluate_score_margin(
        self,
        *,
        incumbent_artifact_id: UUID,
        challenger_total: float,
        incumbent_total: float,
        aggregates: ArtifactAggregateBundle,
    ) -> RankingRuleEvaluation:
        if self._policy.score_margin_kind is _ScoreMarginKind.RELATIVE:
            passed = challenger_total > 0.0 and challenger_total >= incumbent_total * (
                1.0 + self._policy.score_margin_required
            )
            return RankingRuleEvaluation(
                rule=RankingDecisionRule.SCORE_MARGIN,
                status=RankingRuleStatus.PASSED if passed else RankingRuleStatus.FAILED,
                required_relative_improvement=self._policy.score_margin_required,
                observed_relative_improvement=_relative_improvement(
                    challenger_metric=challenger_total,
                    incumbent_metric=incumbent_total,
                    lower_is_better=False,
                ),
            )

        maximum_score = float(len(aggregates.vectors[incumbent_artifact_id]))
        if maximum_score <= 0.0:
            return RankingRuleEvaluation(
                rule=RankingDecisionRule.SCORE_MARGIN,
                status=RankingRuleStatus.UNAVAILABLE,
                required_absolute_improvement=self._policy.score_margin_required,
            )
        required_total = min(
            maximum_score,
            incumbent_total + maximum_score * self._policy.score_margin_required,
        )
        meets_required_total = challenger_total >= required_total or math.isclose(
            challenger_total,
            required_total,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        passed = challenger_total > incumbent_total and meets_required_total
        return RankingRuleEvaluation(
            rule=RankingDecisionRule.SCORE_MARGIN,
            status=RankingRuleStatus.PASSED if passed else RankingRuleStatus.FAILED,
            required_absolute_improvement=_normalize_score(
                max(0.0, required_total - incumbent_total) / maximum_score
            ),
            observed_absolute_improvement=_normalize_score(
                (challenger_total - incumbent_total) / maximum_score
            ),
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
                f"duplicate run pair for validator {row.validator_id}: artifact={row.artifact_id} task={row.task_id}"
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
        artifact_id for validator_vectors in vectors_by_validator.values() for artifact_id in validator_vectors.keys()
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
    return float(details.total_tool_usage.reference_total_cost_usd)


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


def _efficiency_rule_status(
    *,
    metric_available: bool,
    other_dimension_non_regressing: bool | None,
    score_non_regressing: bool,
    metric_passed: bool,
) -> RankingRuleStatus:
    if not metric_available or other_dimension_non_regressing is None:
        return RankingRuleStatus.UNAVAILABLE
    if score_non_regressing and other_dimension_non_regressing and metric_passed:
        return RankingRuleStatus.PASSED
    return RankingRuleStatus.FAILED


def _lower_metric_non_regressing(
    *,
    challenger_metric: float | None,
    incumbent_metric: float | None,
) -> bool | None:
    if challenger_metric is None or incumbent_metric is None:
        return None
    return challenger_metric <= incumbent_metric or math.isclose(
        challenger_metric,
        incumbent_metric,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _relative_improvement(
    *,
    challenger_metric: float | None,
    incumbent_metric: float | None,
    lower_is_better: bool,
) -> float | None:
    if challenger_metric is None or incumbent_metric is None or incumbent_metric <= 0.0:
        return None
    delta = incumbent_metric - challenger_metric if lower_is_better else challenger_metric - incumbent_metric
    return delta / incumbent_metric


def _absolute_reduction(
    *,
    challenger_metric: float | None,
    incumbent_metric: float | None,
) -> float | None:
    if challenger_metric is None or incumbent_metric is None:
        return None
    return incumbent_metric - challenger_metric


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
    "RankingCascadeEvaluation",
    "RankingCascade",
    "RankingCascadeStep",
    "RankingCascadeTrace",
    "RankingDecisionRule",
    "RankingRuleEvaluation",
    "RankingRuleStatus",
    "TIME_REDUCTION_MIN_MS",
    "TIME_REDUCTION_REQUIRED",
    "aggregate_ranking_rows",
    "main_participant_priority",
    "ordered_challengers",
    "run_contribution_score",
    "run_ranking_cost_usd",
    "summarized_run_contribution_score",
]
