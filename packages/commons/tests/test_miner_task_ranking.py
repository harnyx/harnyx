from __future__ import annotations

from uuid import uuid4

import pytest

from harnyx_commons.domain.miner_task import EvaluationDetails, ScoreBreakdown
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.miner_task_emission import compose_champion_weights
from harnyx_commons.miner_task_ranking import (
    ArtifactAggregateBundle,
    ArtifactRankingRow,
    RankingCascade,
    RankingCascadeStep,
    RankingCascadeTrace,
    RankingDecisionRule,
    RankingRuleStatus,
    aggregate_ranking_rows,
    main_participant_priority,
    ordered_challengers,
    run_ranking_cost_usd,
)


def test_run_ranking_cost_usd_uses_reference_total_with_embedding_cost() -> None:
    details = EvaluationDetails(
        score_breakdown=ScoreBreakdown(comparison_score=1.0, total_score=1.0, scoring_version="test"),
        total_tool_usage=ToolUsageSummary(llm_cost=0.12, search_tool_cost=0.03, embedding_cost=0.05),
    )

    assert run_ranking_cost_usd(details) == pytest.approx(0.20)


def test_aggregate_ranking_rows_uses_per_task_validator_medians() -> None:
    validator_a = uuid4()
    validator_b = uuid4()
    task_a = uuid4()
    task_b = uuid4()
    artifact_1 = uuid4()
    artifact_2 = uuid4()

    bundle = aggregate_ranking_rows(
        (
            ArtifactRankingRow(validator_a, artifact_1, task_a, 0.9, 1.0, 3000.0),
            ArtifactRankingRow(validator_a, artifact_1, task_b, 0.3, 1.0, 2000.0),
            ArtifactRankingRow(validator_a, artifact_2, task_a, 0.4, 2.0, 1000.0),
            ArtifactRankingRow(validator_a, artifact_2, task_b, 0.2, 2.0, 1000.0),
            ArtifactRankingRow(validator_b, artifact_1, task_a, 0.7, 3.0, 4000.0),
            ArtifactRankingRow(validator_b, artifact_1, task_b, 0.5, 3.0, 1000.0),
            ArtifactRankingRow(validator_b, artifact_2, task_a, 0.6, 4.0, 2000.0),
            ArtifactRankingRow(validator_b, artifact_2, task_b, 0.4, 4.0, 3000.0),
        )
    )

    ordered_tasks = sorted((task_a, task_b), key=lambda value: value.hex)
    idx_a = ordered_tasks.index(task_a)
    idx_b = ordered_tasks.index(task_b)

    assert bundle.vectors[artifact_1][idx_a] == 0.8
    assert bundle.vectors[artifact_1][idx_b] == 0.4
    assert bundle.vectors[artifact_2][idx_a] == 0.5
    assert bundle.vectors[artifact_2][idx_b] == pytest.approx(0.3)
    assert bundle.totals == {artifact_1: 1.2, artifact_2: 0.8}
    assert bundle.costs == {artifact_1: 4.0, artifact_2: 6.0}
    assert bundle.median_elapsed_ms[artifact_1] == pytest.approx(5000.0)
    assert bundle.median_elapsed_ms[artifact_2] == pytest.approx(3500.0)


def test_aggregate_ranking_rows_omits_elapsed_when_any_validator_elapsed_is_missing() -> None:
    validator_a = uuid4()
    validator_b = uuid4()
    artifact = uuid4()
    task_a = uuid4()
    task_b = uuid4()

    bundle = aggregate_ranking_rows(
        (
            ArtifactRankingRow(validator_a, artifact, task_a, 0.8, 1.0, 2000.0),
            ArtifactRankingRow(validator_a, artifact, task_b, 0.8, 1.0, 2500.0),
            ArtifactRankingRow(validator_b, artifact, task_a, 0.8, 1.0, None),
            ArtifactRankingRow(validator_b, artifact, task_b, 0.8, 1.0, 1500.0),
        )
    )

    assert bundle.totals[artifact] == pytest.approx(1.6)
    assert artifact not in bundle.median_elapsed_ms


@pytest.mark.parametrize(
    ("task_count", "incumbent_total", "challenger_total", "expected_status"),
    (
        (10, 5.0, 5.999, RankingRuleStatus.FAILED),
        (10, 5.0, 6.0, RankingRuleStatus.PASSED),
        (10, 3.000004, 4.000004, RankingRuleStatus.PASSED),
        (30, 15.0, 17.999, RankingRuleStatus.FAILED),
        (30, 15.0, 18.0, RankingRuleStatus.PASSED),
        (30, 13.000002, 16.000002, RankingRuleStatus.PASSED),
        (10, 9.5, 10.0, RankingRuleStatus.PASSED),
        (10, 10.0, 10.0, RankingRuleStatus.FAILED),
    ),
)
def test_current_ranking_cascade_uses_ceiling_aware_ten_percentage_point_score_margin(
    task_count: int,
    incumbent_total: float,
    challenger_total: float,
    expected_status: RankingRuleStatus,
) -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger = uuid4()

    trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=(challenger,),
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.0] * task_count, challenger: [0.0] * task_count},
            totals={incumbent: incumbent_total, challenger: challenger_total},
            costs={incumbent: 10.0, challenger: 10.0},
            median_elapsed_ms={incumbent: 10_000.0, challenger: 10_000.0},
        ),
    )

    evaluation = trace.steps[0].evaluation
    assert evaluation is not None
    score_rule = evaluation.rules[0]
    assert score_rule.status is expected_status
    assert score_rule.required_relative_improvement is None
    assert score_rule.required_absolute_improvement == pytest.approx(
        min(0.10, (task_count - incumbent_total) / task_count)
    )
    assert score_rule.observed_absolute_improvement == pytest.approx((challenger_total - incumbent_total) / task_count)
    assert trace.final_artifact_id == (challenger if expected_status is RankingRuleStatus.PASSED else incumbent)


@pytest.mark.parametrize(
    ("challenger_cost", "expected_status"),
    (
        (90.001, RankingRuleStatus.FAILED),
        (90.0, RankingRuleStatus.PASSED),
    ),
)
def test_current_ranking_cascade_cost_reduction_requires_ten_percent(
    challenger_cost: float,
    expected_status: RankingRuleStatus,
) -> None:
    incumbent = uuid4()
    challenger = uuid4()

    trace = RankingCascade().trace(
        initial=incumbent,
        challengers_ordered=(challenger,),
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5], challenger: [0.5]},
            totals={incumbent: 0.5, challenger: 0.5},
            costs={incumbent: 100.0, challenger: challenger_cost},
            median_elapsed_ms={incumbent: 20_000.0, challenger: 20_000.0},
        ),
    )

    evaluation = trace.steps[0].evaluation
    assert evaluation is not None
    assert evaluation.rules[1].status is expected_status
    assert evaluation.rules[1].required_relative_improvement == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("incumbent_runtime", "challenger_runtime", "expected_status"),
    (
        (20_000.0, 19_000.0, RankingRuleStatus.FAILED),
        (20_000.0, 18_000.0, RankingRuleStatus.PASSED),
        (5_000.0, 4_500.0, RankingRuleStatus.FAILED),
        (5_000.0, 4_000.0, RankingRuleStatus.PASSED),
    ),
)
def test_current_ranking_cascade_runtime_reduction_binds_relative_and_absolute_requirements(
    incumbent_runtime: float,
    challenger_runtime: float,
    expected_status: RankingRuleStatus,
) -> None:
    incumbent = uuid4()
    challenger = uuid4()

    trace = RankingCascade().trace(
        initial=incumbent,
        challengers_ordered=(challenger,),
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5], challenger: [0.5]},
            totals={incumbent: 0.5, challenger: 0.5},
            costs={incumbent: 10.0, challenger: 10.0},
            median_elapsed_ms={incumbent: incumbent_runtime, challenger: challenger_runtime},
        ),
    )

    evaluation = trace.steps[0].evaluation
    assert evaluation is not None
    runtime_rule = evaluation.rules[2]
    assert runtime_rule.status is expected_status
    assert runtime_rule.required_relative_improvement == pytest.approx(0.10)
    assert runtime_rule.required_absolute_improvement == pytest.approx(1_000.0)


def test_historical_ranking_cascade_retains_relative_twenty_percent_margins() -> None:
    incumbent = uuid4()
    challenger = uuid4()

    trace = RankingCascade.for_historical_batches().trace(
        initial=incumbent,
        challengers_ordered=(challenger,),
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.8], challenger: [0.9]},
            totals={incumbent: 0.8, challenger: 0.9},
            costs={incumbent: 100.0, challenger: 85.0},
            median_elapsed_ms={incumbent: 20_000.0, challenger: 17_000.0},
        ),
    )

    evaluation = trace.steps[0].evaluation
    assert evaluation is not None
    assert evaluation.selected_rule is None
    assert [rule.required_relative_improvement for rule in evaluation.rules] == [0.20, 0.20, 0.20]
    assert evaluation.rules[0].required_absolute_improvement is None
    assert trace.final_artifact_id == incumbent


def test_ranking_cascade_preserves_incumbent_without_margin_or_efficiency_win() -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger = uuid4()

    champion = cascade.decide(
        initial=incumbent,
        challengers_ordered=[challenger],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.6, 0.6], challenger: [0.65, 0.61]},
            totals={incumbent: 1.2, challenger: 1.26},
            costs={incumbent: 10.0, challenger: 10.0},
            median_elapsed_ms={incumbent: 4000.0, challenger: 4000.0},
        ),
    )

    assert champion == incumbent


def test_ranking_cascade_dethrones_on_non_regressing_score_and_efficiency() -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger = uuid4()

    champion = cascade.decide(
        initial=incumbent,
        challengers_ordered=[challenger],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5, 0.5], challenger: [0.5, 0.5]},
            totals={incumbent: 1.0, challenger: 1.0},
            costs={incumbent: 10.0, challenger: 7.5},
            median_elapsed_ms={incumbent: 10000.0, challenger: 7000.0},
        ),
    )

    assert champion == challenger


def test_ranking_cascade_score_margin_is_independent_of_efficiency() -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    worse_efficiency = uuid4()
    missing_efficiency = uuid4()

    worse_trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[worse_efficiency],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5, 0.5], worse_efficiency: [0.6, 0.6]},
            totals={incumbent: 1.0, worse_efficiency: 1.2},
            costs={incumbent: 10.0, worse_efficiency: 12.0},
            median_elapsed_ms={incumbent: 10_000.0, worse_efficiency: 12_000.0},
        ),
    )
    missing_trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[missing_efficiency],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5, 0.5], missing_efficiency: [0.6, 0.6]},
            totals={incumbent: 1.0, missing_efficiency: 1.2},
            costs={incumbent: 10.0},
            median_elapsed_ms={incumbent: 10_000.0},
        ),
    )

    assert worse_trace.final_artifact_id == worse_efficiency
    assert worse_trace.steps[0].evaluation is not None
    assert worse_trace.steps[0].evaluation.selected_rule is RankingDecisionRule.SCORE_MARGIN
    assert missing_trace.final_artifact_id == missing_efficiency
    assert missing_trace.steps[0].evaluation is not None
    assert missing_trace.steps[0].evaluation.selected_rule is RankingDecisionRule.SCORE_MARGIN


@pytest.mark.parametrize(
    (
        "challenger_runtime",
        "expected_non_regressing",
        "expected_status",
        "expected_champion",
    ),
    (
        (10_000.0, True, RankingRuleStatus.PASSED, "challenger"),
        (7_000.0, True, RankingRuleStatus.PASSED, "challenger"),
        (10_000.000001, True, RankingRuleStatus.PASSED, "challenger"),
        (11_000.0, False, RankingRuleStatus.FAILED, "incumbent"),
        (None, None, RankingRuleStatus.UNAVAILABLE, "incumbent"),
    ),
)
def test_ranking_cascade_cost_reduction_requires_runtime_non_regression(
    challenger_runtime: float | None,
    expected_non_regressing: bool | None,
    expected_status: RankingRuleStatus,
    expected_champion: str,
) -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger = uuid4()
    runtimes = {incumbent: 10_000.0}
    if challenger_runtime is not None:
        runtimes[challenger] = challenger_runtime

    trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[challenger],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5, 0.5], challenger: [0.5, 0.5]},
            totals={incumbent: 1.0, challenger: 1.0},
            costs={incumbent: 10.0, challenger: 7.0},
            median_elapsed_ms=runtimes,
        ),
    )

    evaluation = trace.steps[0].evaluation
    assert evaluation is not None
    assert evaluation.rules[1].status is expected_status
    assert evaluation.cost_non_regressing is True
    assert evaluation.runtime_non_regressing is expected_non_regressing
    assert trace.final_artifact_id == (challenger if expected_champion == "challenger" else incumbent)


@pytest.mark.parametrize(
    (
        "challenger_cost",
        "expected_non_regressing",
        "expected_status",
        "expected_champion",
    ),
    (
        (10.0, True, RankingRuleStatus.PASSED, "challenger"),
        (7.0, True, RankingRuleStatus.PASSED, "challenger"),
        (10.000000001, True, RankingRuleStatus.PASSED, "challenger"),
        (11.0, False, RankingRuleStatus.FAILED, "incumbent"),
        (None, None, RankingRuleStatus.UNAVAILABLE, "incumbent"),
    ),
)
def test_ranking_cascade_runtime_reduction_requires_cost_non_regression(
    challenger_cost: float | None,
    expected_non_regressing: bool | None,
    expected_status: RankingRuleStatus,
    expected_champion: str,
) -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger = uuid4()
    costs = {incumbent: 10.0}
    if challenger_cost is not None:
        costs[challenger] = challenger_cost

    trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[challenger],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5, 0.5], challenger: [0.5, 0.5]},
            totals={incumbent: 1.0, challenger: 1.0},
            costs=costs,
            median_elapsed_ms={incumbent: 10_000.0, challenger: 7_000.0},
        ),
    )

    evaluation = trace.steps[0].evaluation
    assert evaluation is not None
    assert evaluation.rules[2].status is expected_status
    assert evaluation.cost_non_regressing is expected_non_regressing
    assert evaluation.runtime_non_regressing is True
    assert trace.final_artifact_id == (challenger if expected_champion == "challenger" else incumbent)


def test_ranking_cascade_alternating_efficiency_dethrones_do_not_regress_either_dimension() -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    faster_challenger = uuid4()
    cheaper_challenger = uuid4()

    trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[faster_challenger, cheaper_challenger],
        aggregates=ArtifactAggregateBundle(
            vectors={
                incumbent: [0.5, 0.5],
                faster_challenger: [0.5, 0.5],
                cheaper_challenger: [0.5, 0.5],
            },
            totals={incumbent: 1.0, faster_challenger: 1.0, cheaper_challenger: 1.0},
            costs={incumbent: 10.0, faster_challenger: 10.0, cheaper_challenger: 7.0},
            median_elapsed_ms={
                incumbent: 10_000.0,
                faster_challenger: 7_000.0,
                cheaper_challenger: 7_000.0,
            },
        ),
    )

    assert trace.final_artifact_id == cheaper_challenger
    assert [step.evaluation.selected_rule if step.evaluation else None for step in trace.steps] == [
        RankingDecisionRule.RUNTIME_REDUCTION,
        RankingDecisionRule.COST_REDUCTION,
    ]
    assert [
        (
            step.evaluation.cost_non_regressing,
            step.evaluation.runtime_non_regressing,
        )
        for step in trace.steps
        if step.evaluation is not None
    ] == [(True, True), (True, True)]


def test_ranking_cascade_trace_records_successful_dethrone_sequence() -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger_a = uuid4()
    challenger_b = uuid4()
    rejected = uuid4()

    trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[challenger_a, rejected, challenger_b],
        aggregates=ArtifactAggregateBundle(
            vectors={
                incumbent: [0.5, 0.5],
                challenger_a: [0.7, 0.5],
                rejected: [0.6, 0.4],
                challenger_b: [0.7, 0.5],
            },
            totals={
                incumbent: 1.0,
                challenger_a: 1.2,
                rejected: 1.0,
                challenger_b: 1.2,
            },
            costs={
                incumbent: 10.0,
                challenger_a: 10.0,
                rejected: 1.0,
                challenger_b: 7.0,
            },
            median_elapsed_ms={
                incumbent: 10_000.0,
                challenger_a: 10_000.0,
                rejected: 9_000.0,
                challenger_b: 10_000.0,
            },
        ),
    )

    assert trace.initial_artifact_id == incumbent
    assert trace.final_artifact_id == challenger_b
    assert trace.successful_dethroner_artifact_ids() == (challenger_a, challenger_b)
    assert [step.incumbent_artifact_id for step in trace.steps] == [
        incumbent,
        challenger_a,
        challenger_a,
    ]
    assert [step.selected_artifact_id for step in trace.steps] == [
        challenger_a,
        challenger_a,
        challenger_b,
    ]
    assert [step.dethroned for step in trace.steps] == [True, False, True]
    assert [step.evaluation.selected_rule if step.evaluation else None for step in trace.steps] == [
        RankingDecisionRule.SCORE_MARGIN,
        None,
        RankingDecisionRule.COST_REDUCTION,
    ]
    assert trace.steps[0].evaluation is not None
    assert trace.steps[0].evaluation.rules[0].observed_relative_improvement is None
    assert trace.steps[0].evaluation.rules[0].observed_absolute_improvement == pytest.approx(0.1)
    assert trace.steps[1].evaluation is not None
    assert trace.steps[1].evaluation.score_non_regressing is False
    assert trace.steps[1].evaluation.rules[1].status is RankingRuleStatus.FAILED
    assert trace.steps[2].evaluation is not None
    assert trace.steps[2].evaluation.rules[1].observed_relative_improvement == pytest.approx(0.3)


def test_ranking_cascade_trace_records_championless_champion_lineage() -> None:
    provisional_champion = uuid4()
    rejected = uuid4()
    final_champion = uuid4()
    trace = RankingCascadeTrace(
        initial_artifact_id=None,
        final_artifact_id=final_champion,
        steps=(
            RankingCascadeStep(None, provisional_champion, provisional_champion, False),
            RankingCascadeStep(provisional_champion, rejected, provisional_champion, False),
            RankingCascadeStep(provisional_champion, final_champion, final_champion, True),
        ),
    )

    assert trace.champion_lineage_artifact_ids() == (
        provisional_champion,
        final_champion,
    )
    assert trace.reverse_champion_lineage() == (
        final_champion,
        provisional_champion,
    )
    assert main_participant_priority(
        main_trace=trace,
        qualifying_participant_order=(provisional_champion, final_champion),
    ) == (
        final_champion,
        provisional_champion,
    )


def test_ranking_cascade_trace_treats_positive_replacement_of_zero_incumbent_as_dethrone() -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger = uuid4()

    trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[challenger],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.0], challenger: [0.8]},
            totals={incumbent: 0.0, challenger: 0.8},
            costs={incumbent: 1.0, challenger: 1.0},
        ),
    )

    assert trace.initial_artifact_id == incumbent
    assert trace.final_artifact_id == challenger
    assert trace.successful_dethroner_artifact_ids() == (challenger,)
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.incumbent_artifact_id == incumbent
    assert step.challenger_artifact_id == challenger
    assert step.selected_artifact_id == challenger
    assert step.dethroned is True
    assert step.evaluation is not None
    assert step.evaluation.selected_rule is RankingDecisionRule.POSITIVE_SCORE
    assert step.evaluation.score_non_regressing is None
    assert step.evaluation.rules[0].status is RankingRuleStatus.PASSED
    assert [rule.status for rule in step.evaluation.rules[1:]] == [
        RankingRuleStatus.NOT_APPLICABLE,
        RankingRuleStatus.NOT_APPLICABLE,
        RankingRuleStatus.NOT_APPLICABLE,
    ]


def test_ranking_cascade_trace_marks_missing_runtime_unavailable() -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger = uuid4()

    trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[challenger],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5], challenger: [0.5]},
            totals={incumbent: 0.5, challenger: 0.5},
            costs={incumbent: 1.0, challenger: 1.0},
        ),
    )

    evaluation = trace.steps[0].evaluation
    assert evaluation is not None
    assert evaluation.selected_rule is None
    assert evaluation.cost_non_regressing is True
    assert evaluation.runtime_non_regressing is None
    assert evaluation.rules[1].status is RankingRuleStatus.UNAVAILABLE
    assert evaluation.rules[2].rule is RankingDecisionRule.RUNTIME_REDUCTION
    assert evaluation.rules[2].status is RankingRuleStatus.UNAVAILABLE


def test_ranking_cascade_trace_preserves_rule_thresholds_and_precedence() -> None:
    cascade = RankingCascade()
    incumbent = uuid4()
    challenger = uuid4()

    trace = cascade.trace(
        initial=incumbent,
        challengers_ordered=[challenger],
        aggregates=ArtifactAggregateBundle(
            vectors={incumbent: [0.5], challenger: [0.5]},
            totals={incumbent: 0.5, challenger: 0.5},
            costs={incumbent: 10.0, challenger: 8.0},
            median_elapsed_ms={incumbent: 5000.0, challenger: 4000.0},
        ),
    )

    evaluation = trace.steps[0].evaluation
    assert evaluation is not None
    assert evaluation.selected_rule is RankingDecisionRule.COST_REDUCTION
    assert evaluation.score_non_regressing is True
    assert [rule.status for rule in evaluation.rules] == [
        RankingRuleStatus.FAILED,
        RankingRuleStatus.PASSED,
        RankingRuleStatus.PASSED,
    ]
    assert evaluation.rules[1].observed_relative_improvement == pytest.approx(0.2)
    assert evaluation.rules[2].observed_relative_improvement == pytest.approx(0.2)
    assert evaluation.rules[2].observed_absolute_improvement == pytest.approx(1000.0)


def test_ordered_challengers_excludes_only_the_incumbent() -> None:
    incumbent = uuid4()
    challenger = uuid4()
    assert ordered_challengers(initial=incumbent, candidate_artifact_ids=[incumbent, challenger]) == [challenger]


def test_compose_champion_weights_returns_winner_take_all() -> None:
    assert compose_champion_weights(5) == {5: 1.0}
    assert compose_champion_weights(None) == {}


def test_main_participant_priority_uses_reverse_main_lineage_then_reverse_qualifying_order() -> None:
    incumbent = uuid4()
    dethroner_a = uuid4()
    dethroner_b = uuid4()
    non_dethroner = uuid4()
    trace = RankingCascadeTrace(
        initial_artifact_id=incumbent,
        final_artifact_id=dethroner_b,
        steps=(
            RankingCascadeStep(incumbent, dethroner_a, dethroner_a, True),
            RankingCascadeStep(dethroner_a, non_dethroner, dethroner_a, False),
            RankingCascadeStep(dethroner_a, dethroner_b, dethroner_b, True),
        ),
    )

    assert main_participant_priority(
        main_trace=trace,
        qualifying_participant_order=(incumbent, dethroner_a, dethroner_b, non_dethroner),
    ) == (dethroner_b, dethroner_a, incumbent, non_dethroner)


def test_main_participant_priority_uses_reverse_qualifying_order_without_final_champion() -> None:
    incumbent = uuid4()
    challenger_a = uuid4()
    challenger_b = uuid4()
    trace = RankingCascadeTrace(
        initial_artifact_id=incumbent,
        final_artifact_id=None,
        steps=(
            RankingCascadeStep(None, challenger_a, None, False),
            RankingCascadeStep(None, challenger_b, None, False),
        ),
    )

    assert trace.champion_lineage_artifact_ids() == (incumbent,)
    assert trace.reverse_champion_lineage() == ()
    assert main_participant_priority(
        main_trace=trace,
        qualifying_participant_order=(incumbent, challenger_a, challenger_b),
    ) == (challenger_b, challenger_a, incumbent)
