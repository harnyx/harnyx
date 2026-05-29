from __future__ import annotations

from uuid import uuid4

import pytest

from harnyx_commons.domain.miner_task import EvaluationDetails, ScoreBreakdown
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.miner_task_emission import compose_champion_weights
from harnyx_commons.miner_task_ranking import (
    ArtifactAggregateBundle,
    ArtifactRankingRow,
    CascadeConfig,
    RankingCascade,
    RankingCascadeStep,
    aggregate_ranking_rows,
    ordered_challengers,
    run_ranking_cost_usd,
)


def test_run_ranking_cost_usd_sums_llm_and_search_tool_costs() -> None:
    details = EvaluationDetails(
        score_breakdown=ScoreBreakdown(comparison_score=1.0, total_score=1.0, scoring_version="test"),
        total_tool_usage=ToolUsageSummary(llm_cost=0.12, search_tool_cost=0.03),
    )

    assert run_ranking_cost_usd(details) == pytest.approx(0.15)


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


def test_ranking_cascade_preserves_incumbent_without_margin_or_efficiency_win() -> None:
    cascade = RankingCascade(CascadeConfig())
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
    cascade = RankingCascade(CascadeConfig())
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


def test_ranking_cascade_trace_records_successful_dethrone_sequence() -> None:
    cascade = RankingCascade(CascadeConfig())
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
        ),
    )

    assert trace.initial_artifact_id == incumbent
    assert trace.final_artifact_id == challenger_b
    assert trace.successful_dethroner_artifact_ids() == (challenger_a, challenger_b)
    assert trace.steps == (
        RankingCascadeStep(
            incumbent_artifact_id=incumbent,
            challenger_artifact_id=challenger_a,
            selected_artifact_id=challenger_a,
            dethroned=True,
        ),
        RankingCascadeStep(
            incumbent_artifact_id=challenger_a,
            challenger_artifact_id=rejected,
            selected_artifact_id=challenger_a,
            dethroned=False,
        ),
        RankingCascadeStep(
            incumbent_artifact_id=challenger_a,
            challenger_artifact_id=challenger_b,
            selected_artifact_id=challenger_b,
            dethroned=True,
        ),
    )


def test_ranking_cascade_trace_treats_positive_replacement_of_zero_incumbent_as_dethrone() -> None:
    cascade = RankingCascade(CascadeConfig())
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
    assert trace.steps == (
        RankingCascadeStep(
            incumbent_artifact_id=incumbent,
            challenger_artifact_id=challenger,
            selected_artifact_id=challenger,
            dethroned=True,
        ),
    )


def test_ordered_challengers_excludes_only_the_incumbent() -> None:
    incumbent = uuid4()
    challenger = uuid4()
    assert ordered_challengers(initial=incumbent, candidate_artifact_ids=[incumbent, challenger]) == [challenger]


def test_compose_champion_weights_returns_winner_take_all() -> None:
    assert compose_champion_weights(5) == {5: 1.0}
    assert compose_champion_weights(None) == {}
