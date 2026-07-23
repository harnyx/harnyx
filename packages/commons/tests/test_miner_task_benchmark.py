from __future__ import annotations

from uuid import UUID

import pytest

from harnyx_commons.miner_task_benchmark import (
    BENCHMARK_CORRECTNESS_SCORING_VERSION,
    BENCHMARK_SAMPLE_SIZE,
    BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    SUPPORTED_BENCHMARK_SCORING_VERSION,
    SUPPORTED_BENCHMARK_SCORING_VERSIONS,
    BenchmarkCorrectnessScoringConfig,
    BenchmarkItemOutcome,
    BenchmarkItemState,
    BenchmarkRunState,
    aggregate_benchmark_metrics,
    is_defined_benchmark_scoring_version,
    is_supported_benchmark_scoring_version,
    project_benchmark_run_state,
    sample_benchmark_items,
    unsupported_benchmark_scoring_version_error,
)


class _SampleItem:
    def __init__(self, item_index: int) -> None:
        self.item_index = item_index


def test_benchmark_correctness_scoring_config_default_timeout_is_300_seconds() -> None:
    config = BenchmarkCorrectnessScoringConfig(provider="chutes", model="judge-model")

    assert config.timeout_seconds == pytest.approx(300.0)


def test_aggregate_benchmark_metrics_scores_terminal_items_only() -> None:
    metrics = aggregate_benchmark_metrics(
        (
            BenchmarkItemOutcome(state=BenchmarkItemState.COMPLETED, is_correct=True),
            BenchmarkItemOutcome(state=BenchmarkItemState.COMPLETED, is_correct=False),
            BenchmarkItemOutcome(state=BenchmarkItemState.FAILED, is_correct=None),
        )
    )

    assert metrics.completed_item_count == 2
    assert metrics.failed_item_count == 1
    assert metrics.correct_item_count == 1
    assert metrics.mean_total_score == pytest.approx(1 / 3)
    assert metrics.derive_state() is BenchmarkRunState.PARTIAL_SUCCESS


def test_aggregate_benchmark_metrics_uses_numeric_scores_when_present() -> None:
    metrics = aggregate_benchmark_metrics(
        (
            BenchmarkItemOutcome(
                state=BenchmarkItemState.COMPLETED,
                is_correct=None,
                score=0.75,
            ),
            BenchmarkItemOutcome(
                state=BenchmarkItemState.COMPLETED,
                is_correct=None,
                score=0.25,
            ),
            BenchmarkItemOutcome(state=BenchmarkItemState.FAILED, is_correct=None),
        )
    )

    assert metrics.completed_item_count == 2
    assert metrics.failed_item_count == 1
    assert metrics.correct_item_count is None
    assert metrics.mean_total_score == pytest.approx((0.75 + 0.25) / 3)
    assert metrics.derive_state() is BenchmarkRunState.PARTIAL_SUCCESS


def test_aggregate_benchmark_metrics_rejects_numeric_scores_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="score"):
        aggregate_benchmark_metrics(
            (
                BenchmarkItemOutcome(
                    state=BenchmarkItemState.COMPLETED,
                    is_correct=None,
                    score=1.01,
                ),
            )
        )


def test_aggregate_benchmark_metrics_rejects_mixed_completed_scoring_modes() -> None:
    with pytest.raises(ValueError, match="mixed"):
        aggregate_benchmark_metrics(
            (
                BenchmarkItemOutcome(state=BenchmarkItemState.COMPLETED, is_correct=True),
                BenchmarkItemOutcome(
                    state=BenchmarkItemState.COMPLETED,
                    is_correct=None,
                    score=0.5,
                ),
            )
        )


def test_aggregate_benchmark_metrics_rejects_single_item_with_numeric_and_binary_score() -> None:
    with pytest.raises(ValueError, match="mixed"):
        aggregate_benchmark_metrics(
            (
                BenchmarkItemOutcome(
                    state=BenchmarkItemState.COMPLETED,
                    is_correct=True,
                    score=0.5,
                ),
            )
        )


def test_aggregate_benchmark_metrics_keeps_score_empty_without_terminal_items() -> None:
    metrics = aggregate_benchmark_metrics(
        (
            BenchmarkItemOutcome(state=BenchmarkItemState.QUEUED, is_correct=None),
            BenchmarkItemOutcome(state=BenchmarkItemState.RUNNING, is_correct=None),
        )
    )

    assert metrics.queued_item_count == 1
    assert metrics.running_item_count == 1
    assert metrics.completed_item_count == 0
    assert metrics.failed_item_count == 0
    assert metrics.mean_total_score is None
    assert metrics.derive_state() is BenchmarkRunState.RUNNING


def test_defined_benchmark_scoring_versions_include_correctness_and_weighted_rubric() -> None:
    assert is_defined_benchmark_scoring_version(BENCHMARK_CORRECTNESS_SCORING_VERSION)
    assert is_defined_benchmark_scoring_version(BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION)
    assert not is_defined_benchmark_scoring_version("other")


def test_runtime_supported_benchmark_scoring_versions_include_weighted_rubric() -> None:
    assert is_supported_benchmark_scoring_version(BENCHMARK_CORRECTNESS_SCORING_VERSION)
    assert is_supported_benchmark_scoring_version(BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION)
    assert SUPPORTED_BENCHMARK_SCORING_VERSION == BENCHMARK_CORRECTNESS_SCORING_VERSION
    assert SUPPORTED_BENCHMARK_SCORING_VERSIONS == frozenset(
        {
            BENCHMARK_CORRECTNESS_SCORING_VERSION,
            BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
        }
    )


def test_project_benchmark_run_state_marks_terminal_unfinished_run_failed() -> None:
    metrics = aggregate_benchmark_metrics(
        (BenchmarkItemOutcome(state=BenchmarkItemState.QUEUED, is_correct=None),)
    )

    assert (
        project_benchmark_run_state(metrics=metrics, backing_batch_is_terminal=True)
        is BenchmarkRunState.FAILED
    )


def test_project_benchmark_run_state_marks_terminal_mixed_run_partial_success() -> None:
    metrics = aggregate_benchmark_metrics(
        (
            BenchmarkItemOutcome(state=BenchmarkItemState.COMPLETED, is_correct=True),
            BenchmarkItemOutcome(state=BenchmarkItemState.QUEUED, is_correct=None),
        )
    )

    assert (
        project_benchmark_run_state(metrics=metrics, backing_batch_is_terminal=True)
        is BenchmarkRunState.PARTIAL_SUCCESS
    )


def test_sample_benchmark_items_is_fixed_across_run_ids_and_sorted_by_item_index() -> None:
    items = tuple(_SampleItem(index) for index in range(BENCHMARK_SAMPLE_SIZE + 5))

    first = sample_benchmark_items(
        items=items,
        run_id=UUID("00000000-0000-4000-8000-000000000001"),
        dataset_version="dataset-v1",
        scoring_version="correctness-v1",
    )
    second = sample_benchmark_items(
        items=items,
        run_id=UUID("00000000-0000-4000-8000-000000000002"),
        dataset_version="dataset-v1",
        scoring_version="correctness-v1",
    )

    assert len(first) == BENCHMARK_SAMPLE_SIZE
    assert first == second
    assert tuple(item.item_index for item in first) == tuple(sorted(item.item_index for item in first))


def test_unsupported_benchmark_scoring_version_error_names_expected_version() -> None:
    exc = unsupported_benchmark_scoring_version_error("other")

    assert "unsupported benchmark scoring_version 'other'" in str(exc)
    assert "correctness-v1" in str(exc)
    assert "weighted-rubric-v1" in str(exc)
