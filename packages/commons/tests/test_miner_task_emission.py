from __future__ import annotations

from math import fsum
from uuid import UUID

import pytest

from harnyx_commons.miner_task_emission import (
    OWNER_UID,
    ParticipantEmissionScore,
    apply_miner_emission_cap,
    compose_artifact_participant_distribution_weights,
    compose_emission_weights,
    compose_equal_participant_emission_allocations,
    compose_flat_participant_emission_allocations,
    compose_novelty_distribution_weights,
    compose_novelty_emission_allocations,
    compose_participant_emission_weights,
    compose_prioritized_emission,
    compose_tiered_participant_emission_allocations,
    compose_weighted_participant_emission_allocations,
    owner_fallback_weights,
    participant_emission_fraction,
    select_participant_emission_scores,
)


def test_apply_miner_emission_cap_defaults_can_burn_all_miner_emission() -> None:
    weights = apply_miner_emission_cap(
        {7: 0.6, 8: 0.4},
        batch_score=0.5,
        max_miner_emission_fraction=0.0,
    )

    assert weights == {
        0: pytest.approx(1.0),
        7: pytest.approx(0.0),
        8: pytest.approx(0.0),
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_apply_miner_emission_cap_scales_by_configured_max_fraction() -> None:
    weights = apply_miner_emission_cap(
        {7: 0.6, 8: 0.4},
        batch_score=0.5,
        max_miner_emission_fraction=0.2,
    )

    assert weights == {
        0: pytest.approx(0.9),
        7: pytest.approx(0.06),
        8: pytest.approx(0.04),
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_apply_miner_emission_cap_ignores_owner_weight_in_base_vector() -> None:
    weights = apply_miner_emission_cap(
        {0: 0.8, 7: 0.6, 8: 0.4},
        batch_score=1.0,
        max_miner_emission_fraction=0.2,
    )

    assert weights == {
        0: pytest.approx(0.8),
        7: pytest.approx(0.12),
        8: pytest.approx(0.08),
    }


@pytest.mark.parametrize("weights", [{}, {0: 1.0}])
def test_apply_miner_emission_cap_rejects_empty_miner_weights(weights: dict[int, float]) -> None:
    with pytest.raises(ValueError, match="miner weights are empty"):
        apply_miner_emission_cap(weights, batch_score=1.0, max_miner_emission_fraction=0.0)


def test_apply_miner_emission_cap_rejects_non_positive_miner_total() -> None:
    with pytest.raises(ValueError, match="miner weights must have positive miner total"):
        apply_miner_emission_cap({7: 0.0, 8: 0.0}, batch_score=1.0, max_miner_emission_fraction=0.0)


@pytest.mark.parametrize("batch_score", [-0.1, 1.1, float("nan")])
def test_apply_miner_emission_cap_rejects_invalid_batch_score(batch_score: float) -> None:
    with pytest.raises(ValueError, match="miner task batch score must be between 0.0 and 1.0"):
        apply_miner_emission_cap({7: 1.0}, batch_score=batch_score, max_miner_emission_fraction=0.0)


@pytest.mark.parametrize("max_fraction", [-0.1, 1.1, float("nan")])
def test_apply_miner_emission_cap_rejects_invalid_max_fraction(max_fraction: float) -> None:
    with pytest.raises(ValueError, match="max miner emission fraction must be between 0.0 and 1.0"):
        apply_miner_emission_cap(
            {7: 1.0},
            batch_score=1.0,
            max_miner_emission_fraction=max_fraction,
        )


def test_owner_uid_weights_burn_all_miner_emission() -> None:
    assert owner_fallback_weights() == {0: 1.0}


def test_participant_emission_pays_fixed_weight_per_registered_uid() -> None:
    weights = compose_participant_emission_weights((10, 11))

    assert weights == {
        10: pytest.approx(0.004),
        11: pytest.approx(0.004),
    }


def test_participant_emission_uses_configured_weight_per_registered_uid() -> None:
    weights = compose_participant_emission_weights(
        (10, 11),
        miner_participation_emission=0.01,
    )

    assert weights == {
        10: pytest.approx(0.01),
        11: pytest.approx(0.01),
    }


def test_participant_emission_empty_participants_returns_empty_component() -> None:
    assert compose_participant_emission_weights(()) == {}


def test_participant_emission_deduplicates_registered_uids() -> None:
    assert compose_participant_emission_weights((10, 10)) == {
        10: pytest.approx(0.004),
    }


def test_participant_emission_stops_before_total_weight_overflow() -> None:
    weights = compose_participant_emission_weights(tuple(range(1, 252)))

    assert len(weights) == 250
    assert weights[250] == pytest.approx(0.004)
    assert 251 not in weights
    assert fsum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("raw_value", [-0.1, 1.1, float("nan")])
def test_participant_emission_rejects_invalid_configured_weight(raw_value: float) -> None:
    with pytest.raises(ValueError, match="miner participation emission must be between 0.0 and 1.0"):
        compose_participant_emission_weights((10,), miner_participation_emission=raw_value)


def test_participant_emission_fraction_uses_configured_weight() -> None:
    assert participant_emission_fraction(3, miner_participation_emission=0.01) == pytest.approx(0.03)


def test_participant_emission_fraction_stops_before_partial_overflow() -> None:
    assert participant_emission_fraction(4, miner_participation_emission=0.3) == pytest.approx(0.9)


def test_flat_participant_emission_allocations_stop_in_input_order_before_overflow() -> None:
    weights = compose_flat_participant_emission_allocations(
        ("hotkey-a", "hotkey-b", "hotkey-c", "hotkey-d"),
        miner_participation_emission=0.3,
    )

    assert weights == {
        "hotkey-a": pytest.approx(0.3),
        "hotkey-b": pytest.approx(0.3),
        "hotkey-c": pytest.approx(0.3),
    }
    assert fsum(weights.values()) == pytest.approx(0.9)


def test_tiered_participant_emission_uses_ceil_10_and_50_thresholds() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore("hotkey-a", 1.0),
            ParticipantEmissionScore("hotkey-b", 0.9),
            ParticipantEmissionScore("hotkey-c", 0.8),
            ParticipantEmissionScore("hotkey-d", 0.7),
            ParticipantEmissionScore("hotkey-e", 0.6),
            ParticipantEmissionScore("hotkey-f", 0.5),
            ParticipantEmissionScore("hotkey-g", 0.4),
            ParticipantEmissionScore("hotkey-h", 0.3),
            ParticipantEmissionScore("hotkey-i", 0.2),
            ParticipantEmissionScore("hotkey-j", 0.1),
        ),
        miner_participation_emission=0.01,
    )

    assert weights == {
        "hotkey-a": pytest.approx(0.02),
        "hotkey-b": pytest.approx(0.01),
        "hotkey-c": pytest.approx(0.01),
        "hotkey-d": pytest.approx(0.01),
        "hotkey-e": pytest.approx(0.01),
    }


def test_tiered_participant_emission_expands_ties_at_boundaries() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore("hotkey-a", 1.0),
            ParticipantEmissionScore("hotkey-b", 1.0),
            ParticipantEmissionScore("hotkey-c", 0.9),
            ParticipantEmissionScore("hotkey-d", 0.8),
            ParticipantEmissionScore("hotkey-e", 0.7),
            ParticipantEmissionScore("hotkey-f", 0.7),
            ParticipantEmissionScore("hotkey-g", 0.6),
            ParticipantEmissionScore("hotkey-h", 0.5),
            ParticipantEmissionScore("hotkey-i", 0.4),
            ParticipantEmissionScore("hotkey-j", 0.3),
        ),
        miner_participation_emission=0.01,
    )

    assert weights == {
        "hotkey-a": pytest.approx(0.02),
        "hotkey-b": pytest.approx(0.02),
        "hotkey-c": pytest.approx(0.01),
        "hotkey-d": pytest.approx(0.01),
        "hotkey-e": pytest.approx(0.01),
        "hotkey-f": pytest.approx(0.01),
    }


def test_tiered_participant_emission_excludes_zero_scores_from_allocations() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore("hotkey-a", 1.0),
            ParticipantEmissionScore("hotkey-b", 0.5),
            ParticipantEmissionScore("hotkey-c", 0.0),
        ),
        miner_participation_emission=0.01,
    )

    assert weights == {
        "hotkey-a": pytest.approx(0.02),
        "hotkey-b": pytest.approx(0.01),
    }


def test_tiered_participant_emission_counts_zero_scores_in_cutoff_population() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore("hotkey-a", 1.0),
            ParticipantEmissionScore("hotkey-b", 0.5),
            ParticipantEmissionScore("hotkey-c", 0.0),
            ParticipantEmissionScore("hotkey-d", 0.0),
            ParticipantEmissionScore("hotkey-e", 0.0),
            ParticipantEmissionScore("hotkey-f", 0.0),
            ParticipantEmissionScore("hotkey-g", 0.0),
            ParticipantEmissionScore("hotkey-h", 0.0),
            ParticipantEmissionScore("hotkey-i", 0.0),
            ParticipantEmissionScore("hotkey-j", 0.0),
        ),
        miner_participation_emission=0.01,
    )

    assert weights == {
        "hotkey-a": pytest.approx(0.02),
        "hotkey-b": pytest.approx(0.01),
    }


def test_tiered_participant_emission_uses_hotkey_tie_breaker_before_capping() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore("hotkey-c", 1.0),
            ParticipantEmissionScore("hotkey-b", 1.0),
            ParticipantEmissionScore("hotkey-a", 1.0),
        ),
        miner_participation_emission=0.34,
    )

    assert weights == {"hotkey-a": pytest.approx(0.68)}


def test_tiered_participant_emission_all_zero_scores_return_empty_allocations() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore("hotkey-a", 0.0),
            ParticipantEmissionScore("hotkey-b", 0.0),
        ),
        miner_participation_emission=0.01,
    )

    assert weights == {}


def test_tiered_participant_emission_deduplicates_participant_key_by_higher_score() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore("hotkey-a", 0.2),
            ParticipantEmissionScore("hotkey-a", 0.9),
            ParticipantEmissionScore("hotkey-b", 0.8),
            ParticipantEmissionScore("hotkey-c", 0.1),
        ),
        miner_participation_emission=0.01,
    )

    assert weights["hotkey-a"] == pytest.approx(0.02)


def test_tiered_participant_emission_assigns_same_fixed_reward_to_near_duplicate_and_novel() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore(
                "top-near",
                1.0,
                artifact_id=UUID(int=1),
                classification="near_duplicate",
            ),
            ParticipantEmissionScore(
                "top-novel",
                1.0,
                artifact_id=UUID(int=2),
                classification="novel",
            ),
            ParticipantEmissionScore(
                "middle-near",
                0.9,
                artifact_id=UUID(int=3),
                classification="near_duplicate",
            ),
            ParticipantEmissionScore(
                "middle-novel",
                0.9,
                artifact_id=UUID(int=4),
                classification="novel",
            ),
            ParticipantEmissionScore("filler-1", 0.8),
            ParticipantEmissionScore("filler-2", 0.7),
            ParticipantEmissionScore("filler-3", 0.6),
            ParticipantEmissionScore("filler-4", 0.5),
            ParticipantEmissionScore("filler-5", 0.4),
            ParticipantEmissionScore("filler-6", 0.3),
        ),
        miner_participation_emission=0.008,
    )

    assert weights["top-near"] == pytest.approx(0.004)
    assert weights["top-novel"] == pytest.approx(0.004)
    assert weights["middle-near"] == pytest.approx(0.002)
    assert weights["middle-novel"] == pytest.approx(0.002)


def test_novelty_distribution_weights_use_exclusive_main_top_10_and_top_50_tiers() -> None:
    participant_scores = (
        ParticipantEmissionScore(
            "top-novel",
            1.0,
            artifact_id=UUID(int=1),
            classification="novel",
        ),
        ParticipantEmissionScore(
            "top-near",
            1.0,
            artifact_id=UUID(int=2),
            classification="near_duplicate",
        ),
        ParticipantEmissionScore(
            "middle-novel",
            0.9,
            artifact_id=UUID(int=3),
            classification="novel",
        ),
        ParticipantEmissionScore("filler-1", 0.8),
        ParticipantEmissionScore("filler-2", 0.7),
        ParticipantEmissionScore("filler-3", 0.6),
        ParticipantEmissionScore("filler-4", 0.5),
        ParticipantEmissionScore("filler-5", 0.4),
        ParticipantEmissionScore(
            "main-novel-outside-top-50",
            0.3,
            artifact_id=UUID(int=4),
            classification="novel",
        ),
    )

    weights = compose_novelty_distribution_weights(
        participant_scores,
        main_participant_artifact_ids=(UUID(int=4),),
    )

    assert weights == {
        "top-novel": pytest.approx(3.0),
        "middle-novel": pytest.approx(1.0),
        "main-novel-outside-top-50": pytest.approx(5.0),
    }


def test_novelty_main_weight_requires_the_selected_artifact_to_have_participated_in_main() -> None:
    shared_artifact_id = UUID(int=10)
    participant_scores = (
        *(
            ParticipantEmissionScore(
                f"higher-{index}",
                1.0 - index / 10,
                artifact_id=UUID(int=index),
                classification="near_duplicate",
            )
            for index in range(1, 10)
        ),
        ParticipantEmissionScore(
            "shared-hotkey",
            0.05,
            artifact_id=shared_artifact_id,
            classification="novel",
        ),
    )

    weights = compose_novelty_distribution_weights(
        participant_scores,
        main_participant_artifact_ids=(UUID(int=99),),
    )

    assert "shared-hotkey" not in weights


def test_selected_main_artifact_receives_main_novelty_weight_outside_score_tiers() -> None:
    shared_artifact_id = UUID(int=10)
    participant_scores = (
        *(
            ParticipantEmissionScore(
                f"higher-{index}",
                1.0 - index / 10,
                artifact_id=UUID(int=index),
                classification="near_duplicate",
            )
            for index in range(1, 10)
        ),
        ParticipantEmissionScore(
            "shared-hotkey",
            0.05,
            artifact_id=shared_artifact_id,
            classification="novel",
        ),
    )

    weights = compose_novelty_distribution_weights(
        participant_scores,
        main_participant_artifact_ids=(shared_artifact_id,),
    )

    assert weights == {"shared-hotkey": pytest.approx(5.0)}


def test_novelty_emission_divides_entire_remaining_fraction_by_distribution_weight() -> None:
    allocations = compose_novelty_emission_allocations(
        {
            "main": 5.0,
            "top": 3.0,
            "middle": 1.0,
        },
        remaining_emission_fraction=0.7,
    )

    assert allocations == {
        "main": pytest.approx(0.7 * 5.0 / 9.0),
        "top": pytest.approx(0.7 / 3.0),
        "middle": pytest.approx(0.7 / 9.0),
    }
    assert fsum(allocations.values()) == pytest.approx(0.7)


def test_artifact_participant_weights_multiply_stage_and_novelty_per_artifact() -> None:
    scores = tuple(
        ParticipantEmissionScore(
            f"hotkey-{index % 3}",
            1.0 - index / 10,
            artifact_id=UUID(int=index + 1),
            classification=classification,
        )
        for index, classification in enumerate(
            (
                "novel",
                "notable_change",
                "near_duplicate",
                "near_duplicate",
                "near_duplicate",
                "near_duplicate",
                "near_duplicate",
                "near_duplicate",
                "near_duplicate",
                "novel",
            )
        )
    )

    weights = compose_artifact_participant_distribution_weights(
        scores,
        main_participant_artifact_ids=(UUID(int=10),),
    )

    assert weights[UUID(int=1)].participation_stage_multiplier == 2
    assert weights[UUID(int=1)].novelty_multiplier == 5
    assert weights[UUID(int=1)].participant_distribution_weight == 10
    assert weights[UUID(int=2)].participation_stage_multiplier == 1
    assert weights[UUID(int=2)].novelty_multiplier == 3
    assert weights[UUID(int=2)].participant_distribution_weight == 3
    assert weights[UUID(int=10)].participation_stage_multiplier == 5
    assert weights[UUID(int=10)].novelty_multiplier == 5
    assert weights[UUID(int=10)].participant_distribution_weight == 25


def test_artifact_participant_weights_preserve_multiple_artifacts_for_one_hotkey() -> None:
    artifact_a = UUID(int=1)
    artifact_b = UUID(int=2)

    weights = compose_artifact_participant_distribution_weights(
        (
            ParticipantEmissionScore(
                "shared-hotkey",
                1.0,
                artifact_id=artifact_a,
                classification="near_duplicate",
            ),
            ParticipantEmissionScore(
                "shared-hotkey",
                0.9,
                artifact_id=artifact_b,
                classification="novel",
            ),
        ),
        main_participant_artifact_ids=(artifact_b,),
    )

    assert set(weights) == {artifact_a, artifact_b}
    assert weights[artifact_a].participant_distribution_weight == 2
    assert weights[artifact_b].participant_distribution_weight == 25


def test_weighted_participant_allocations_divide_the_entire_pool() -> None:
    allocations = compose_weighted_participant_emission_allocations(
        {UUID(int=1): 1, UUID(int=2): 3, UUID(int=3): 25},
        emission_fraction=0.8,
    )

    assert allocations[UUID(int=1)] == pytest.approx(0.8 / 29)
    assert allocations[UUID(int=2)] == pytest.approx(0.8 * 3 / 29)
    assert allocations[UUID(int=3)] == pytest.approx(0.8 * 25 / 29)
    assert fsum(allocations.values()) == pytest.approx(0.8)


def test_equal_participant_emission_divides_entire_remaining_fraction() -> None:
    allocations = compose_equal_participant_emission_allocations(
        ("hotkey-a", "hotkey-b", "hotkey-a", "hotkey-c"),
        remaining_emission_fraction=0.7,
    )

    assert allocations == {
        "hotkey-a": pytest.approx(0.7 / 3.0),
        "hotkey-b": pytest.approx(0.7 / 3.0),
        "hotkey-c": pytest.approx(0.7 / 3.0),
    }
    assert fsum(allocations.values()) == pytest.approx(0.7)


def test_participant_selection_keeps_score_and_classification_on_same_artifact() -> None:
    higher_score_near_duplicate = ParticipantEmissionScore(
        "hotkey-a",
        0.9,
        artifact_id=UUID(int=1),
        classification="near_duplicate",
    )
    lower_score_novel = ParticipantEmissionScore(
        "hotkey-a",
        0.8,
        artifact_id=UUID(int=2),
        classification="novel",
    )

    selected = select_participant_emission_scores((lower_score_novel, higher_score_near_duplicate))

    assert selected == (higher_score_near_duplicate,)


@pytest.mark.parametrize("raw_score", [-0.1, 1.1, float("nan")])
def test_tiered_participant_emission_rejects_invalid_scores(raw_score: float) -> None:
    with pytest.raises(ValueError, match="participant score must be between 0.0 and 1.0"):
        compose_tiered_participant_emission_allocations(
            (ParticipantEmissionScore("hotkey-a", raw_score),)
        )


def test_tiered_participant_emission_rejects_empty_participant_key() -> None:
    with pytest.raises(ValueError, match="participant key must be non-empty"):
        compose_tiered_participant_emission_allocations((ParticipantEmissionScore("", 1.0),))


def test_tiered_participant_emission_stops_by_score_before_total_weight_overflow() -> None:
    weights = compose_tiered_participant_emission_allocations(
        (
            ParticipantEmissionScore("hotkey-a", 1.0),
            ParticipantEmissionScore("hotkey-b", 0.9),
            ParticipantEmissionScore("hotkey-c", 0.8),
        ),
        miner_participation_emission=0.3,
    )

    assert weights == {
        "hotkey-a": pytest.approx(0.6),
        "hotkey-b": pytest.approx(0.3),
    }
    assert fsum(weights.values()) == pytest.approx(0.9)


def test_compose_emission_weights_adds_champion_and_participant_components() -> None:
    weights = compose_emission_weights({10: 0.1}, {11: 0.004, 12: 0.004})

    assert weights == {
        OWNER_UID: pytest.approx(0.892),
        10: pytest.approx(0.1),
        11: pytest.approx(0.004),
        12: pytest.approx(0.004),
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_compose_emission_weights_adds_same_uid_components() -> None:
    weights = compose_emission_weights({10: 0.1}, {10: 0.004})

    assert weights == {
        OWNER_UID: pytest.approx(0.896),
        10: pytest.approx(0.104),
    }


def test_compose_emission_weights_ignores_component_owner_remainders() -> None:
    weights = compose_emission_weights({OWNER_UID: 0.9, 10: 0.1}, {OWNER_UID: 0.996, 11: 0.004})

    assert weights == {
        OWNER_UID: pytest.approx(0.896),
        10: pytest.approx(0.1),
        11: pytest.approx(0.004),
    }


def test_compose_emission_weights_rejects_combined_overflow() -> None:
    with pytest.raises(ValueError, match="emission exceeds total weight"):
        compose_emission_weights({10: 0.998}, {11: 0.004})


def test_prioritized_emission_fills_champion_then_main_then_general_with_full_allocations() -> None:
    composition = compose_prioritized_emission(
        {10: 0.992},
        {11: 0.004, 12: 0.004, 13: 0.004},
        {14: 0.004},
    )

    assert composition.weights == {
        OWNER_UID: pytest.approx(0.0),
        10: pytest.approx(0.992),
        11: pytest.approx(0.004),
        12: pytest.approx(0.004),
    }
    assert composition.accepted_main_additions == {11: pytest.approx(0.004), 12: pytest.approx(0.004)}
    assert composition.dropped_main_additions == {13: pytest.approx(0.004)}
    assert composition.accepted_general_participation == {}
    assert composition.dropped_general_participation == {14: pytest.approx(0.004)}
