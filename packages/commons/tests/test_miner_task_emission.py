from __future__ import annotations

import pytest

from harnyx_commons.miner_task_emission import (
    OWNER_UID,
    apply_miner_emission_cap,
    compose_emission_weights,
    compose_participant_emission_weights,
    owner_fallback_weights,
    participant_emission_fraction,
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


def test_owner_fallback_weights_assigns_all_emission_to_owner() -> None:
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


def test_participant_emission_rejects_total_weight_over_one() -> None:
    with pytest.raises(ValueError, match="participant emission exceeds total weight"):
        compose_participant_emission_weights(tuple(range(1, 252)))


@pytest.mark.parametrize("raw_value", [-0.1, 1.1, float("nan")])
def test_participant_emission_rejects_invalid_configured_weight(raw_value: float) -> None:
    with pytest.raises(ValueError, match="miner participation emission must be between 0.0 and 1.0"):
        compose_participant_emission_weights((10,), miner_participation_emission=raw_value)


def test_participant_emission_fraction_uses_configured_weight() -> None:
    assert participant_emission_fraction(3, miner_participation_emission=0.01) == pytest.approx(0.03)


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
