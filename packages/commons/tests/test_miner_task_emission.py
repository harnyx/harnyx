from __future__ import annotations

import pytest

from harnyx_commons.miner_task_emission import (
    apply_miner_emission_cap,
    owner_fallback_weights,
)


def test_apply_miner_emission_cap_burns_all_miner_emission() -> None:
    weights = apply_miner_emission_cap({7: 0.6, 8: 0.4}, batch_score=0.5)

    assert weights == {
        0: pytest.approx(1.0),
        7: pytest.approx(0.0),
        8: pytest.approx(0.0),
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_apply_miner_emission_cap_ignores_owner_weight_in_base_vector() -> None:
    weights = apply_miner_emission_cap({0: 0.8, 7: 0.6, 8: 0.4}, batch_score=1.0)

    assert weights == {
        0: pytest.approx(1.0),
        7: pytest.approx(0.0),
        8: pytest.approx(0.0),
    }


@pytest.mark.parametrize("weights", [{}, {0: 1.0}])
def test_apply_miner_emission_cap_rejects_empty_miner_weights(weights: dict[int, float]) -> None:
    with pytest.raises(ValueError, match="miner weights are empty"):
        apply_miner_emission_cap(weights, batch_score=1.0)


def test_apply_miner_emission_cap_rejects_non_positive_miner_total() -> None:
    with pytest.raises(ValueError, match="miner weights must have positive miner total"):
        apply_miner_emission_cap({7: 0.0, 8: 0.0}, batch_score=1.0)


@pytest.mark.parametrize("batch_score", [-0.1, 1.1, float("nan")])
def test_apply_miner_emission_cap_rejects_invalid_batch_score(batch_score: float) -> None:
    with pytest.raises(ValueError, match="miner task batch score must be between 0.0 and 1.0"):
        apply_miner_emission_cap({7: 1.0}, batch_score=batch_score)


def test_owner_fallback_weights_assigns_all_emission_to_owner() -> None:
    assert owner_fallback_weights() == {0: 1.0}
