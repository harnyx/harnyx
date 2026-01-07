from __future__ import annotations

import pytest

from caster_validator.runtime.evaluation_worker import estimate_cycle_duration_seconds


def test_estimate_cycle_duration_includes_bootstrap_and_evaluations() -> None:
    # Use explicit values matching the function defaults
    sandbox_startup = 2.0
    sandbox_healthz = True
    bootstrap_padding = 2.0

    estimate = estimate_cycle_duration_seconds(
        uid_count=2,
        claim_count=3,
        sandbox_startup_delay_seconds=sandbox_startup,
        sandbox_wait_for_healthz=sandbox_healthz,
        bootstrap_padding_seconds=bootstrap_padding,
    )

    # Sandbox startup delay + healthz wait + padding
    expected_startup = sandbox_startup + 15.0 + bootstrap_padding
    # Client timeouts are all 30.0 by default
    expected_per_eval = 30.0
    assert estimate == pytest.approx(2 * expected_startup + 2 * 3 * expected_per_eval)


def test_estimate_cycle_duration_handles_empty_inputs() -> None:
    assert estimate_cycle_duration_seconds(uid_count=0, claim_count=5) == 0.0
    assert estimate_cycle_duration_seconds(uid_count=4, claim_count=0) == 0.0
