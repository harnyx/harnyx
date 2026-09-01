from __future__ import annotations

import pytest

from harnyx_miner_sdk.tools.time_budget import ExecutionTimeBudgetDTO


def test_execution_time_budget_contains_only_positive_limit() -> None:
    budget = ExecutionTimeBudgetDTO(limit_seconds=0.5)

    assert budget.limit_seconds == 0.5
    assert not hasattr(budget, "remaining_seconds")


@pytest.mark.parametrize("limit_seconds", (True, 0.0, -1.0, float("nan"), float("inf")))
def test_execution_time_budget_rejects_invalid_limit(limit_seconds: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExecutionTimeBudgetDTO(limit_seconds=limit_seconds)  # type: ignore[arg-type]
