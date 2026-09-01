from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.context import ContextSnapshot

_CONTEXT_PAYLOAD = {
    "cost_budget": {
        "session_budget_usd": 1.0,
        "session_hard_limit_usd": 1.5,
        "session_used_budget_usd": 0.25,
        "session_remaining_budget_usd": 0.75,
    },
    "time_budget": {"limit_seconds": 300.0},
}


def test_context_snapshot_validates_exact_budget_contract() -> None:
    context = ContextSnapshot.model_validate(_CONTEXT_PAYLOAD)

    assert context.cost_budget.session_remaining_budget_usd == 0.75
    assert context.time_budget.limit_seconds == 300.0
    assert context.model_dump(mode="json") == _CONTEXT_PAYLOAD


def test_context_snapshot_is_frozen() -> None:
    context = ContextSnapshot.model_validate(_CONTEXT_PAYLOAD)

    with pytest.raises(ValidationError):
        context.time_budget = context.time_budget


@pytest.mark.parametrize(
    "payload",
    (
        {"cost_budget": _CONTEXT_PAYLOAD["cost_budget"]},
        {"time_budget": _CONTEXT_PAYLOAD["time_budget"]},
        {**_CONTEXT_PAYLOAD, "metadata": {}},
    ),
)
def test_context_snapshot_rejects_missing_or_extra_fields(payload: object) -> None:
    with pytest.raises(ValidationError):
        ContextSnapshot.model_validate(payload)
