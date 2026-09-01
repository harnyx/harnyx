from __future__ import annotations

from harnyx_commons.tools.dto import ToolBudgetSnapshot
from harnyx_commons.tools.http_serialization import serialize_tool_budget


def test_serialize_tool_budget_preserves_session_snapshot() -> None:
    snapshot = ToolBudgetSnapshot(
        session_budget_usd=2.0,
        session_hard_limit_usd=3.0,
        session_used_budget_usd=0.75,
        session_remaining_budget_usd=1.25,
    )

    assert serialize_tool_budget(snapshot).model_dump(mode="json") == {
        "session_budget_usd": 2.0,
        "session_hard_limit_usd": 3.0,
        "session_used_budget_usd": 0.75,
        "session_remaining_budget_usd": 1.25,
    }
