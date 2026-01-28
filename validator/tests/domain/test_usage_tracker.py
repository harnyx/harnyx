from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from caster_commons.domain.session import Session, SessionStatus, SessionUsage
from caster_commons.tools.usage_tracker import ToolCallUsage, UsageTracker
from caster_validator.domain.exceptions import BudgetExceededError


def make_session(*, budget_usd: float) -> Session:
    return Session(
        session_id=uuid4(),
        uid=1,
        claim_id=uuid4(),
        issued_at=datetime(2025, 10, 15, tzinfo=UTC),
        expires_at=datetime(2025, 10, 16, tzinfo=UTC),
        budget_usd=budget_usd,
        usage=SessionUsage(),
        status=SessionStatus.ACTIVE,
    )


def test_usage_tracker_records_tool_call_within_limits() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.2)

    updated = tracker.record_tool_call(
        session,
        tool_name="search_web",
        llm_tokens=500,
        cost_usd=0.05,
    )

    assert updated.usage.total_cost_usd == pytest.approx(0.05)
    assert updated.usage.llm_tokens_last_call == 500
    assert updated.usage.llm_usage_totals == {}


def test_usage_tracker_blocks_when_limits_exceeded() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.05)
    first = tracker.record_tool_call(session, tool_name="llm_chat", llm_tokens=50, cost_usd=0.04)
    assert first.usage.total_cost_usd == pytest.approx(0.04)

    with pytest.raises(BudgetExceededError):
        tracker.record_tool_call(first, tool_name="search_web", llm_tokens=50, cost_usd=0.02)


def test_usage_tracker_rejects_calls_when_session_inactive() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.1).mark_exhausted()

    with pytest.raises(BudgetExceededError):
        tracker.record_tool_call(session, tool_name="search_web", llm_tokens=10, cost_usd=0.01)


def test_usage_tracker_accumulates_llm_usage() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=1.0)

    usage = ToolCallUsage(
        provider="chutes",
        model="demo",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    first = tracker.record_tool_call(
        session,
        tool_name="llm_chat",
        llm_tokens=15,
        usage=usage,
        cost_usd=0.01,
    )

    aggregated = first.usage.llm_usage_totals["chutes"]["demo"]
    assert aggregated.prompt_tokens == 10
    assert aggregated.completion_tokens == 5
    assert aggregated.total_tokens == 15
    assert aggregated.call_count == 1

    second = tracker.record_tool_call(
        first,
        tool_name="llm_chat",
        llm_tokens=20,
        usage=ToolCallUsage(
            provider="chutes",
            model="demo",
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=None,
        ),
        cost_usd=0.02,
    )

    combined = second.usage.llm_usage_totals["chutes"]["demo"]
    assert combined.prompt_tokens == 22
    assert combined.completion_tokens == 13
    assert combined.total_tokens == 35
    assert combined.call_count == 2
