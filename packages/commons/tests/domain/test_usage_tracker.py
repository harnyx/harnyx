from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.errors import BudgetExceededError
from harnyx_commons.tools.usage_tracker import ToolCallUsage, UsageTracker


def make_session(*, budget_usd: float, hard_limit_usd: float | None = None) -> Session:
    return Session(
        session_id=uuid4(),
        uid=1,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 15, tzinfo=UTC),
        expires_at=datetime(2025, 10, 16, tzinfo=UTC),
        budget_usd=budget_usd,
        hard_limit_usd=hard_limit_usd,
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
    assert updated.usage.reference_total_cost_usd == pytest.approx(0.05)
    assert updated.usage.actual_total_cost_usd == pytest.approx(0.05)
    assert updated.usage.actual_cost_by_provider == {"desearch": pytest.approx(0.05)}
    assert updated.usage.llm_tokens_last_call == 500
    assert updated.usage.llm_usage_totals == {}


def test_usage_tracker_records_matching_actual_cost_for_session_cost() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.2)

    updated = tracker.record_tool_call(
        session,
        tool_name="llm_chat",
        llm_tokens=15,
        usage=ToolCallUsage(
            provider="chutes",
            model="openai/gpt-oss-20b",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        cost_usd=0.03,
        actual_cost_usd=0.03,
        actual_cost_provider="openrouter",
    )

    assert updated.usage.total_cost_usd == pytest.approx(0.03)
    assert updated.usage.reference_total_cost_usd == pytest.approx(0.03)
    assert updated.usage.cost_by_provider == {"openrouter": pytest.approx(0.03)}
    assert updated.usage.reference_cost_by_provider == {"openrouter": pytest.approx(0.03)}
    assert updated.usage.actual_total_cost_usd == pytest.approx(0.03)
    assert updated.usage.actual_cost_by_provider == {"openrouter": pytest.approx(0.03)}


def test_usage_tracker_accumulates_one_cost_space_for_cost_only_and_actual_cost_calls() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.2)
    first = tracker.record_tool_call(
        session,
        tool_name="search_web",
        llm_tokens=0,
        cost_usd=0.005,
        actual_cost_usd=0.005,
        actual_cost_provider="parallel",
    )

    second = tracker.record_tool_call(first, tool_name="llm_chat", llm_tokens=10, cost_usd=0.02)

    assert second.usage.total_cost_usd == pytest.approx(0.025)
    assert second.usage.reference_total_cost_usd == pytest.approx(0.025)
    assert second.usage.cost_by_provider == {
        "parallel": pytest.approx(0.005),
        "chutes": pytest.approx(0.02),
    }
    assert second.usage.reference_cost_by_provider == second.usage.cost_by_provider
    assert second.usage.actual_total_cost_usd == pytest.approx(0.025)
    assert second.usage.actual_cost_by_provider == {
        "parallel": pytest.approx(0.005),
        "chutes": pytest.approx(0.02),
    }


def test_usage_tracker_uses_reference_for_limits_and_actual_for_financial_totals() -> None:
    updated = UsageTracker().record_tool_call(
        make_session(budget_usd=0.2),
        tool_name="llm_chat",
        llm_tokens=15,
        usage=ToolCallUsage(provider="openrouter", model="inception/mercury-3", total_tokens=15),
        cost_usd=0.04,
        actual_cost_usd=0.02,
        actual_cost_provider="openrouter",
    )
    assert updated.usage.total_cost_usd == pytest.approx(0.04)
    assert updated.usage.reference_total_cost_usd == pytest.approx(0.04)
    assert updated.usage.actual_total_cost_usd == pytest.approx(0.02)
    assert updated.usage.actual_cost_by_provider == {"openrouter": pytest.approx(0.02)}


def test_usage_tracker_retains_observed_charge_without_reference() -> None:
    updated = UsageTracker().record_tool_call(
        make_session(budget_usd=0.2),
        tool_name="llm_chat",
        llm_tokens=0,
        actual_cost_usd=0.005,
        actual_cost_provider="openrouter",
    )
    assert updated.usage.total_cost_usd == 0
    assert updated.usage.actual_total_cost_usd == pytest.approx(0.005)


def test_usage_tracker_rejects_boolean_cost() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.2)

    with pytest.raises(ValueError, match="cost_usd must be numeric"):
        tracker.record_tool_call(
            session,
            tool_name="search_web",
            llm_tokens=0,
            cost_usd=cast(float, True),
            actual_cost_usd=cast(float, True),
            actual_cost_provider="parallel",
        )


def test_session_usage_normalizes_reference_fields_to_session_cost() -> None:
    usage = SessionUsage(
        total_cost_usd=0.005,
        cost_by_provider={"parallel": 0.005},
        reference_total_cost_usd=0.0,
        reference_cost_by_provider={"desearch": 0.0},
        actual_total_cost_usd=0.005,
        actual_cost_by_provider={"parallel": 0.005},
    )

    assert usage.total_cost_usd == pytest.approx(0.005)
    assert usage.cost_by_provider == {"parallel": pytest.approx(0.005)}
    assert usage.reference_total_cost_usd == pytest.approx(0.005)
    assert usage.reference_cost_by_provider == {"parallel": pytest.approx(0.005)}


def test_session_usage_update_keeps_reference_fields_aligned_with_updated_session_cost() -> None:
    usage = SessionUsage(
        total_cost_usd=0.0,
        cost_by_provider={"openrouter": 0.0},
        reference_total_cost_usd=0.04,
        reference_cost_by_provider={"chutes": 0.04},
        actual_total_cost_usd=0.0,
        actual_cost_by_provider={"openrouter": 0.0},
    )

    updated = usage.update(
        total_cost_usd=0.006,
        cost_by_provider={"parallel": 0.006},
    )

    assert updated.total_cost_usd == pytest.approx(0.006)
    assert updated.cost_by_provider == {"parallel": pytest.approx(0.006)}
    assert updated.reference_total_cost_usd == pytest.approx(0.006)
    assert updated.reference_cost_by_provider == {"parallel": pytest.approx(0.006)}


def test_session_usage_update_ignores_legacy_reference_arguments_for_session_cost_aliases() -> None:
    usage = SessionUsage(
        total_cost_usd=0.005,
        cost_by_provider={"parallel": 0.005},
        reference_total_cost_usd=0.001,
        reference_cost_by_provider={"desearch": 0.001},
        actual_total_cost_usd=0.005,
        actual_cost_by_provider={"parallel": 0.005},
    )

    updated = usage.update(
        reference_total_cost_usd=0.02,
        reference_cost_by_provider={"chutes": 0.02},
    )

    assert updated.total_cost_usd == pytest.approx(0.005)
    assert updated.cost_by_provider == {"parallel": pytest.approx(0.005)}
    assert updated.reference_total_cost_usd == pytest.approx(0.005)
    assert updated.reference_cost_by_provider == {"parallel": pytest.approx(0.005)}


def test_usage_tracker_records_completed_call_even_when_it_exhausts_budget() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.05)
    first = tracker.record_tool_call(session, tool_name="llm_chat", llm_tokens=50, cost_usd=0.04)
    assert first.usage.total_cost_usd == pytest.approx(0.04)

    second = tracker.record_tool_call(first, tool_name="search_web", llm_tokens=50, cost_usd=0.02)

    assert second.usage.total_cost_usd == pytest.approx(0.06)


def test_usage_tracker_rejects_calls_when_session_inactive() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.1).mark_exhausted()

    with pytest.raises(BudgetExceededError):
        tracker.record_tool_call(session, tool_name="search_web", llm_tokens=10, cost_usd=0.01)


def test_usage_tracker_keeps_actual_total_unavailable_after_unknown_provider_cost() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.1)

    updated = tracker.record_tool_call(
        session,
        tool_name="embed_text",
        llm_tokens=0,
        actual_cost_provider="openrouter",
    )

    assert updated.usage.total_cost_usd == 0.0
    assert updated.usage.actual_total_cost_usd is None
    assert updated.usage.actual_cost_by_provider == {}


def test_usage_tracker_allows_usage_past_soft_budget_when_hard_limit_is_higher() -> None:
    tracker = UsageTracker()
    session = make_session(budget_usd=0.5, hard_limit_usd=1.0)

    updated = tracker.record_tool_call(
        session,
        tool_name="search_web",
        llm_tokens=10,
        cost_usd=0.52,
    )

    assert updated.usage.total_cost_usd == pytest.approx(0.52)


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


@pytest.mark.parametrize("provider", ["openrouter", "cloudflare"])
def test_recovered_chat_costs_preserve_other_providers_nonchat_charges_and_unknown_actual_total(provider: str) -> None:
    from harnyx_commons.domain.session import LlmUsageTotals
    from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome
    from harnyx_commons.domain.tool_usage import LlmModelUsageCost, LlmUsageSummary, ToolUsageSummary
    from harnyx_commons.tools.receipt_accounting import recover_inference_usage
    from harnyx_commons.tools.types import ToolName

    def receipt(tool: ToolName, actual: float, reference: float, receipt_provider: str = provider) -> ToolCall:
        return ToolCall(
            receipt_id=str(uuid4()),
            session_id=uuid4(),
            uid=1,
            tool=tool,
            issued_at=datetime.now(UTC),
            outcome=ToolCallOutcome.OK,
            details=ToolCallDetails(
                request_hash=uuid4().hex,
                request_payload={"kwargs": {"provider": receipt_provider, "model": "promo/model"}},
                response_payload={
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 19, "reasoning_tokens": 4}
                },
                cost_usd=reference,
                actual_cost_usd=actual,
                actual_cost_provider=receipt_provider,
            ),
        )

    gateway = LlmModelUsageCost(usage=LlmUsageTotals(call_count=1, total_tokens=7), cost=0.03, actual_cost=0.03)
    old_chat = LlmModelUsageCost(usage=LlmUsageTotals(call_count=2, total_tokens=9), cost=0.02, actual_cost=None)
    submitted = ToolUsageSummary(
        llm=LlmUsageSummary(
            providers={provider: {"promo/model": old_chat}, "ai_gateway": {"gateway/model": gateway}}, cost=0.05
        ),
        llm_cost=0.05,
        reference_total_cost_usd=0.055,
        reference_cost_by_provider={provider: 0.025, "ai_gateway": 0.03},
        decision_reference_cost_by_provider={provider: 0.005},
        actual_total_cost_usd=None,
        actual_cost_by_provider={provider: 0.015, "ai_gateway": 0.03},
    )
    recovered_receipts = (
        receipt("llm_chat", 0.01, 0.02),
        receipt("llm_chat", 0.01, 0.02),
        receipt("decision_query", 0.005, 0.005),
        # A lost Decision response is recovered through the same inference policy.
        receipt("decision_query", 0.02, 0.02),
        receipt("llm_chat", 0.07, 0.07, "ai_gateway"),
        receipt("search_web", 0.09, 0.09, "tavily"),
    )
    recovered = recover_inference_usage(submitted, recovered_receipts)
    assert recover_inference_usage(recovered, recovered_receipts) == recovered
    assert recovered.llm.providers["ai_gateway"]["gateway/model"] == gateway
    assert recovered.reference_total_cost_usd == pytest.approx(0.095)
    assert recovered.reference_cost_by_provider[provider] == pytest.approx(0.065)
    assert recovered.actual_cost_by_provider == pytest.approx({provider: 0.045, "ai_gateway": 0.03})
    assert recovered.actual_total_cost_usd is None
    assert recovered.llm.call_count == 3 and recovered.llm.total_tokens == 45
    assert recovered.llm.reasoning_tokens == 8
    assert submitted.actual_cost_by_provider[provider] == 0.015
