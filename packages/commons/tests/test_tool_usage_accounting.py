from __future__ import annotations

import pytest

from harnyx_commons.domain.session import LlmUsageTotals
from harnyx_commons.domain.tool_usage import (
    EmbeddingToolUsageSummary,
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries, tool_usage_from_llm_usage
from harnyx_commons.llm.schema import LlmUsage


def test_tool_usage_from_llm_usage_splits_gemini_grounding_from_tokens() -> None:
    summary = tool_usage_from_llm_usage(
        LlmUsage(
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            reasoning_tokens=200,
            web_search_calls=2,
        ),
        provider="vertex",
        model="gemini-3-pro-preview",
    )

    assert summary.search_tool_cost == pytest.approx(0.028)
    assert summary.llm_cost == pytest.approx(0.0104)
    assert summary.reference_total_cost_usd == pytest.approx(summary.search_tool_cost + summary.llm_cost)
    assert summary.reference_cost_by_provider["vertex"] == pytest.approx(summary.reference_total_cost_usd)
    assert summary.actual_total_cost_usd is None
    assert summary.llm.providers["vertex"]["gemini-3-pro-preview"].usage.reasoning_tokens == 200


def test_unknown_pricing_keeps_grounding_cost_and_zero_token_cost() -> None:
    summary = tool_usage_from_llm_usage(
        LlmUsage(
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            reasoning_tokens=200,
            web_search_calls=1,
        ),
        provider="vertex",
        model="unknown-model",
    )

    assert summary.search_tool_cost == pytest.approx(0.035)
    assert summary.llm_cost == pytest.approx(0.0)
    assert summary.reference_total_cost_usd == pytest.approx(0.035)


def test_tool_usage_from_llm_usage_coalesces_unavailable_reasoning_tokens() -> None:
    summary = tool_usage_from_llm_usage(
        LlmUsage(
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            reasoning_tokens=None,
        ),
        provider="vertex",
        model="gemini-3-pro-preview",
    )

    assert summary.llm.reasoning_tokens == 0
    assert summary.llm.providers["vertex"]["gemini-3-pro-preview"].usage.reasoning_tokens == 0


def test_merge_tool_usage_summaries_combines_provider_model_usage() -> None:
    left = _usage_summary(
        provider="vertex",
        model="gemini-2.5-pro",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        reasoning_tokens=7,
        llm_cost=0.2,
        search_cost=0.1,
        embedding_cost=0.05,
        actual_llm_cost=0.18,
        actual_search_cost=0.09,
        actual_embedding_cost=0.04,
    )
    right = _usage_summary(
        provider="vertex",
        model="gemini-2.5-pro",
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
        reasoning_tokens=11,
        llm_cost=0.4,
        search_cost=0.3,
        embedding_cost=0.1,
        actual_llm_cost=0.38,
        actual_search_cost=0.29,
        actual_embedding_cost=0.08,
    )

    merged = merge_tool_usage_summaries(left, right)

    model_usage = merged.llm.providers["vertex"]["gemini-2.5-pro"]
    assert merged.search_tool.call_count == 2
    assert merged.search_tool.reference_cost == pytest.approx(0.4)
    assert merged.search_tool.actual_cost == pytest.approx(0.38)
    assert merged.embedding.call_count == 2
    assert merged.embedding_cost == pytest.approx(0.15)
    assert merged.embedding.reference_cost == pytest.approx(0.15)
    assert merged.embedding.actual_cost == pytest.approx(0.12)
    assert merged.llm.reasoning_tokens == 18
    assert merged.llm.reference_cost == pytest.approx(0.6)
    assert merged.llm.actual_cost == pytest.approx(0.56)
    assert model_usage.usage.prompt_tokens == 30
    assert model_usage.usage.reasoning_tokens == 18
    assert model_usage.reference_cost == pytest.approx(0.6)
    assert model_usage.actual_cost == pytest.approx(0.56)
    assert merged.reference_total_cost_usd == pytest.approx(1.15)
    assert merged.reference_cost_by_provider["vertex"] == pytest.approx(1.15)
    assert merged.actual_total_cost_usd == pytest.approx(1.06)
    assert merged.actual_cost_by_provider["vertex"] == pytest.approx(1.06)


def test_merge_tool_usage_summaries_preserves_asymmetric_known_actual_costs() -> None:
    left = _usage_summary(
        provider="vertex",
        model="gemini-2.5-pro",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        reasoning_tokens=7,
        llm_cost=0.2,
        search_cost=0.1,
        embedding_cost=0.05,
        actual_llm_cost=0.18,
        actual_search_cost=0.09,
        actual_embedding_cost=0.04,
    )
    right = _usage_summary(
        provider="vertex",
        model="gemini-2.5-flash",
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
        reasoning_tokens=11,
        llm_cost=0.4,
        search_cost=0.3,
        embedding_cost=0.1,
        actual_llm_cost=None,
        actual_search_cost=None,
        actual_embedding_cost=None,
    )

    merged = merge_tool_usage_summaries(left, right)

    assert merged.search_tool.actual_cost == pytest.approx(0.09)
    assert merged.embedding.actual_cost == pytest.approx(0.04)
    assert merged.llm.actual_cost == pytest.approx(0.18)
    assert merged.actual_total_cost_usd == pytest.approx(0.31)
    assert merged.llm.providers["vertex"]["gemini-2.5-pro"].actual_cost == pytest.approx(0.18)
    assert merged.llm.providers["vertex"]["gemini-2.5-flash"].actual_cost is None


def _usage_summary(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    reasoning_tokens: int,
    llm_cost: float,
    search_cost: float,
    actual_llm_cost: float | None,
    actual_search_cost: float | None,
    embedding_cost: float = 0.0,
    actual_embedding_cost: float | None = None,
) -> ToolUsageSummary:
    actual_total = None
    actual_by_provider: dict[str, float] = {}
    if actual_llm_cost is not None or actual_search_cost is not None or actual_embedding_cost is not None:
        actual_total = (actual_llm_cost or 0.0) + (actual_search_cost or 0.0) + (actual_embedding_cost or 0.0)
        actual_by_provider[provider] = actual_total
    return ToolUsageSummary(
        search_tool=SearchToolUsageSummary(
            call_count=1,
            cost=search_cost,
            reference_cost=search_cost,
            actual_cost=actual_search_cost,
        ),
        search_tool_cost=search_cost,
        embedding=EmbeddingToolUsageSummary(
            call_count=1 if embedding_cost else 0,
            cost=embedding_cost,
            reference_cost=embedding_cost,
            actual_cost=actual_embedding_cost,
        ),
        embedding_cost=embedding_cost,
        llm=LlmUsageSummary(
            call_count=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            cost=llm_cost,
            reference_cost=llm_cost,
            actual_cost=actual_llm_cost,
            providers={
                provider: {
                    model: LlmModelUsageCost(
                        usage=LlmUsageTotals(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            reasoning_tokens=reasoning_tokens,
                            call_count=1,
                        ),
                        cost=llm_cost,
                        reference_cost=llm_cost,
                        actual_cost=actual_llm_cost,
                    )
                }
            },
        ),
        llm_cost=llm_cost,
        reference_total_cost_usd=llm_cost + search_cost + embedding_cost,
        reference_cost_by_provider={provider: llm_cost + search_cost + embedding_cost},
        actual_total_cost_usd=actual_total,
        actual_cost_by_provider=actual_by_provider,
    )
