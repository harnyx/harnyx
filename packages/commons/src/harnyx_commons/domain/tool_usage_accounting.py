"""Accounting helpers for shared tool usage summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from harnyx_commons.domain.session import LlmUsageTotals
from harnyx_commons.domain.tool_usage import (
    EmbeddingToolUsageSummary,
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from harnyx_commons.llm.pricing import generation_usage_cost_breakdown
from harnyx_commons.llm.schema import LlmUsage


def tool_usage_from_llm_usage(
    usage: LlmUsage,
    *,
    provider: str,
    model: str,
) -> ToolUsageSummary:
    """Convert one LLM call usage record into a shared tool-usage summary."""
    breakdown = generation_usage_cost_breakdown(usage, provider=provider, model=model)
    grounded_cost = _breakdown_float(breakdown.get("usd_cost_grounded"))
    total_reference_cost = _breakdown_float(breakdown.get("usd_cost"))
    llm_cost = max(total_reference_cost - grounded_cost, 0.0)
    reasoning_tokens = _coalesce_reasoning_tokens((usage.reasoning_tokens,))
    model_usage = LlmModelUsageCost(
        usage=LlmUsageTotals(
            prompt_tokens=int(usage.prompt_tokens or 0),
            completion_tokens=int(usage.completion_tokens or 0),
            total_tokens=int(usage.total_tokens or 0),
            reasoning_tokens=reasoning_tokens,
            call_count=1,
        ),
        cost=llm_cost,
        reference_cost=llm_cost,
        actual_cost=None,
    )
    return ToolUsageSummary(
        search_tool=SearchToolUsageSummary(
            call_count=int(usage.web_search_calls or 0),
            cost=grounded_cost,
            reference_cost=grounded_cost,
            actual_cost=None,
        ),
        search_tool_cost=grounded_cost,
        llm=LlmUsageSummary(
            call_count=1,
            prompt_tokens=int(usage.prompt_tokens or 0),
            completion_tokens=int(usage.completion_tokens or 0),
            total_tokens=int(usage.total_tokens or 0),
            reasoning_tokens=reasoning_tokens,
            cost=llm_cost,
            reference_cost=llm_cost,
            actual_cost=None,
            providers={provider: {model: model_usage}},
        ),
        llm_cost=llm_cost,
        reference_total_cost_usd=total_reference_cost,
        reference_cost_by_provider={provider: total_reference_cost},
        actual_total_cost_usd=None,
        actual_cost_by_provider={},
    )


def merge_tool_usage_summaries(left: ToolUsageSummary, right: ToolUsageSummary) -> ToolUsageSummary:
    """Merge two shared tool-usage summaries without dropping cost subtotals."""
    return ToolUsageSummary(
        search_tool=SearchToolUsageSummary(
            call_count=left.search_tool.call_count + right.search_tool.call_count,
            cost=left.search_tool.cost + right.search_tool.cost,
            reference_cost=left.search_tool.reference_cost + right.search_tool.reference_cost,
            actual_cost=_merge_optional_cost(left.search_tool.actual_cost, right.search_tool.actual_cost),
        ),
        search_tool_cost=left.search_tool_cost + right.search_tool_cost,
        llm=LlmUsageSummary(
            call_count=left.llm.call_count + right.llm.call_count,
            prompt_tokens=left.llm.prompt_tokens + right.llm.prompt_tokens,
            completion_tokens=left.llm.completion_tokens + right.llm.completion_tokens,
            total_tokens=left.llm.total_tokens + right.llm.total_tokens,
            reasoning_tokens=left.llm.reasoning_tokens + right.llm.reasoning_tokens,
            cost=left.llm.cost + right.llm.cost,
            reference_cost=left.llm.reference_cost + right.llm.reference_cost,
            actual_cost=_merge_optional_cost(left.llm.actual_cost, right.llm.actual_cost),
            providers=_merge_llm_provider_usage(left.llm.providers, right.llm.providers),
        ),
        llm_cost=left.llm_cost + right.llm_cost,
        embedding=EmbeddingToolUsageSummary(
            call_count=left.embedding.call_count + right.embedding.call_count,
            cost=left.embedding.cost + right.embedding.cost,
            reference_cost=left.embedding.reference_cost + right.embedding.reference_cost,
            actual_cost=_merge_optional_cost(left.embedding.actual_cost, right.embedding.actual_cost),
        ),
        embedding_cost=left.embedding_cost + right.embedding_cost,
        reference_total_cost_usd=left.reference_total_cost_usd + right.reference_total_cost_usd,
        reference_cost_by_provider=_merge_cost_by_provider(
            left.reference_cost_by_provider,
            right.reference_cost_by_provider,
        ),
        actual_total_cost_usd=_merge_optional_cost(left.actual_total_cost_usd, right.actual_total_cost_usd),
        actual_cost_by_provider=_merge_cost_by_provider(
            left.actual_cost_by_provider,
            right.actual_cost_by_provider,
        ),
    )


def _breakdown_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _coalesce_reasoning_tokens(values: Iterable[int | None]) -> int:
    return sum(value or 0 for value in values)


def _merge_optional_cost(left: float | None, right: float | None) -> float | None:
    if left is None and right is None:
        return None
    return (left or 0.0) + (right or 0.0)


def _merge_cost_by_provider(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {
        provider: left.get(provider, 0.0) + right.get(provider, 0.0)
        for provider in set(left) | set(right)
    }


def _merge_llm_provider_usage(
    left: Mapping[str, Mapping[str, LlmModelUsageCost]],
    right: Mapping[str, Mapping[str, LlmModelUsageCost]],
) -> dict[str, dict[str, LlmModelUsageCost]]:
    merged: dict[str, dict[str, LlmModelUsageCost]] = {}
    for provider_name in set(left) | set(right):
        left_models = left.get(provider_name, {})
        right_models = right.get(provider_name, {})
        merged[provider_name] = {
            model_name: _merge_model_usage(
                left_models.get(model_name),
                right_models.get(model_name),
            )
            for model_name in set(left_models) | set(right_models)
        }
    return merged


def _merge_model_usage(
    left: LlmModelUsageCost | None,
    right: LlmModelUsageCost | None,
) -> LlmModelUsageCost:
    left_usage = left.usage if left is not None else LlmUsageTotals()
    right_usage = right.usage if right is not None else LlmUsageTotals()
    return LlmModelUsageCost(
        usage=LlmUsageTotals(
            prompt_tokens=left_usage.prompt_tokens + right_usage.prompt_tokens,
            completion_tokens=left_usage.completion_tokens + right_usage.completion_tokens,
            total_tokens=left_usage.total_tokens + right_usage.total_tokens,
            reasoning_tokens=left_usage.reasoning_tokens + right_usage.reasoning_tokens,
            call_count=left_usage.call_count + right_usage.call_count,
        ),
        cost=(left.cost if left is not None else 0.0) + (right.cost if right is not None else 0.0),
        reference_cost=(
            (left.reference_cost if left is not None else 0.0)
            + (right.reference_cost if right is not None else 0.0)
        ),
        actual_cost=_merge_optional_cost(
            left.actual_cost if left is not None else None,
            right.actual_cost if right is not None else None,
        ),
    )


__all__ = [
    "merge_tool_usage_summaries",
    "tool_usage_from_llm_usage",
]
