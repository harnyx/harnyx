"""Budget enforcement and usage accounting shared between services."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING

from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.errors import BudgetExceededError
from harnyx_commons.tools.cost_accumulator import accumulate_costs, resolve_provider
from harnyx_commons.tools.llm_usage_accumulator import accumulate_llm_usage
from harnyx_commons.tools.types import ToolName

if TYPE_CHECKING:
    from harnyx_commons.domain.session import LlmUsageTotals


@dataclass(frozen=True, slots=True)
class ToolCallUsage:
    """Structured LLM usage metadata captured from a tool response."""

    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None


class UsageTracker:
    """Records per-session tool usage for already-completed invocations."""

    def record_tool_call(
        self,
        session: Session,
        *,
        tool_name: ToolName,
        llm_tokens: int,
        usage: ToolCallUsage | None = None,
        cost_usd: float | None = None,
        actual_cost_usd: float | None = None,
        actual_cost_provider: str | None = None,
    ) -> Session:
        self._validate_session(session, llm_tokens)
        normalized_name, usage_details = self._prepare_usage(tool_name, usage, cost_usd)

        updated_usage = self._update_usage(
            session.usage,
            normalized_name=normalized_name,
            llm_tokens=llm_tokens,
            usage=usage_details,
            cost_usd=cost_usd,
            actual_cost_usd=actual_cost_usd,
            actual_cost_provider=actual_cost_provider,
        )

        return session.with_usage(updated_usage)

    @staticmethod
    def _validate_session(session: Session, llm_tokens: int) -> None:
        if llm_tokens < 0:
            raise ValueError("llm_tokens must be non-negative")
        if session.status is not SessionStatus.ACTIVE:
            raise BudgetExceededError("cannot record tool calls on inactive sessions")

    def _prepare_usage(
        self,
        tool_name: ToolName,
        usage: ToolCallUsage | None,
        cost_usd: float | None,
    ) -> tuple[ToolName, ToolCallUsage | None]:
        return tool_name, self._normalize_usage(usage, cost_usd)

    def _update_usage(
        self,
        budget: SessionUsage,
        *,
        normalized_name: str,
        llm_tokens: int,
        usage: ToolCallUsage | None,
        cost_usd: float | None,
        actual_cost_usd: float | None,
        actual_cost_provider: str | None,
    ) -> SessionUsage:
        usage_totals = accumulate_llm_usage(
            budget.llm_usage_totals,
            usage=usage,
            llm_tokens=llm_tokens,
        )
        session_cost = _resolve_one_cost_value(
            cost_usd=cost_usd,
            actual_cost_usd=actual_cost_usd,
            usage=usage,
        )

        total_cost, provider_costs = accumulate_costs(
            budget.total_cost_usd,
            budget.cost_by_provider,
            usage=usage,
            cost_usd=session_cost,
            normalized_tool_name=normalized_name,
            provider_override=actual_cost_provider,
        )

        actual_total_cost, actual_provider_costs = accumulate_actual_costs(
            budget.actual_total_cost_usd,
            budget.actual_cost_by_provider,
            cost_usd=session_cost,
            actual_cost_usd=actual_cost_usd,
            actual_cost_provider=actual_cost_provider,
            usage=usage,
            normalized_tool_name=normalized_name,
        )

        return self._build_usage(
            budget=budget,
            llm_tokens=llm_tokens,
            usage_totals=usage_totals,
            total_cost=total_cost,
            provider_costs=provider_costs,
            actual_total_cost=actual_total_cost,
            actual_provider_costs=actual_provider_costs,
        )

    @staticmethod
    def _normalize_usage(usage: ToolCallUsage | None, cost_usd: float | None) -> ToolCallUsage | None:
        if usage is None:
            return None
        if cost_usd is None and usage.cost_usd is not None:
            return usage
        return ToolCallUsage(
            provider=usage.provider,
            model=usage.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cost_usd=cost_usd,
        )

    @staticmethod
    def _build_usage(
        *,
        budget: SessionUsage,
        llm_tokens: int,
        usage_totals: dict[str, dict[str, LlmUsageTotals]],
        total_cost: float,
        provider_costs: dict[str, float],
        actual_total_cost: float | None,
        actual_provider_costs: dict[str, float],
    ) -> SessionUsage:
        return SessionUsage(
            llm_tokens_last_call=llm_tokens,
            llm_usage_totals=usage_totals,
            total_cost_usd=total_cost,
            cost_by_provider=provider_costs,
            reference_total_cost_usd=total_cost,
            reference_cost_by_provider=provider_costs,
            actual_total_cost_usd=actual_total_cost,
            actual_cost_by_provider=actual_provider_costs,
        )


def accumulate_actual_costs(
    actual_total_cost_usd: float | None,
    actual_cost_by_provider: dict[str, float],
    *,
    cost_usd: float | None = None,
    reference_cost_usd: float | None = None,
    actual_cost_usd: float | None = None,
    actual_cost_provider: str | None = None,
    usage: ToolCallUsage | None,
    normalized_tool_name: str,
) -> tuple[float | None, dict[str, float]]:
    """Produce updated compatibility actual-cost totals from the one cost value."""
    resolved_cost = _resolve_one_cost_value(
        cost_usd=cost_usd,
        reference_cost_usd=reference_cost_usd,
        actual_cost_usd=actual_cost_usd,
        usage=usage,
    )
    if resolved_cost is None:
        return actual_total_cost_usd, dict(actual_cost_by_provider)

    provider = resolve_provider(
        usage,
        normalized_tool_name=normalized_tool_name,
        provider_override=actual_cost_provider,
    )
    provider_costs = dict(actual_cost_by_provider)
    provider_costs[provider] = provider_costs.get(provider, 0.0) + resolved_cost

    if actual_total_cost_usd is None:
        return None, provider_costs
    return actual_total_cost_usd + resolved_cost, provider_costs


def _resolve_one_cost_value(
    *,
    cost_usd: float | None,
    actual_cost_usd: float | None,
    usage: ToolCallUsage | None,
    reference_cost_usd: float | None = None,
) -> float | None:
    usage_cost_usd = None if usage is None else usage.cost_usd
    supplied = []
    for value in (cost_usd, reference_cost_usd, usage_cost_usd, actual_cost_usd):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("cost_usd must be numeric")
        supplied.append(float(value))
    if not supplied:
        return None
    if actual_cost_usd is not None and cost_usd is None and reference_cost_usd is None and usage_cost_usd is None:
        raise ValueError("actual_cost_usd requires matching cost_usd")

    resolved = supplied[0]
    if resolved < 0.0:
        raise ValueError("cost_usd must be non-negative")
    for value in supplied[1:]:
        if value < 0.0:
            raise ValueError("cost_usd must be non-negative")
        if not isclose(value, resolved, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("cost_usd, reference_cost_usd, and actual_cost_usd must match")
    return resolved


__all__ = ["ToolCallUsage", "UsageTracker", "accumulate_actual_costs"]
