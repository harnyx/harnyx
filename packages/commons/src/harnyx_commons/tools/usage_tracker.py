"""Budget enforcement and usage accounting shared between services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.errors import BudgetExceededError
from harnyx_commons.tools.cost_accumulator import accumulate_costs, effective_cost, resolve_provider
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

        total_cost, provider_costs = accumulate_costs(
            budget.total_cost_usd,
            budget.cost_by_provider,
            usage=usage,
            cost_usd=cost_usd,
            normalized_tool_name=normalized_name,
        )

        actual_total_cost, actual_provider_costs = accumulate_actual_costs(
            budget.actual_total_cost_usd,
            budget.actual_cost_by_provider,
            reference_cost_usd=effective_cost(cost_usd, usage),
            actual_cost_usd=actual_cost_usd,
            actual_cost_provider=actual_cost_provider,
            usage=usage,
            normalized_tool_name=normalized_name,
        )

        return self._build_usage(
            budget=budget,
            llm_tokens=llm_tokens,
            usage_totals=usage_totals,
            reference_total_cost=total_cost,
            reference_provider_costs=provider_costs,
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
        reference_total_cost: float,
        reference_provider_costs: dict[str, float],
        actual_total_cost: float | None,
        actual_provider_costs: dict[str, float],
    ) -> SessionUsage:
        return SessionUsage(
            llm_tokens_last_call=llm_tokens,
            llm_usage_totals=usage_totals,
            total_cost_usd=reference_total_cost,
            cost_by_provider=reference_provider_costs,
            reference_total_cost_usd=reference_total_cost,
            reference_cost_by_provider=reference_provider_costs,
            actual_total_cost_usd=actual_total_cost,
            actual_cost_by_provider=actual_provider_costs,
        )


def accumulate_actual_costs(
    actual_total_cost_usd: float | None,
    actual_cost_by_provider: dict[str, float],
    *,
    reference_cost_usd: float | None,
    actual_cost_usd: float | None,
    actual_cost_provider: str | None,
    usage: ToolCallUsage | None,
    normalized_tool_name: str,
) -> tuple[float | None, dict[str, float]]:
    """Produce updated provider-billed cost totals.

    Unknown actual cost for a positive-reference call makes the aggregate
    unknown, while known actual costs still remain visible by provider.
    """
    if reference_cost_usd is None:
        return actual_total_cost_usd, dict(actual_cost_by_provider)
    if reference_cost_usd < 0.0:
        raise ValueError("reference_cost_usd must be non-negative")
    if reference_cost_usd == 0.0 and actual_cost_usd is None:
        return actual_total_cost_usd, dict(actual_cost_by_provider)
    if actual_cost_usd is None:
        return None, dict(actual_cost_by_provider)
    if actual_cost_usd < 0.0:
        raise ValueError("actual_cost_usd must be non-negative")

    provider = resolve_provider(
        usage,
        normalized_tool_name=normalized_tool_name,
        provider_override=actual_cost_provider,
    )
    provider_costs = dict(actual_cost_by_provider)
    provider_costs[provider] = provider_costs.get(provider, 0.0) + actual_cost_usd

    if actual_total_cost_usd is None:
        return None, provider_costs
    return actual_total_cost_usd + actual_cost_usd, provider_costs


__all__ = ["ToolCallUsage", "UsageTracker", "accumulate_actual_costs"]
