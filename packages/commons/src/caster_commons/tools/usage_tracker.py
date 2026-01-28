"""Budget enforcement and usage accounting shared between services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from caster_commons.domain.session import Session, SessionStatus, SessionUsage
from caster_commons.errors import BudgetExceededError
from caster_commons.tools.budget_validator import BudgetValidator
from caster_commons.tools.cost_accumulator import accumulate_costs
from caster_commons.tools.llm_usage_accumulator import accumulate_llm_usage
from caster_commons.tools.types import ToolName

if TYPE_CHECKING:
    from caster_commons.domain.session import LlmUsageTotals


@dataclass(frozen=True, slots=True)
class ToolCallUsage:
    """Structured LLM usage metadata captured from a tool response."""

    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


class UsageTracker:
    """Enforces per-session budget limits for tool invocations."""

    def record_tool_call(
        self,
        session: Session,
        *,
        tool_name: ToolName,
        llm_tokens: int,
        usage: ToolCallUsage | None = None,
        cost_usd: float | None = None,
    ) -> Session:
        self._validate_session(session, llm_tokens)
        normalized_name, usage_details = self._prepare_usage(tool_name, usage, cost_usd)

        updated_usage = self._update_usage(
            session.usage,
            session_budget_usd=session.budget_usd,
            normalized_name=normalized_name,
            llm_tokens=llm_tokens,
            usage=usage_details,
            cost_usd=cost_usd,
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
        session_budget_usd: float,
        normalized_name: str,
        llm_tokens: int,
        usage: ToolCallUsage | None,
        cost_usd: float | None,
    ) -> SessionUsage:
        budget_validator = BudgetValidator(session_budget_usd)
        projected_total = budget.total_cost_usd + (cost_usd or 0.0)
        budget_validator.assert_within_limits(projected_total)

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
        budget_validator.assert_within_limits(total_cost)

        return self._build_usage(
            budget=budget,
            llm_tokens=llm_tokens,
            usage_totals=usage_totals,
            total_cost=total_cost,
            provider_costs=provider_costs,
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
    ) -> SessionUsage:
        return SessionUsage(
            llm_tokens_last_call=llm_tokens,
            llm_usage_totals=usage_totals,
            total_cost_usd=total_cost,
            cost_by_provider=provider_costs,
        )

__all__ = ["ToolCallUsage", "UsageTracker"]
