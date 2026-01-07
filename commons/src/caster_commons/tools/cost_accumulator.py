"""Cost accumulation utilities for tool usage tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from caster_commons.tools.usage_tracker import ToolCallUsage


def effective_cost(cost_usd: float | None, usage: ToolCallUsage | None) -> float | None:
    """Return the explicit cost or any cost embedded in the usage payload."""
    if cost_usd is not None:
        return float(cost_usd)
    if usage is None or usage.cost_usd is None:
        return None
    return float(usage.cost_usd)


def resolve_provider(
    usage: ToolCallUsage | None,
    normalized_tool_name: str,
    provider_override: str | None,
) -> str:
    if provider_override:
        return provider_override
    if usage and usage.provider:
        return usage.provider
    return "desearch" if normalized_tool_name.startswith("search") else "chutes"


def accumulate_costs(
    total_cost_usd: float,
    cost_by_provider: dict[str, float],
    *,
    usage: ToolCallUsage | None,
    cost_usd: float | None,
    normalized_tool_name: str,
    provider_override: str | None = None,
) -> tuple[float, dict[str, float]]:
    """Produce updated total and per-provider costs."""
    cost = effective_cost(cost_usd, usage)
    if cost is None:
        return total_cost_usd, dict(cost_by_provider)

    if cost < 0:
        raise ValueError("cost_usd must be non-negative")

    provider = resolve_provider(
        usage,
        normalized_tool_name=normalized_tool_name,
        provider_override=provider_override,
    )
    updated_total = total_cost_usd + cost

    provider_costs = dict(cost_by_provider)
    provider_costs[provider] = provider_costs.get(provider, 0.0) + cost
    return updated_total, provider_costs
