"""DTOs for tool execution shared across platform and validator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_commons.json_types import JsonValue
from harnyx_commons.tools.types import ToolName
from harnyx_commons.tools.usage_tracker import ToolCallUsage


@dataclass(frozen=True, slots=True)
class ToolBudgetSnapshot:
    """Budget values captured from a tool session at one point in time."""

    session_budget_usd: float
    session_hard_limit_usd: float
    session_used_budget_usd: float
    session_remaining_budget_usd: float

    def __post_init__(self) -> None:
        if self.session_budget_usd < 0.0:
            raise ValueError("session_budget_usd must be non-negative")
        if self.session_hard_limit_usd < 0.0:
            raise ValueError("session_hard_limit_usd must be non-negative")
        if self.session_used_budget_usd < 0.0:
            raise ValueError("session_used_budget_usd must be non-negative")
        if self.session_remaining_budget_usd < 0.0:
            raise ValueError("session_remaining_budget_usd must be non-negative")
        if self.session_hard_limit_usd < self.session_budget_usd:
            raise ValueError("session_hard_limit_usd must be greater than or equal to session_budget_usd")
        expected_remaining = max(self.session_budget_usd - self.session_used_budget_usd, 0.0)
        if abs(self.session_remaining_budget_usd - expected_remaining) > 1e-9:
            raise ValueError(
                "session_remaining_budget_usd must equal "
                "max(session_budget_usd - session_used_budget_usd, 0)"
            )


def session_budget_snapshot(session: Session) -> ToolBudgetSnapshot:
    """Project the current authoritative values from a tool session."""

    used_budget_usd = session.usage.total_cost_usd
    return ToolBudgetSnapshot(
        session_budget_usd=session.budget_usd,
        session_hard_limit_usd=session.effective_hard_limit_usd,
        session_used_budget_usd=used_budget_usd,
        session_remaining_budget_usd=max(session.budget_usd - used_budget_usd, 0.0),
    )


@dataclass(frozen=True)
class ToolInvocationRequest:
    """Canonical payload describing a sandbox tool invocation."""

    session_id: UUID
    token: str
    tool: ToolName
    args: Sequence[JsonValue] = field(default_factory=tuple)
    kwargs: Mapping[str, JsonValue] = field(default_factory=dict)


def tool_payload_from_args_kwargs(
    args: Sequence[JsonValue],
    kwargs: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    if kwargs:
        return dict(kwargs)
    if args:
        first = args[0]
        if not isinstance(first, dict):
            raise TypeError("expected JSON object payload as first positional argument")
        for key in first:
            if not isinstance(key, str):
                raise TypeError("expected JSON object with string keys")
        return dict(first)
    return {}


def tool_payload_for_invocation(request: ToolInvocationRequest) -> dict[str, JsonValue]:
    return tool_payload_from_args_kwargs(request.args, request.kwargs)


@dataclass(frozen=True)
class ToolInvocationResult:
    """Result of a tool invocation."""

    receipt: ToolCall
    response_payload: JsonValue
    budget: ToolBudgetSnapshot
    usage: ToolCallUsage | None = None


__all__ = [
    "ToolBudgetSnapshot",
    "ToolInvocationRequest",
    "ToolInvocationResult",
    "session_budget_snapshot",
    "tool_payload_for_invocation",
    "tool_payload_from_args_kwargs",
]
