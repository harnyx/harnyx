"""DTOs for tool execution shared across platform and validator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from caster_commons.domain.tool_call import ToolCall
from caster_commons.json_types import JsonValue
from caster_commons.tools.types import ToolName
from caster_commons.tools.usage_tracker import ToolCallUsage


@dataclass(frozen=True, slots=True)
class ToolBudgetSnapshot:
    """Session budget snapshot captured after executing a tool call."""

    session_budget_usd: float
    session_used_budget_usd: float
    session_remaining_budget_usd: float

    def __post_init__(self) -> None:
        if self.session_budget_usd < 0.0:
            raise ValueError("session_budget_usd must be non-negative")
        if self.session_used_budget_usd < 0.0:
            raise ValueError("session_used_budget_usd must be non-negative")
        if self.session_remaining_budget_usd < 0.0:
            raise ValueError("session_remaining_budget_usd must be non-negative")
        if self.session_used_budget_usd > self.session_budget_usd:
            raise ValueError("session_used_budget_usd must not exceed session_budget_usd")
        expected_remaining = self.session_budget_usd - self.session_used_budget_usd
        if abs(self.session_remaining_budget_usd - expected_remaining) > 1e-9:
            raise ValueError("session_remaining_budget_usd must equal budget - used")


@dataclass(frozen=True)
class ToolInvocationRequest:
    """Canonical payload describing a sandbox tool invocation."""

    session_id: UUID
    token: str
    tool: ToolName
    args: Sequence[JsonValue] = field(default_factory=tuple)
    kwargs: Mapping[str, JsonValue] = field(default_factory=dict)


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
]
