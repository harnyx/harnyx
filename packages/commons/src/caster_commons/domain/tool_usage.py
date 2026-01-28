"""Typed tool usage summaries for cost monitoring.

These dataclasses are shared across validator + platform boundaries to ensure
JSON payloads stored in Postgres (JSONB) are validated into a stable shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from caster_commons.domain.session import LlmUsageTotals


@dataclass(frozen=True, slots=True)
class SearchToolUsageSummary:
    call_count: int = 0
    cost: float = 0.0

    def __post_init__(self) -> None:
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative")
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True, slots=True)
class LlmModelUsageCost:
    usage: LlmUsageTotals = field(default_factory=LlmUsageTotals)
    cost: float = 0.0

    def __post_init__(self) -> None:
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True, slots=True)
class LlmUsageSummary:
    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    providers: dict[str, dict[str, LlmModelUsageCost]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True, slots=True)
class ToolUsageSummary:
    search_tool: SearchToolUsageSummary = field(default_factory=SearchToolUsageSummary)
    search_tool_cost: float = 0.0
    llm: LlmUsageSummary = field(default_factory=LlmUsageSummary)
    llm_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.search_tool_cost < 0.0:
            raise ValueError("search_tool_cost must be non-negative")
        if self.llm_cost < 0.0:
            raise ValueError("llm_cost must be non-negative")

    @classmethod
    def zero(cls) -> ToolUsageSummary:
        return cls()


__all__ = [
    "LlmModelUsageCost",
    "LlmUsageSummary",
    "SearchToolUsageSummary",
    "ToolUsageSummary",
]

