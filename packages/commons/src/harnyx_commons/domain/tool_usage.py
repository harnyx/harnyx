"""Typed tool usage summaries for cost monitoring.

These dataclasses are shared across validator + platform boundaries to ensure
JSON payloads stored in Postgres (JSONB) are validated into a stable shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnyx_commons.domain.session import LlmUsageTotals


@dataclass(frozen=True, slots=True)
class SearchToolUsageSummary:
    call_count: int = 0
    cost: float = 0.0
    reference_cost: float = 0.0
    actual_cost: float | None = None

    def __post_init__(self) -> None:
        if self.reference_cost == 0.0 and self.cost != 0.0:
            object.__setattr__(self, "reference_cost", self.cost)
        if self.cost == 0.0 and self.reference_cost != 0.0:
            object.__setattr__(self, "cost", self.reference_cost)
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative")
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")
        if self.reference_cost < 0.0:
            raise ValueError("reference_cost must be non-negative")
        if self.actual_cost is not None and self.actual_cost < 0.0:
            raise ValueError("actual_cost must be non-negative when supplied")


@dataclass(frozen=True, slots=True)
class LlmModelUsageCost:
    usage: LlmUsageTotals = field(default_factory=LlmUsageTotals)
    cost: float = 0.0
    reference_cost: float = 0.0
    actual_cost: float | None = None

    def __post_init__(self) -> None:
        if self.reference_cost == 0.0 and self.cost != 0.0:
            object.__setattr__(self, "reference_cost", self.cost)
        if self.cost == 0.0 and self.reference_cost != 0.0:
            object.__setattr__(self, "cost", self.reference_cost)
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")
        if self.reference_cost < 0.0:
            raise ValueError("reference_cost must be non-negative")
        if self.actual_cost is not None and self.actual_cost < 0.0:
            raise ValueError("actual_cost must be non-negative when supplied")


@dataclass(frozen=True, slots=True)
class LlmUsageSummary:
    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    reference_cost: float = 0.0
    actual_cost: float | None = None
    providers: dict[str, dict[str, LlmModelUsageCost]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_cost == 0.0 and self.cost != 0.0:
            object.__setattr__(self, "reference_cost", self.cost)
        if self.cost == 0.0 and self.reference_cost != 0.0:
            object.__setattr__(self, "cost", self.reference_cost)
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.reasoning_tokens < 0:
            raise ValueError("reasoning_tokens must be non-negative")
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")
        if self.reference_cost < 0.0:
            raise ValueError("reference_cost must be non-negative")
        if self.actual_cost is not None and self.actual_cost < 0.0:
            raise ValueError("actual_cost must be non-negative when supplied")


@dataclass(frozen=True, slots=True)
class ToolUsageSummary:
    search_tool: SearchToolUsageSummary = field(default_factory=SearchToolUsageSummary)
    search_tool_cost: float = 0.0
    llm: LlmUsageSummary = field(default_factory=LlmUsageSummary)
    llm_cost: float = 0.0
    reference_total_cost_usd: float = 0.0
    reference_cost_by_provider: dict[str, float] = field(default_factory=dict)
    actual_total_cost_usd: float | None = None
    actual_cost_by_provider: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        legacy_reference_total = self.llm_cost + self.search_tool_cost
        if self.reference_total_cost_usd == 0.0 and legacy_reference_total != 0.0:
            object.__setattr__(self, "reference_total_cost_usd", legacy_reference_total)
        if self.search_tool_cost < 0.0:
            raise ValueError("search_tool_cost must be non-negative")
        if self.llm_cost < 0.0:
            raise ValueError("llm_cost must be non-negative")
        if self.reference_total_cost_usd < 0.0:
            raise ValueError("reference_total_cost_usd must be non-negative")
        if self.actual_total_cost_usd is not None and self.actual_total_cost_usd < 0.0:
            raise ValueError("actual_total_cost_usd must be non-negative when supplied")

    @classmethod
    def zero(cls) -> ToolUsageSummary:
        return cls()


__all__ = [
    "LlmModelUsageCost",
    "LlmUsageSummary",
    "SearchToolUsageSummary",
    "ToolUsageSummary",
]
