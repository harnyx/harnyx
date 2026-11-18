"""Validator-owned judge LLM usage summaries."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

JudgeActualCostSource = Literal["provider_actual", "unavailable"]


def _validate_actual_cost_usd(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("actual_cost_usd must be numeric when supplied")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("actual_cost_usd must be finite when supplied")
    if numeric < 0.0:
        raise ValueError("actual_cost_usd must be non-negative when supplied")


@dataclass(frozen=True, slots=True)
class JudgeModelUsage:
    provider: str
    model: str
    call_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None
    actual_cost_usd: float | None
    actual_cost_source: JudgeActualCostSource
    actual_cost_provider: str | None = None
    actual_cost_evidence: str | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be blank")
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.reasoning_tokens is not None and self.reasoning_tokens < 0:
            raise ValueError("reasoning_tokens must be non-negative")
        _validate_actual_cost_usd(self.actual_cost_usd)
        if self.actual_cost_source == "provider_actual" and self.actual_cost_usd is None:
            raise ValueError("provider_actual judge usage requires actual_cost_usd")
        if self.actual_cost_source == "unavailable" and self.actual_cost_usd is not None:
            raise ValueError("unavailable judge usage must not include actual_cost_usd")


@dataclass(frozen=True, slots=True)
class JudgeUsageSummary:
    call_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int
    actual_cost_usd: float | None
    models: tuple[JudgeModelUsage, ...]

    def __post_init__(self) -> None:
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
        _validate_actual_cost_usd(self.actual_cost_usd)
        if self.call_count != sum(model.call_count for model in self.models):
            raise ValueError("judge usage call_count must equal model call counts")
        if self.prompt_tokens != sum(model.prompt_tokens for model in self.models):
            raise ValueError("judge usage prompt_tokens must equal model prompt tokens")
        if self.completion_tokens != sum(model.completion_tokens for model in self.models):
            raise ValueError("judge usage completion_tokens must equal model completion tokens")
        if self.total_tokens != sum(model.total_tokens for model in self.models):
            raise ValueError("judge usage total_tokens must equal model total tokens")
        if self.reasoning_tokens != _sum_reasoning_tokens(model.reasoning_tokens for model in self.models):
            raise ValueError("judge usage reasoning_tokens must equal model reasoning tokens")
        model_costs = tuple(model.actual_cost_usd for model in self.models)
        if any(cost is None for cost in model_costs):
            if self.actual_cost_usd is not None:
                raise ValueError("judge usage actual_cost_usd must be unavailable when any model cost is unavailable")
        else:
            model_cost_total = round(sum(cost for cost in model_costs if cost is not None), 12)
            if self.actual_cost_usd is None or not math.isclose(
                self.actual_cost_usd,
                model_cost_total,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("judge usage actual_cost_usd must equal model actual costs")


def _sum_reasoning_tokens(values: Iterable[int | None]) -> int:
    return sum(value or 0 for value in values)


__all__ = [
    "JudgeActualCostSource",
    "JudgeModelUsage",
    "JudgeUsageSummary",
]
