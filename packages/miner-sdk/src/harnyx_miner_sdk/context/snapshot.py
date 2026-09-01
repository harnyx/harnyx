"""Typed invocation context exposed to sandboxed miner queries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from harnyx_miner_sdk.tools.http_models import ToolBudgetDTO
from harnyx_miner_sdk.tools.time_budget import ExecutionTimeBudgetDTO


class ContextSnapshot(BaseModel):
    """Immutable cost and time budgets supplied to one miner query."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    cost_budget: ToolBudgetDTO
    time_budget: ExecutionTimeBudgetDTO


__all__ = ["ContextSnapshot"]
