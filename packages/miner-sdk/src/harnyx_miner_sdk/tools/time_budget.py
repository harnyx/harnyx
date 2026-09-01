"""Miner-visible execution time budget."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExecutionTimeBudgetDTO(BaseModel):
    """Configured full time limit for one miner invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    limit_seconds: float = Field(strict=True, gt=0.0, allow_inf_nan=False)


__all__ = ["ExecutionTimeBudgetDTO"]
