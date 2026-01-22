"""Shared Pydantic response shapes for tool-related HTTP endpoints."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic import JsonValue as PydanticJsonValue

from caster_miner_sdk.tools.types import ToolName


class ToolExecuteRequestDTO(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: UUID
    token: str
    tool: ToolName
    args: tuple[PydanticJsonValue, ...] = ()
    kwargs: dict[str, PydanticJsonValue] = {}


class ToolResultDTO(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    result_id: str
    url: str | None = None
    note: str | None = None
    title: str | None = None
    raw: PydanticJsonValue | None = None


class ToolExecuteResponseDTO(BaseModel):
    model_config = ConfigDict(frozen=True)

    receipt_id: str
    response: PydanticJsonValue
    results: tuple[ToolResultDTO, ...]
    result_policy: str
    cost_usd: float | None = None
    usage: ToolUsageDTO | None = None
    budget: ToolBudgetDTO


class ToolUsageDTO(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ToolBudgetDTO(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_budget_usd: float = Field(ge=0.0)
    session_used_budget_usd: float = Field(ge=0.0)
    session_remaining_budget_usd: float = Field(ge=0.0)


__all__ = [
    "ToolExecuteRequestDTO",
    "ToolResultDTO",
    "ToolExecuteResponseDTO",
    "ToolUsageDTO",
    "ToolBudgetDTO",
]
