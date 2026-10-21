"""Shared Pydantic shapes for tool-related HTTP endpoints."""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, TypeAdapter

from harnyx_commons.llm.schema import LlmUsage
from harnyx_miner_sdk.tools.http_models import (
    ToolBudgetDTO,
    ToolExecuteRequestDTO,
    ToolExecuteResponseDTO,
    ToolResultDTO,
    ToolUsageDTO,
)

# HTTP JSON contains an object; strict enclosing DTOs otherwise require a
# preconstructed dataclass. Parse once through the canonical usage type.
LlmUsageDTO = Annotated[LlmUsage, BeforeValidator(TypeAdapter(LlmUsage).validate_python)]

__all__ = [
    "LlmUsageDTO",
    "ToolBudgetDTO",
    "ToolExecuteRequestDTO",
    "ToolExecuteResponseDTO",
    "ToolResultDTO",
    "ToolUsageDTO",
]
