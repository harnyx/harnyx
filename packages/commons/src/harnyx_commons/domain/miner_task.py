"""Shared miner-task query/run value objects."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary

_TOOL_USAGE_ADAPTER = TypeAdapter(ToolUsageSummary)
DEFAULT_MINER_TASK_BUDGET_USD = 0.5


class _TextModel(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    text: str = Field(min_length=1)


class Query(_TextModel):
    pass


class ReferenceAnswer(_TextModel):
    pass


class Response(_TextModel):
    pass


class ScoreBreakdown(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    comparison_score: float = Field(ge=0.0, le=1.0)
    similarity_score: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)
    scoring_version: str = Field(min_length=1)


class EvaluationError(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class EvaluationDetails(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    score_breakdown: ScoreBreakdown | None = None
    total_tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    elapsed_ms: float | None = Field(default=None, ge=0.0)
    error: EvaluationError | None = None

    @field_validator("total_tool_usage", mode="before")
    @classmethod
    def _validate_total_tool_usage(cls, value: object) -> ToolUsageSummary:
        return _TOOL_USAGE_ADAPTER.validate_python(value)

    @model_validator(mode="after")
    def _validate_state(self) -> EvaluationDetails:
        has_score_breakdown = self.score_breakdown is not None
        has_error = self.error is not None
        if has_score_breakdown == has_error:
            raise ValueError("evaluation details must include exactly one of score_breakdown or error")
        return self


class MinerTask(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    task_id: UUID
    query: Query
    reference_answer: ReferenceAnswer
    budget_usd: float = Field(default=DEFAULT_MINER_TASK_BUDGET_USD, ge=0.0)


__all__ = [
    "DEFAULT_MINER_TASK_BUDGET_USD",
    "EvaluationDetails",
    "EvaluationError",
    "MinerTask",
    "Query",
    "ReferenceAnswer",
    "Response",
    "ScoreBreakdown",
]
