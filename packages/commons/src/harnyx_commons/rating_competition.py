"""Strict Platform/validator rating work and quality evidence contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.domain.miner_task import EvaluationTrace, Query, ScorerReasoning
from harnyx_commons.domain.tool_call import SearchToolResult, ToolCall, ToolResultPolicy
from harnyx_commons.tools.types import ToolName

MAX_RATING_JUDGMENT_BODY_BYTES = 1024 * 1024
RATING_JUDGMENT_ALREADY_ACCEPTED = "rating_judgment_already_accepted"


class RatingReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    receipt_id: str
    assignment_id: UUID
    tool: ToolName
    issued_at: datetime
    results: tuple[SearchToolResult, ...]

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> Self:
        if not call.is_successful() or call.details.result_policy is not ToolResultPolicy.REFERENCEABLE:
            raise ValueError("rating receipt must be successful and referenceable")
        results = tuple(result for result in call.details.results if isinstance(result, SearchToolResult))
        if len(results) != len(call.details.results):
            raise ValueError("rating receipt requires normalized search results")
        return cls(
            receipt_id=call.receipt_id,
            assignment_id=call.session_id,
            tool=call.tool,
            issued_at=call.issued_at,
            results=results,
        )


class RatingAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    assignment_id: UUID
    expected_hotkey: str = Field(min_length=1)
    callback_body_utf8: str
    signature_hex: str = Field(min_length=1)
    signed_callback_path: str = Field(pattern=r"^/")
    receipt_logs: tuple[RatingReceipt, ...]


class RatingWork(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    comparison_id: UUID
    query: Query
    first: RatingAnswer
    second: RatingAnswer

    @model_validator(mode="after")
    def distinct_assignments(self) -> Self:
        if self.first.assignment_id == self.second.assignment_id:
            raise ValueError("comparison requires distinct assignments")
        return self


class RatingQualityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    first_order_preference: Literal["first", "second"]
    second_order_preference: Literal["first", "second"]
    reasoning: ScorerReasoning | None
    judge_usage: JudgeUsageSummary
    evaluation_trace: EvaluationTrace | None = None

    @property
    def quality_result(self) -> Literal[-1, 0, 1]:
        wins = int(self.first_order_preference == "first") + int(self.second_order_preference == "second")
        return -1 if wins == 0 else 1 if wins == 2 else 0


class RatingJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    comparison_id: UUID
    quality_result: Literal[-1, 0, 1]
    two_order_evidence: RatingQualityEvidence

    @field_validator("quality_result", mode="before")
    @classmethod
    def integer_quality(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("quality_result must be an integer")
        return value

    @model_validator(mode="after")
    def consistent_quality(self) -> Self:
        if self.quality_result != self.two_order_evidence.quality_result:
            raise ValueError("quality result disagrees with two-order evidence")
        return self


class RatingWorkPage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    items: tuple[RatingWork, ...]
    next_after: UUID | None = None
