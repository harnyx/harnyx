"""Shared generation requests and reference output contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harnyx_commons.domain.miner_task import MinerTask, ReferenceAnswer
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import known_zero_actual_cost_tool_usage
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment

StageName = Literal["question_generation", "reference"]
ResponseMode = Literal["plain_text", "structured"]
CandidateFailureClass = Literal[
    "reasoning_no_generate",
    "transient_provider",
    "source_fetch_rejected",
    "source_extraction_limit",
    "source_unavailable",
    "contract_invalid",
]
_SOURCE_FAILURE_CLASSES = frozenset(
    {
        "source_fetch_rejected",
        "source_extraction_limit",
        "source_unavailable",
    }
)


class MinerTaskDatasetRequest(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    batch_id: UUID
    created_at: datetime | None = None
    minimum_task_total: int = Field(gt=0)
    plain_text_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    fast_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class RegisteredReferenceOutput(BaseModel):
    """Writer output; excerpts and public citation positions belong to the host."""

    model_config = COMMONS_STRICT_CONFIG
    answer_text: str | None = Field(min_length=1)
    structured_answer_json: str | None = Field(min_length=1)
    note: str | None = Field(max_length=80_000)
    citation_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=200)

    @field_validator("citation_evidence_ids", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("answer_text", "structured_answer_json", "note")
    @classmethod
    def _nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("reference fields must not be blank")
        return value

    @model_validator(mode="after")
    def _representation(self) -> RegisteredReferenceOutput:
        if (self.answer_text is None) == (self.structured_answer_json is None):
            raise ValueError("reference requires exactly one answer representation")
        return self


class BatchTerminalGenerationError(RuntimeError):
    """A provider/configuration fault that makes every sibling attempt invalid."""

    def __init__(
        self,
        failure_class: str,
        message: str,
        *,
        stage: StageName,
        tool_usage: ToolUsageSummary | None = None,
        stage_summaries: tuple[GenerationStageSummary, ...] = (),
        elapsed_ms: float = 0.0,
        actual_llm_cost_usd: float | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.stage = stage
        self.tool_usage = tool_usage or ToolUsageSummary.zero()
        self.stage_summaries = stage_summaries
        self.elapsed_ms = elapsed_ms
        self.actual_llm_cost_usd = actual_llm_cost_usd


class CandidateStageError(RuntimeError):
    """A terminal failure for one fresh candidate, never an in-place replay request."""

    def __init__(
        self,
        failure_class: CandidateFailureClass,
        stage: StageName,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        tool_usage: ToolUsageSummary | None = None,
        elapsed_ms: float = 0.0,
        actual_llm_cost_usd: float | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_class: CandidateFailureClass = failure_class
        self.stage: StageName = stage
        self.retry_after_seconds = retry_after_seconds
        self.tool_usage = tool_usage or ToolUsageSummary.zero()
        self.elapsed_ms = elapsed_ms
        self.actual_llm_cost_usd = actual_llm_cost_usd


class SourceSupport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class SeedSource(SourceSupport):
    publication_date: str


class Seed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    seed_id: str = Field(min_length=1)
    seed_question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    event_date: str
    supporting_sources: list[SeedSource] = Field(min_length=1)


class Seeds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seeds: list[Seed]


class QuestionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    answer_components: list[str] = Field(min_length=1)
    scope: str = Field(min_length=1)
    source_support: list[SourceSupport] = Field(min_length=1)
    revision_note: str = Field(min_length=1)

    @field_validator("answer_components")
    @classmethod
    def _components(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("answer components must be nonblank")
        return [value.strip() for value in values]


class StructuredQuestionDraft(QuestionDraft):
    output_schema_json: str = Field(min_length=1)
    structured_answer_json: str = Field(min_length=1)

    @model_validator(mode="after")
    def _contract(self) -> StructuredQuestionDraft:
        from .output_schema import validate_structured_draft

        validate_structured_draft(self.output_schema_json, self.structured_answer_json)
        return self


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    passed: bool = Field(alias="pass")
    feedback: str

    @model_validator(mode="after")
    def _feedback(self) -> ReviewDecision:
        if self.passed and self.feedback != "":
            raise ValueError("Pass requires empty feedback")
        if not self.passed and not self.feedback.strip():
            raise ValueError("Rejection requires actionable feedback")
        return self


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    answer: str = Field(min_length=1)
    request: dict[str, Any] = Field(default_factory=dict)
    provider_responses: list[dict[str, Any]] = Field(default_factory=list)
    search_worker_responses: list[dict[str, Any]] = Field(default_factory=list)

    def evidence_responses(self) -> list[dict[str, Any]]:
        """Keep completed evidence and stable pointers without abandoned payloads."""
        return [
            envelope
            if envelope.get("attempt_status", "completed") == "completed"
            else {"provider": envelope["provider"], "attempt_status": envelope["attempt_status"], "response": {}}
            for envelope in self.provider_responses
        ]


@dataclass(slots=True)
class CandidateSession:
    task_id: str
    effective_date: date
    cycle: int = 0
    attempt: int = 0
    mode: ResponseMode = "plain_text"
    author_history: list[Any] = field(default_factory=list)
    tool_usage: ToolUsageSummary = field(default_factory=known_zero_actual_cost_tool_usage)
    attempted_calls: int = 0
    missing_usage_calls: int = 0
    source_support: list[SourceSupport] = field(default_factory=list)
    stage_summaries: list[GenerationStageSummary] = field(default_factory=list)


class TerminalReference(BaseModel):
    model_config = COMMONS_STRICT_CONFIG
    reference_answer: ReferenceAnswer | None
    factual_correction: bool = False
    unsupported_reason: str | None = None


class GenerationStageSummary(BaseModel):
    model_config = COMMONS_STRICT_CONFIG
    stage: str
    outcome: str
    elapsed_ms: float
    provider: str | None = None
    model: str | None = None
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    attempted_calls: int = 0
    missing_usage_calls: int = 0
    available_cost_usd: float = 0


class FinalizedTask(BaseModel):
    model_config = COMMONS_STRICT_CONFIG
    task: MinerTask
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    stage_summaries: tuple[GenerationStageSummary, ...] = ()


class CycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    cycle: int
    draft: QuestionDraft | StructuredQuestionDraft
    review: ReviewDecision
    solver_results: dict[str, AgentResult] = Field(default_factory=dict)
    assessments: dict[str, FastJudgeAssessment] = Field(default_factory=dict)
    f1_scores: dict[str, float] = Field(default_factory=dict)
    format_results: dict[str, Any] = Field(default_factory=dict)
    analysis: str | None = None
    verification: ReviewDecision | None = None


class CandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    output_slot: int = 0
    seed: Seed | None
    mode: ResponseMode
    status: Literal["finalized", "exhausted", "unresolved_reference", "operational_failure"]
    cycles: list[CycleRecord] = Field(default_factory=list)
    reference: TerminalReference | None = None
    finalized: FinalizedTask | None = None
    error_type: str | None = None
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    attempted_calls: int = 0
    missing_usage_calls: int = 0
    elapsed_ms: float = 0
    stage_summaries: list[GenerationStageSummary] = Field(default_factory=list)


FinalizedTaskCallback = Callable[[int, FinalizedTask], Awaitable[None]]


class FinalizedTaskRejectedError(RuntimeError):
    """Acceptance definitively rejected the task without publishing it."""


class BatchGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    target_count: int
    candidates: list[CandidateResult] = Field(default_factory=list)
    tool_usage: ToolUsageSummary = Field(default_factory=known_zero_actual_cost_tool_usage)
    elapsed_ms: float = 0
    seed_attempted_calls: int = 0
    seed_missing_usage_calls: int = 0
    stage_summaries: list[GenerationStageSummary] = Field(default_factory=list)
    error_type: str | None = None

    @property
    def available_cost_usd(self) -> float:
        return sum(stage.available_cost_usd for stage in self.stage_summaries) + sum(
            stage.available_cost_usd for candidate in self.candidates for stage in candidate.stage_summaries
        )

    @property
    def finalized_tasks(self) -> tuple[FinalizedTask, ...]:
        return tuple(item.finalized for item in self.candidates if item.finalized is not None)

    @property
    def failure_counts(self) -> dict[str, int]:
        return dict(Counter(item.status for item in self.candidates if item.status != "finalized"))

    @property
    def round_count(self) -> int:
        return max((len(item.cycles) for item in self.candidates), default=0)

    @property
    def slot_attempt_count(self) -> int:
        return len(self.candidates)
