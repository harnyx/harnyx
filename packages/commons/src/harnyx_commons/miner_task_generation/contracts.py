"""Shared generation requests and reference output contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harnyx_commons.application.miner_response_hydration import MAX_CITATION_REFS
from harnyx_commons.domain.miner_task import AnswerCitation, MinerTask, ReferenceAnswer
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import known_zero_actual_cost_tool_usage
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment

StageName = Literal["question_generation"]
ResponseMode = Literal["plain_text", "structured"]
CandidateFailureClass = Literal[
    "transient_provider",
    "contract_invalid",
]


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
    generation_checkpoint: GenerationCheckpoint | None = None
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


class GenerationCheckpoint(BaseModel):
    """Small author-lifecycle projection; raw evidence belongs in trace storage."""

    model_config = COMMONS_STRICT_CONFIG
    candidate_id: str = Field(min_length=1, max_length=160)
    model: str = Field(default="gpt-6.1-sol", max_length=128)
    reasoning_effort: Literal["high"] = "high"
    prompt_sha256: str = Field(min_length=1, max_length=64)
    cwd: str | None = Field(default=None, max_length=4096)
    thread_id: str | None = Field(default=None, max_length=128)
    turn_id: str | None = Field(default=None, max_length=128)
    status: Literal["thread_created", "running", "testing", "tested", "completed", "failed", "cleanup"]
    round: int = Field(default=0, ge=0, le=5)
    draft_sha256: str | None = None
    f1_scores: dict[Literal["solver", "grounded_solver"], float] = Field(default_factory=dict)
    interrupt_acknowledged: bool | None = None
    remote_termination_confirmed: bool = False
    completed_at: datetime | None = None
    reconciled_terminal_ids: tuple[str, ...] = ()

    @field_validator("completed_at")
    @classmethod
    def _completion_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("Codex completion time must include a timezone")
        return value


MAX_REFERENCE_CORRECTION_RETRIES = 5


class ReferenceValidationError(ValueError):
    """An author reference can be corrected without changing its frozen tested draft."""


class AuthorReference(BaseModel):
    """Publishable answer and evidence constructed by the author for the frozen key."""

    model_config = COMMONS_STRICT_CONFIG
    answer_text: str | None = Field(
        min_length=1,
        max_length=80_000,
        description="Self-contained deep-research report answering the frozen plain-text question: "
        "conclusion, source-supported reasoning, necessary selection/comparison or calculation, and "
        "claim-level [[n]] citations. Preserve the tested facts and scope. For structured questions use "
        "null: the frozen JSON is the answer and the supporting research report belongs in note.",
    )
    citations: list[AnswerCitation] = Field(
        min_length=1,
        max_length=MAX_CITATION_REFS,
        description="Public source URLs and exact quotations, with offsets in the author's source text when known.",
    )
    note: str | None = Field(
        default=None,
        max_length=80000,
        description="For structured questions, the cited research report supporting the frozen JSON. "
        "For plain-text questions use null: the complete report belongs in answer_text.",
    )


class AuthorDeclaration(BaseModel):
    """Reference completion or explicit operational/exhaustion outcome; no new key."""

    model_config = COMMONS_STRICT_CONFIG
    status: Literal["completed", "exhausted", "error", "deadline"]
    reference: AuthorReference | None = None
    reconciliation: str = Field(min_length=1)
    unresolved_limits: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FinishDecision:
    eligible: bool
    outcome: Literal["finalized", "exhausted", "unresolved_reference", "operational_failure"]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AuthorOutcome:
    declaration: AuthorDeclaration
    turn_completed: bool
    checkpoint: GenerationCheckpoint


CheckpointCallback = Callable[[GenerationCheckpoint], Awaitable[None]]
TraceCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class ConstructionAuthorPort(Protocol):
    async def run(
        self,
        *,
        session: CandidateSession,
        mode: ResponseMode,
        deadline: float,
        challenge: Callable[[QuestionDraft, int], Awaitable[ChallengeFeedback]],
        finish: Callable[[AuthorDeclaration], Awaitable[FinishDecision]],
        checkpoint: CheckpointCallback,
    ) -> AuthorOutcome: ...

    async def aclose(self) -> None: ...


class FinalizedTask(BaseModel):
    model_config = COMMONS_STRICT_CONFIG
    task: MinerTask
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    stage_summaries: tuple[GenerationStageSummary, ...] = ()
    generation_metadata: GenerationCheckpoint | None = None


class CycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    cycle: int
    draft: QuestionDraft | StructuredQuestionDraft
    solver_results: dict[str, AgentResult] = Field(default_factory=dict)
    assessments: dict[str, FastJudgeAssessment] = Field(default_factory=dict)
    f1_scores: dict[str, float] = Field(default_factory=dict)
    format_results: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChallengeFeedback:
    round: int
    draft_sha256: str
    record: CycleRecord
    paired_miss: bool
    must_revise: bool
    rounds_remaining: int


class CandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    output_slot: int = 0
    mode: ResponseMode
    status: Literal["finalized", "exhausted", "unresolved_reference", "operational_failure"]
    cycles: list[CycleRecord] = Field(default_factory=list)
    reference: TerminalReference | None = None
    reference_consistency_assessments: list[FastJudgeAssessment] = Field(default_factory=list)
    finalized: FinalizedTask | None = None
    error_type: str | None = None
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    attempted_calls: int = 0
    missing_usage_calls: int = 0
    elapsed_ms: float = 0
    stage_summaries: list[GenerationStageSummary] = Field(default_factory=list)
    author_declaration: AuthorDeclaration | None = None
    generation_metadata: GenerationCheckpoint | None = None


FinalizedTaskCallback = Callable[[int, FinalizedTask], Awaitable[None]]


class FinalizedTaskRejectedError(RuntimeError):
    """Acceptance definitively rejected the task without publishing it."""


class BatchGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    target_count: int
    candidates: list[CandidateResult] = Field(default_factory=list)
    tool_usage: ToolUsageSummary = Field(default_factory=known_zero_actual_cost_tool_usage)
    elapsed_ms: float = 0
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
