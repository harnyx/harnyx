"""Domain contracts for end-to-end miner-task generation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary

StageName = Literal["portfolio", "question_generation", "reference", "reference_repair", "audit"]
CapabilityPreference = Literal[
    "general_deep_research",
    "false_premise_correction",
    "source_conflict_time_uncertainty",
    "evidence_grounded_calculation_or_proof",
    "structured_field_semantics",
]
CAPABILITY_PREFERENCES: tuple[CapabilityPreference, ...] = (
    "general_deep_research",
    "false_premise_correction",
    "source_conflict_time_uncertainty",
    "evidence_grounded_calculation_or_proof",
    "structured_field_semantics",
)
ResponseMode = Literal["plain_text", "structured"]
CandidateFailureClass = Literal[
    "reasoning_no_generate",
    "transient_provider",
    "source_fetch_rejected",
    "source_extraction_limit",
    "source_unavailable",
    "contract_invalid",
    "proof_invalid",
    "audit_rejected",
]
AttemptOutcome = Literal[
    "finalized",
    "batch_terminal",
    "reasoning_no_generate",
    "transient_provider",
    "source_fetch_rejected",
    "source_extraction_limit",
    "source_unavailable",
    "contract_invalid",
    "proof_invalid",
    "audit_rejected",
]
PortfolioOutcome = Literal["succeeded", "batch_terminal", "transient_provider", "contract_invalid"]
_SOURCE_FAILURE_CLASSES = frozenset(
    {
        "source_fetch_rejected",
        "source_extraction_limit",
        "source_unavailable",
    }
)


class PortfolioAllocation(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    slot: int = Field(ge=0)
    ecosystems: tuple[str, ...] = Field(min_length=5, max_length=5)

    @field_validator("ecosystems", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PortfolioPacket(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    allocations: tuple[PortfolioAllocation, ...] = Field(min_length=1, max_length=10)

    @field_validator("allocations", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _unique_slots(self) -> PortfolioPacket:
        slots = tuple(item.slot for item in self.allocations)
        if len(slots) != len(set(slots)):
            raise ValueError("portfolio allocation slots must be unique")
        return self


class DossierRequirement(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    description: str = Field(min_length=1)


class DossierFact(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    statement: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class DossierAnswer(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    answer_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    value: str = Field(min_length=1)


class GroundedQuestionDossier(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    status: Literal["ready", "no_generate"]
    subject: str | None = Field(default=None, max_length=160)
    route_summary: str | None = Field(default=None, max_length=500)
    question: str | None = None
    answers: tuple[DossierAnswer, ...] = ()
    requirements: tuple[DossierRequirement, ...] = ()
    source_facts: tuple[DossierFact, ...] = ()
    derivation: str | None = None
    why_not_one_page: str | None = None
    substantive_final_condition: str | None = None
    response_mode: ResponseMode | None = None
    output_schema_json: str | None = Field(default=None, min_length=1)
    structured_answer_json: str | None = Field(default=None, min_length=1)
    failure_reason: str | None = None
    failure_class: Literal[
        "reasoning_no_generate",
        "source_fetch_rejected",
        "source_extraction_limit",
        "source_unavailable",
    ] | None = None
    source_failure_id: str | None = Field(default=None, pattern=r"^source_failure:[1-9][0-9]*$")

    @field_validator(
        "answers",
        "requirements",
        "source_facts",
        mode="before",
    )
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _status_contract(self) -> GroundedQuestionDossier:
        if self.status == "no_generate":
            if not self.failure_reason:
                raise ValueError("no_generate dossier requires failure_reason")
            if self.failure_class is None:
                raise ValueError("no_generate dossier requires failure_class")
            if self.failure_class in _SOURCE_FAILURE_CLASSES and self.source_failure_id is None:
                raise ValueError("source-related no_generate dossier requires source_failure_id")
            if self.failure_class == "reasoning_no_generate" and self.source_failure_id is not None:
                raise ValueError("reasoning_no_generate cannot contain source_failure_id")
            semantic_values = (
                self.subject,
                self.route_summary,
                self.question,
                self.derivation,
                self.why_not_one_page,
                self.substantive_final_condition,
                self.response_mode,
                self.output_schema_json,
                self.structured_answer_json,
            )
            if (
                any(value is not None for value in semantic_values)
                or self.answers
                or self.requirements
                or self.source_facts
            ):
                raise ValueError("no_generate dossier cannot contain question semantics")
            return self
        if self.failure_reason is not None:
            raise ValueError("ready dossier cannot contain failure_reason")
        if self.failure_class is not None:
            raise ValueError("ready dossier cannot contain failure_class")
        if self.source_failure_id is not None:
            raise ValueError("ready dossier cannot contain source_failure_id")
        if not all(
            (
                self.subject,
                self.route_summary,
                self.question,
                self.derivation,
                self.why_not_one_page,
                self.substantive_final_condition,
            )
        ):
            raise ValueError(
                "ready dossier requires subject, route, question, derivation, one-page explanation, and final condition"
            )
        if self.response_mode is None:
            raise ValueError("ready dossier requires response_mode")
        if self.response_mode == "plain_text" and (
            self.output_schema_json is not None or self.structured_answer_json is not None
        ):
            raise ValueError("plain_text dossier cannot contain structured schema or answer")
        if self.response_mode == "structured" and (
            self.output_schema_json is None or self.structured_answer_json is None
        ):
            raise ValueError("structured dossier requires output schema and answer hypothesis")
        if not self.requirements or not self.source_facts:
            raise ValueError("ready dossier requires load-bearing requirements and source facts")
        answer_ids = tuple(item.answer_id for item in self.answers)
        if not answer_ids or len(answer_ids) != len(set(answer_ids)):
            raise ValueError("ready dossier requires unique answer IDs")
        return self


class ProofStep(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    step_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    statement: str = Field(min_length=1)
    kind: Literal["supported", "derived"]
    evidence_ids: tuple[str, ...] = ()
    depends_on_step_ids: tuple[str, ...] = ()
    scan_certificate_ids: tuple[str, ...] = ()

    @field_validator("evidence_ids", "depends_on_step_ids", "scan_certificate_ids", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class ReferenceAnswerSelection(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    answer_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    corrected_value: str | None = Field(default=None, min_length=1)


class ReferenceProof(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    status: Literal["finalized", "giveup"]
    answer_text: str | None = Field(min_length=1)
    citation_evidence_ids: tuple[str, ...] = Field(max_length=200)
    answers: tuple[ReferenceAnswerSelection, ...] = ()
    proof_steps: tuple[ProofStep, ...] = ()
    structured_answer_json: str | None = Field(default=None, min_length=1)
    giveup_reason: str | None = None

    @field_validator("answers", "citation_evidence_ids", "proof_steps", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _status_contract(self) -> ReferenceProof:
        if self.status == "finalized" and (not self.answers or not self.proof_steps):
            raise ValueError("finalized proof requires answers and proof steps")
        if self.status == "finalized" and ((self.answer_text is None) == (self.structured_answer_json is None)):
            raise ValueError("finalized proof requires exactly one public answer representation")
        if self.status == "finalized" and self.giveup_reason is not None:
            raise ValueError("finalized proof cannot contain giveup_reason")
        if self.status == "giveup" and not self.giveup_reason:
            raise ValueError("giveup proof requires giveup_reason")
        if self.status == "giveup" and (
            self.answer_text is not None or self.structured_answer_json is not None or self.citation_evidence_ids
        ):
            raise ValueError("giveup proof cannot contain a public answer or citation positions")
        return self


class AuditResult(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    status: Literal["pass", "reject"]
    defects: tuple[str, ...] = ()
    explanation: str = Field(min_length=1)

    @field_validator("defects", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _status_contract(self) -> AuditResult:
        if self.status == "pass" and self.defects:
            raise ValueError("passing audit cannot contain defects")
        if self.status == "reject" and not self.defects:
            raise ValueError("rejected audit requires concrete defects")
        return self


class DomainTweakStageSummary(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    stage: StageName
    outcome: str = Field(min_length=1)
    elapsed_ms: float = Field(ge=0)


class AcceptedRouteContext(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    subject: str = Field(min_length=1, max_length=160)
    route_summary: str = Field(min_length=1, max_length=500)
    source_urls: tuple[str, ...] = Field(min_length=1)

    @field_validator("source_urls", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("source_urls")
    @classmethod
    def _valid_source_urls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("accepted route source URLs must be unique")
        if any(len(item) > 2_048 for item in value):
            raise ValueError("accepted route source URL exceeds 2048 characters")
        return value


class DomainTweakFinalizedTask(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    task: MinerTask
    stage_summaries: tuple[DomainTweakStageSummary, ...] = ()
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    repaired: bool = False
    route_context: AcceptedRouteContext | None = Field(default=None, exclude=True)

    @field_validator("stage_summaries", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PortfolioCallEvent(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    portfolio_call_id: str = Field(min_length=1)
    round_index: int = Field(gt=0)
    group_index: int = Field(ge=0)
    slot_count: int = Field(gt=0, le=10)
    outcome: PortfolioOutcome
    elapsed_ms: float = Field(ge=0)
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    retry_after_seconds: float | None = Field(default=None, ge=0)
    failure_class: str | None = Field(default=None, max_length=64)


class SlotAttemptEvent(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    attempt_id: str = Field(min_length=1)
    round_index: int = Field(gt=0)
    output_slot: int = Field(ge=0)
    outcome: AttemptOutcome
    terminal_stage: StageName
    elapsed_ms: float = Field(ge=0)
    stage_summaries: tuple[DomainTweakStageSummary, ...] = ()
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    retry_after_seconds: float | None = Field(default=None, ge=0)
    portfolio_call_id: str | None = None
    failure_class: str | None = Field(default=None, max_length=64)
    repaired: bool = False
    failure_reason: str | None = Field(default=None, exclude=True)
    source_failure_id: str | None = Field(
        default=None,
        pattern=r"^source_failure:[1-9][0-9]*$",
        exclude=True,
    )

    @field_validator("stage_summaries", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class DomainTweakBatchGenerationResult(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    target_count: int = Field(gt=0)
    finalized_tasks: tuple[DomainTweakFinalizedTask, ...] = ()
    portfolio_call_count: int = Field(default=0, ge=0)
    slot_attempt_count: int = Field(default=0, ge=0)
    round_count: int = Field(default=0, ge=0)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    elapsed_ms: float = Field(default=0, ge=0)

    @field_validator("finalized_tasks", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _exact_finalized_count(self) -> DomainTweakBatchGenerationResult:
        if len(self.finalized_tasks) != self.target_count:
            raise ValueError("finalized task count must equal target_count")
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
        stage_summaries: tuple[DomainTweakStageSummary, ...] = (),
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


@dataclass(frozen=True, slots=True)
class CandidateFailure:
    failure_class: CandidateFailureClass
    terminal_stage: StageName
    stage_summaries: tuple[DomainTweakStageSummary, ...]
    tool_usage: ToolUsageSummary = field(default_factory=ToolUsageSummary.zero)
    retry_after_seconds: float | None = None
    failure_reason: str | None = None
    source_failure_id: str | None = None


CandidateOutcome = DomainTweakFinalizedTask | CandidateFailure
DomainTweakFinalizedTaskCallback = Callable[[int, DomainTweakFinalizedTask], Awaitable[None]]
PortfolioCallCallback = Callable[[PortfolioCallEvent], Awaitable[None]]
SlotAttemptCallback = Callable[[SlotAttemptEvent], Awaitable[None]]

TOutput = TypeVar("TOutput", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class StageRunResult:
    output: BaseModel
    elapsed_ms: float
    tool_usage: ToolUsageSummary
    retry_after_seconds: float | None = None
    validation_repaired: bool = False
    actual_llm_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class AgentToolSet:
    allowed_tools: tuple[str, ...] = ()
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    search_result_registrar: Callable[[object], str] | None = None


__all__ = [
    "AgentToolSet",
    "AcceptedRouteContext",
    "AuditResult",
    "BatchTerminalGenerationError",
    "CandidateFailure",
    "CandidateFailureClass",
    "CandidateOutcome",
    "CandidateStageError",
    "DossierAnswer",
    "DossierFact",
    "DossierRequirement",
    "DomainTweakBatchGenerationResult",
    "DomainTweakFinalizedTask",
    "DomainTweakFinalizedTaskCallback",
    "DomainTweakStageSummary",
    "GroundedQuestionDossier",
    "PortfolioAllocation",
    "PortfolioCallCallback",
    "PortfolioCallEvent",
    "PortfolioPacket",
    "ProofStep",
    "ReferenceAnswerSelection",
    "ReferenceProof",
    "SlotAttemptCallback",
    "SlotAttemptEvent",
    "StageName",
    "StageRunResult",
]
