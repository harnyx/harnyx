"""Typed contracts for source-aware domain-tweak generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_generation import (
    DomainTweakFormBlueprint,
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionPacket,
    DomainTweakReferenceAnswerOutput,
    DomainTweakSemanticSupportReview,
)

DomainTweakAdkPhase = Literal[
    "form_blueprint",
    "question_generation",
    "form_review",
    "reference_answer_generation",
    "semantic_support_gate",
]
DomainTweakPipelineStage = Literal[
    "form_blueprint",
    "question_generation",
    "form_review",
    "initial_evidence_acquisition",
    "reference_answer_generation",
    "deterministic_evidence_delivery",
    "semantic_support_gate",
    "citation_hydration",
]
DomainTweakFinalizationStage = Literal[
    "initial_evidence_acquisition",
    "reference_answer_generation",
    "deterministic_evidence_delivery",
    "semantic_support_gate",
    "citation_hydration",
]
DomainTweakAdkPromptKind = Literal["initial", "feedback"]
DomainTweakAdkTerminalStatus = Literal[
    "validated",
    "no_generate",
    "form_rejected",
    "abandoned",
    "semantic_rejected",
    "validation_failed",
    "timeout",
    "invocation_error",
]
DomainTweakDiscardReason = Literal[
    "no_generate",
    "form_rejected",
    "duplicate_question",
    "validation_failed",
    "timeout",
    "invocation_error",
    "source_unavailable",
    "source_content_limit",
    "source_declared_url_limit",
    "source_url_invalid",
    "source_tool_call_limit",
    "source_tool_response_limit",
    "source_supplemental_url_limit",
    "source_acquisition_round_limit",
    "reference_abandon",
    "delivery_validation_failed",
    "semantic_support_abandon",
    "citation_hydration_failed",
]
DomainTweakParsedOutput = (
    DomainTweakFormBlueprint
    | DomainTweakQuestionPacket
    | DomainTweakFormReview
    | DomainTweakReferenceAnswerOutput
    | DomainTweakSemanticSupportReview
    | None
)


class DomainTweakAdkRunConfig(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    provider: Literal["vertex"] = "vertex"
    model: str = Field(min_length=1)
    max_retries: int = Field(default=2, ge=0)
    phase_timeout_seconds: float = Field(default=600.0, gt=0)
    app_name: str = Field(default="harnyx_domain_tweak_generation", min_length=1)
    user_id: str = Field(default="domain_tweak_generator", min_length=1)


class DomainTweakAdkEventSummary(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    is_final_response: bool = False
    function_call_names: tuple[str, ...] = ()
    function_response_names: tuple[str, ...] = ()
    content_text_preview: str | None = Field(default=None, max_length=500)
    content_text_length: int = Field(default=0, ge=0)
    usage: LlmUsage = Field(default_factory=LlmUsage)
    web_search_queries: tuple[str, ...] = ()
    web_search_query_count: int = Field(default=0, ge=0)

    @field_validator("function_call_names", "function_response_names", "web_search_queries", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class DomainTweakValidationOutcome(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    ok: bool
    terminal_status: DomainTweakAdkTerminalStatus
    parsed_output: DomainTweakParsedOutput = None
    feedback: tuple[str, ...] = ()
    error_type: str | None = None
    error: str | None = None

    @field_validator("feedback", mode="before")
    @classmethod
    def _feedback_tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(str(item) for item in value)
        return value


class DomainTweakAdkAttempt(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    attempt_index: int = Field(ge=0)
    prompt_kind: DomainTweakAdkPromptKind
    final_text_preview: str = Field(default="", max_length=500)
    final_text_length: int = Field(default=0, ge=0)
    validation_ok: bool
    validation_feedback: tuple[str, ...] = ()
    event_summaries: tuple[DomainTweakAdkEventSummary, ...] = ()
    event_summary_count: int = Field(default=0, ge=0)
    event_summaries_truncated: bool = False
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)

    @field_validator("validation_feedback", "event_summaries", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class DomainTweakAdkPhaseResult(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    phase: DomainTweakAdkPhase
    terminal_status: DomainTweakAdkTerminalStatus
    parsed_output: DomainTweakParsedOutput = None
    attempts: tuple[DomainTweakAdkAttempt, ...] = ()
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    error_type: str | None = None
    error: str | None = None

    @field_validator("attempts", mode="before")
    @classmethod
    def _attempts_tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class DomainTweakQuestionPhasePolicy(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    validation_retries_per_stage: int = Field(default=2, ge=0)
    timeout_seconds: float = Field(default=600.0, gt=0)
    target_attempt_multiplier: float = Field(default=3.0, gt=0)
    underfill_extra_passes: int = Field(default=3, ge=0)
    hard_attempt_cap_multiplier: int = Field(default=4, ge=1)

    def hard_attempt_cap(self, target_count: int) -> int:
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        return self.hard_attempt_cap_multiplier * target_count


class DomainTweakReferenceAnswerPhasePolicy(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    timeout_seconds: float = Field(default=600.0, gt=0)


class DomainTweakSourceEvidencePolicy(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    max_cached_chars_per_source: int = Field(default=200_000, gt=0)
    max_initial_prompt_chars: int = Field(default=30_000, gt=0)
    max_tool_response_chars: int = Field(default=10_000, gt=0)
    max_tool_calls_per_invocation: int = Field(default=30, gt=0)
    max_supplemental_urls_per_invocation: int = Field(default=10, ge=0)
    max_acquisition_rounds_per_claim: int = Field(default=3, ge=0)


class DomainTweakBatchGenerationConfig(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    target_count: int = Field(gt=0)
    question_policy: DomainTweakQuestionPhasePolicy = Field(default_factory=DomainTweakQuestionPhasePolicy)
    reference_answer_policy: DomainTweakReferenceAnswerPhasePolicy = Field(
        default_factory=DomainTweakReferenceAnswerPhasePolicy
    )
    source_evidence_policy: DomainTweakSourceEvidencePolicy = Field(
        default_factory=DomainTweakSourceEvidencePolicy
    )

    @property
    def candidate_attempt_cap(self) -> int:
        return self.question_policy.hard_attempt_cap(self.target_count)


class DomainTweakStageSummary(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    stage: DomainTweakPipelineStage
    outcome: str = Field(min_length=1)
    elapsed_ms: float = Field(default=0.0, ge=0.0)


class DomainTweakSourceEvidenceSummary(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    provider_request_count: int = Field(default=0, ge=0)
    provider_billable_units: int = Field(default=0, ge=0)
    provider_cost_usd: float = Field(default=0.0, ge=0.0)
    submitted_url_count: int = Field(default=0, ge=0)
    acquired_source_count: int = Field(default=0, ge=0)
    failed_source_count: int = Field(default=0, ge=0)
    positive_cache_hit_count: int = Field(default=0, ge=0)
    negative_cache_hit_count: int = Field(default=0, ge=0)
    initial_window_count: int = Field(default=0, ge=0)
    source_tool_call_count: int = Field(default=0, ge=0)
    cached_source_read_count: int = Field(default=0, ge=0)
    supplemental_acquisition_call_count: int = Field(default=0, ge=0)
    supplemental_url_count: int = Field(default=0, ge=0)
    acquisition_round_count: int = Field(default=0, ge=0)
    round_limit_activation_count: int = Field(default=0, ge=0)


class DomainTweakReviewedQuestion(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    pair_input: DomainTweakPairInput
    form_blueprint: DomainTweakFormBlueprint
    question_packet: DomainTweakQuestionPacket
    form_review: DomainTweakFormReview
    form_blueprint_result: DomainTweakAdkPhaseResult
    question_generation_result: DomainTweakAdkPhaseResult
    form_review_result: DomainTweakAdkPhaseResult
    stage_summaries: tuple[DomainTweakStageSummary, ...] = ()
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)

    @field_validator("stage_summaries", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class DomainTweakRejectedQuestionAttempt(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    pair_input: DomainTweakPairInput
    terminal_stage: Literal["form_blueprint", "question_generation", "form_review"]
    reason: DomainTweakDiscardReason
    form_blueprint_result: DomainTweakAdkPhaseResult
    question_generation_result: DomainTweakAdkPhaseResult | None = None
    form_review_result: DomainTweakAdkPhaseResult | None = None
    stage_summaries: tuple[DomainTweakStageSummary, ...] = ()
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)

    @field_validator("stage_summaries", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class DomainTweakFinalizedTask(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    reviewed_question: DomainTweakReviewedQuestion
    reference_answer_result: DomainTweakAdkPhaseResult
    semantic_support_result: DomainTweakAdkPhaseResult
    task: MinerTask
    stage_summaries: tuple[DomainTweakStageSummary, ...] = ()
    source_evidence: DomainTweakSourceEvidenceSummary = Field(
        default_factory=DomainTweakSourceEvidenceSummary
    )
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)

    @field_validator("stage_summaries", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class DomainTweakDiscardedCandidate(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    reviewed_question: DomainTweakReviewedQuestion
    terminal_stage: DomainTweakFinalizationStage
    reason: DomainTweakDiscardReason
    reference_answer_result: DomainTweakAdkPhaseResult | None = None
    semantic_support_result: DomainTweakAdkPhaseResult | None = None
    stage_summaries: tuple[DomainTweakStageSummary, ...] = ()
    source_evidence: DomainTweakSourceEvidenceSummary = Field(
        default_factory=DomainTweakSourceEvidenceSummary
    )
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)

    @field_validator("stage_summaries", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class DomainTweakBatchGenerationResult(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    target_count: int = Field(gt=0)
    selected_questions: tuple[DomainTweakReviewedQuestion, ...] = ()
    finalized_tasks: tuple[DomainTweakFinalizedTask, ...] = ()
    rejected_attempts: tuple[DomainTweakRejectedQuestionAttempt, ...] = ()
    discarded_candidates: tuple[DomainTweakDiscardedCandidate, ...] = ()
    candidate_finalization_attempt_count: int = Field(default=0, ge=0)
    underfilled: bool
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)

    @field_validator(
        "selected_questions",
        "finalized_tasks",
        "rejected_attempts",
        "discarded_candidates",
        mode="before",
    )
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


__all__ = [
    "DomainTweakAdkAttempt",
    "DomainTweakAdkEventSummary",
    "DomainTweakAdkPhase",
    "DomainTweakAdkPhaseResult",
    "DomainTweakAdkPromptKind",
    "DomainTweakAdkRunConfig",
    "DomainTweakAdkTerminalStatus",
    "DomainTweakBatchGenerationConfig",
    "DomainTweakBatchGenerationResult",
    "DomainTweakDiscardReason",
    "DomainTweakDiscardedCandidate",
    "DomainTweakFinalizedTask",
    "DomainTweakFinalizationStage",
    "DomainTweakParsedOutput",
    "DomainTweakPipelineStage",
    "DomainTweakQuestionPhasePolicy",
    "DomainTweakReferenceAnswerPhasePolicy",
    "DomainTweakRejectedQuestionAttempt",
    "DomainTweakReviewedQuestion",
    "DomainTweakSourceEvidencePolicy",
    "DomainTweakSourceEvidenceSummary",
    "DomainTweakStageSummary",
    "DomainTweakValidationOutcome",
]
