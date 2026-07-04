"""Typed contracts for the domain-tweak ADK harness."""

from __future__ import annotations

from math import ceil
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_generation import (
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionCandidate,
    DomainTweakReferenceAnswerCandidate,
)

DomainTweakAdkPhase = Literal["question_generation", "form_review", "reference_answer"]
DomainTweakAdkPromptKind = Literal["initial", "feedback", "soft_timeout_feedback"]
DomainTweakAdkTerminalStatus = Literal[
    "validated",
    "no_generate",
    "form_rejected",
    "validation_failed",
    "timeout",
    "invocation_error",
]
DomainTweakParsedOutput = (
    DomainTweakQuestionCandidate
    | DomainTweakFormReview
    | DomainTweakReferenceAnswerCandidate
    | None
)


class DomainTweakAdkRunConfig(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    provider: Literal["vertex"] = "vertex"
    model: str = Field(min_length=1)
    max_retries: int = Field(default=2, ge=0)
    phase_timeout_seconds: float = Field(default=900.0, gt=0)
    soft_timeout_seconds: float | None = Field(default=None, gt=0)
    soft_timeout_interval_seconds: float | None = Field(default=None, gt=0)
    app_name: str = Field(default="harnyx_domain_tweak_generation", min_length=1)
    user_id: str = Field(default="domain_tweak_generator", min_length=1)

    @model_validator(mode="after")
    def _soft_timeout_must_leave_hard_timeout_budget(self) -> DomainTweakAdkRunConfig:
        if self.soft_timeout_seconds is not None and self.soft_timeout_seconds >= self.phase_timeout_seconds:
            raise ValueError("soft_timeout_seconds must be lower than phase_timeout_seconds")
        if self.soft_timeout_interval_seconds is not None and self.soft_timeout_seconds is None:
            raise ValueError("soft_timeout_interval_seconds requires soft_timeout_seconds")
        if (
            self.soft_timeout_interval_seconds is not None
            and self.soft_timeout_interval_seconds >= self.phase_timeout_seconds
        ):
            raise ValueError("soft_timeout_interval_seconds must be lower than phase_timeout_seconds")
        return self


class DomainTweakNoGenerateDecision(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    no_generate: Literal[True] = True
    reason: str = Field(min_length=1)
    retry_recommended: bool = False


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

    validation_retries_per_pair: int = Field(default=2, ge=0)
    form_review_retries_per_pair: int = Field(default=2, ge=0)
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

    validation_retries_per_answer: int = Field(default=1, ge=0)
    invocation_retries_per_answer: int = Field(default=1, ge=0)
    failed_finalization_retries_per_batch_item: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=1800.0, gt=0)
    soft_timeout_seconds: float | None = Field(default=600.0, gt=0)
    soft_timeout_interval_seconds: float | None = Field(default=300.0, gt=0)
    answer_attempt_multiplier: float = Field(default=1.0, ge=1.0)
    hard_answer_attempt_cap_multiplier: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _soft_timeout_must_leave_hard_timeout_budget(self) -> DomainTweakReferenceAnswerPhasePolicy:
        if self.soft_timeout_seconds is not None and self.soft_timeout_seconds >= self.timeout_seconds:
            raise ValueError("soft_timeout_seconds must be lower than timeout_seconds")
        if self.soft_timeout_interval_seconds is not None and self.soft_timeout_seconds is None:
            raise ValueError("soft_timeout_interval_seconds requires soft_timeout_seconds")
        if (
            self.soft_timeout_interval_seconds is not None
            and self.soft_timeout_interval_seconds >= self.timeout_seconds
        ):
            raise ValueError("soft_timeout_interval_seconds must be lower than timeout_seconds")
        return self

    def hard_attempt_cap(self, target_count: int) -> int:
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        return self.hard_answer_attempt_cap_multiplier * target_count

    def invocation_attempt_cap(self, target_count: int) -> int:
        base_attempts = ceil(target_count * self.answer_attempt_multiplier)
        retry_attempts = target_count * (
            self.invocation_retries_per_answer + self.failed_finalization_retries_per_batch_item
        )
        return min(self.hard_attempt_cap(target_count), base_attempts + retry_attempts)


class DomainTweakBatchGenerationConfig(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    target_count: int = Field(gt=0)
    question_policy: DomainTweakQuestionPhasePolicy = Field(default_factory=DomainTweakQuestionPhasePolicy)
    reference_answer_policy: DomainTweakReferenceAnswerPhasePolicy = Field(
        default_factory=DomainTweakReferenceAnswerPhasePolicy
    )


class DomainTweakReviewedQuestion(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    pair_input: DomainTweakPairInput
    question_candidate: DomainTweakQuestionCandidate
    form_review: DomainTweakFormReview
    question_generation_result: DomainTweakAdkPhaseResult
    form_review_result: DomainTweakAdkPhaseResult
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)


class DomainTweakRejectedQuestionAttempt(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    pair_input: DomainTweakPairInput
    question_generation_result: DomainTweakAdkPhaseResult
    form_review_result: DomainTweakAdkPhaseResult | None = None
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)


class DomainTweakFinalizedTask(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    reviewed_question: DomainTweakReviewedQuestion
    reference_answer_result: DomainTweakAdkPhaseResult
    task: MinerTask
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)


class DomainTweakFailedFinalization(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    reviewed_question: DomainTweakReviewedQuestion
    reference_answer_results: tuple[DomainTweakAdkPhaseResult, ...] = ()
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)

    @field_validator("reference_answer_results", mode="before")
    @classmethod
    def _reference_answer_results_tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class DomainTweakBatchGenerationResult(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    target_count: int = Field(gt=0)
    selected_questions: tuple[DomainTweakReviewedQuestion, ...] = ()
    finalized_tasks: tuple[DomainTweakFinalizedTask, ...] = ()
    rejected_attempts: tuple[DomainTweakRejectedQuestionAttempt, ...] = ()
    failed_finalizations: tuple[DomainTweakFailedFinalization, ...] = ()
    reference_answer_finalization_attempt_count: int = Field(default=0, ge=0)
    reference_answer_retry_attempt_count: int = Field(default=0, ge=0)
    reference_answer_retry_round_count: int = Field(default=0, ge=0)
    underfilled: bool
    tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)

    @field_validator(
        "selected_questions",
        "finalized_tasks",
        "rejected_attempts",
        "failed_finalizations",
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
    "DomainTweakAdkPromptKind",
    "DomainTweakAdkPhaseResult",
    "DomainTweakAdkRunConfig",
    "DomainTweakAdkTerminalStatus",
    "DomainTweakBatchGenerationConfig",
    "DomainTweakBatchGenerationResult",
    "DomainTweakFailedFinalization",
    "DomainTweakFinalizedTask",
    "DomainTweakNoGenerateDecision",
    "DomainTweakParsedOutput",
    "DomainTweakQuestionPhasePolicy",
    "DomainTweakReferenceAnswerPhasePolicy",
    "DomainTweakRejectedQuestionAttempt",
    "DomainTweakReviewedQuestion",
    "DomainTweakValidationOutcome",
]
