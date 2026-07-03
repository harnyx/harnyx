"""Shared miner-task query/run value objects."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary

_JUDGE_USAGE_ADAPTER = TypeAdapter(JudgeUsageSummary)
_TOOL_USAGE_ADAPTER = TypeAdapter(ToolUsageSummary)
DEFAULT_MINER_TASK_BUDGET_USD = 0.5


class _TextModel(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    text: str = Field(min_length=1)


class Query(_TextModel):
    pass


class AnswerCitation(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    url: str = Field(min_length=1)
    note: str | None = None
    title: str | None = None


class ReferenceAnswer(_TextModel):
    citations: tuple[AnswerCitation, ...] | None = None

    @field_validator("citations", mode="before")
    @classmethod
    def _normalize_citations(
        cls,
        value: object,
    ) -> object:
        if value is None:
            return None
        if isinstance(value, list):
            return tuple(value)
        return value


class Response(_TextModel):
    citations: tuple[AnswerCitation, ...] | None = None

    @field_validator("citations", mode="before")
    @classmethod
    def _normalize_citations(
        cls,
        value: object,
    ) -> object:
        if value is None:
            return None
        if isinstance(value, list):
            return tuple(value)
        return value


class ScorerReasoning(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    text: str | None = Field(default=None, min_length=1)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class ScoreBreakdown(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    comparison_score: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)
    scoring_version: str = Field(min_length=1)
    reasoning: ScorerReasoning | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        similarity_score = normalized.pop("similarity_score", None)
        if similarity_score is not None and "total_score" in normalized:
            normalized["comparison_score"] = normalized["total_score"]
        return normalized

    @model_validator(mode="after")
    def _validate_total_matches_comparison(self) -> ScoreBreakdown:
        if self.total_score != self.comparison_score:
            raise ValueError("score breakdown total_score must equal comparison_score")
        return self


class MinerTaskErrorCode(StrEnum):
    # Shared serialized codes for miner-task pair outcomes.
    ARTIFACT_BREAKER_TRIPPED = "artifact_breaker_tripped"
    ARTIFACT_FETCH_FAILED = "artifact_fetch_failed"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    ARTIFACT_SETUP_FAILED = "artifact_setup_failed"
    ARTIFACT_SIZE_INVALID = "artifact_size_invalid"
    ARTIFACT_STAGING_FAILED = "artifact_staging_failed"
    BATCH_EXECUTION_FAILED = "batch_execution_failed"
    MINER_RESPONSE_INVALID = "miner_response_invalid"
    MINER_UNHANDLED_EXCEPTION = "miner_unhandled_exception"
    NEVER_RAN = "never_ran"
    PROGRESS_SNAPSHOT_FAILED = "progress_snapshot_failed"
    PROVIDER_BATCH_FAILURE = "provider_batch_failure"
    SANDBOX_FAILED = "sandbox_failed"
    SANDBOX_INVOCATION_FAILED = "sandbox_invocation_failed"
    SANDBOX_START_FAILED = "sandbox_start_failed"
    SCORING_LLM_RETRY_EXHAUSTED = "scoring_llm_retry_exhausted"
    SCRIPT_VALIDATION_FAILED = "script_validation_failed"
    SESSION_BUDGET_EXHAUSTED = "session_budget_exhausted"
    TIMEOUT_INCONCLUSIVE = "timeout_inconclusive"
    TIMEOUT_MINER_OWNED = "timeout_miner_owned"
    TOOL_PROVIDER_FAILED = "tool_provider_failed"
    UNEXPECTED_VALIDATOR_FAILURE = "unexpected_validator_failure"
    VALIDATOR_FAILED = "validator_failed"
    VALIDATOR_INTERNAL_TIMEOUT = "validator_internal_timeout"
    VALIDATOR_TIMEOUT = "validator_timeout"


DELIVERY_DISQUALIFYING_VALIDATOR_PAIR_ERROR_CODES: frozenset[MinerTaskErrorCode] = frozenset(
    (
        MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED,
        MinerTaskErrorCode.ARTIFACT_FETCH_FAILED,
        MinerTaskErrorCode.ARTIFACT_HASH_MISMATCH,
        MinerTaskErrorCode.ARTIFACT_STAGING_FAILED,
        MinerTaskErrorCode.ARTIFACT_SETUP_FAILED,
        MinerTaskErrorCode.SANDBOX_START_FAILED,
        MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
    )
)

MINER_ATTRIBUTED_PAIR_ERROR_CODES: frozenset[MinerTaskErrorCode] = frozenset(
    (
        MinerTaskErrorCode.MINER_RESPONSE_INVALID,
        MinerTaskErrorCode.MINER_UNHANDLED_EXCEPTION,
        MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED,
        MinerTaskErrorCode.SESSION_BUDGET_EXHAUSTED,
        MinerTaskErrorCode.TIMEOUT_MINER_OWNED,
        MinerTaskErrorCode.ARTIFACT_SIZE_INVALID,
    )
)


def is_delivery_disqualifying_validator_pair_error(code: MinerTaskErrorCode) -> bool:
    return code in DELIVERY_DISQUALIFYING_VALIDATOR_PAIR_ERROR_CODES


def is_miner_attributed_pair_error(code: MinerTaskErrorCode) -> bool:
    return code in MINER_ATTRIBUTED_PAIR_ERROR_CODES


class EvaluationError(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    code: MinerTaskErrorCode
    message: str = Field(min_length=1)

    @field_validator("code", mode="before")
    @classmethod
    def _normalize_code(
        cls,
        value: object,
    ) -> object:
        if isinstance(value, str):
            return MinerTaskErrorCode(value)
        return value


class EvaluationTrace(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    entrypoint_invocation_ms: float | None = Field(default=None, ge=0.0)
    scoring_ms: float | None = Field(default=None, ge=0.0)
    orchestration_ms: float | None = Field(default=None, ge=0.0)
    scoring_judge_selected_routes: tuple[str, ...] = ()
    scoring_judge_attempt_count: int | None = Field(default=None, ge=0)
    scoring_judge_retry_count: int | None = Field(default=None, ge=0)
    scoring_judge_retry_reasons: tuple[str, ...] = ()
    scoring_judge_duration_ms: float | None = Field(default=None, ge=0.0)
    scoring_judge_status: Literal["ok", "exhausted", "failed"] | None = None

    @field_validator("scoring_judge_selected_routes", "scoring_judge_retry_reasons", mode="before")
    @classmethod
    def _normalize_tuple_fields(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class EvaluationDetails(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    score_breakdown: ScoreBreakdown | None = None
    scoring_judge_usage: JudgeUsageSummary | None = None
    trace: EvaluationTrace | None = None
    total_tool_usage: ToolUsageSummary = Field(default_factory=ToolUsageSummary.zero)
    elapsed_ms: float | None = Field(default=None, ge=0.0)
    error: EvaluationError | None = None

    @field_validator("scoring_judge_usage", mode="before")
    @classmethod
    def _validate_scoring_judge_usage(cls, value: object) -> JudgeUsageSummary | None:
        if value is None:
            return None
        return _JUDGE_USAGE_ADAPTER.validate_python(value)

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
    "AnswerCitation",
    "DEFAULT_MINER_TASK_BUDGET_USD",
    "DELIVERY_DISQUALIFYING_VALIDATOR_PAIR_ERROR_CODES",
    "EvaluationDetails",
    "EvaluationTrace",
    "EvaluationError",
    "MINER_ATTRIBUTED_PAIR_ERROR_CODES",
    "MinerTask",
    "MinerTaskErrorCode",
    "Query",
    "ReferenceAnswer",
    "Response",
    "ScorerReasoning",
    "ScoreBreakdown",
    "is_delivery_disqualifying_validator_pair_error",
    "is_miner_attributed_pair_error",
]
