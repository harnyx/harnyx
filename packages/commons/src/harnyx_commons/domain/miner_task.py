"""Shared miner-task query/run value objects."""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import StrEnum
from typing import Literal, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_miner_sdk.json_types import JsonValue
from harnyx_miner_sdk.query import Query
from harnyx_miner_sdk.structured_output import compact_json, validate_output_size

_JUDGE_USAGE_ADAPTER = TypeAdapter(JudgeUsageSummary)
_TOOL_USAGE_ADAPTER = TypeAdapter(ToolUsageSummary)
DEFAULT_MINER_TASK_BUDGET_USD = 0.5
_POSITIONAL_CITATIONS_DESCRIPTION = (
    "Hydrated submitted citation positions in order. Miners submit only non-null CitationRef entries. "
    "An AnswerCitation means that the submitted position resolved to authoritative public evidence; null means "
    "that the submitted position could not be resolved or hydrated. A null provides no factual support, and "
    "submitted positions are never deleted, renumbered, or remapped."
)
_RESPONSE_NOTE_DESCRIPTION = (
    "Optional public supplementary content that may explain, qualify, support, or correct the required answer. "
    "It cannot replace or repair a missing or invalid answer. Factual claims use the same citations array."
)


class _TextModel(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    text: str = Field(min_length=1)


class AnswerCitation(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    url: str = Field(min_length=1)
    note: str | None = None
    title: str | None = None


class ReferenceAnswer(_TextModel):
    note: str | None = Field(
        default=None,
        max_length=80_000,
        description=_RESPONSE_NOTE_DESCRIPTION,
        exclude_if=lambda value: value is None,
    )
    citations: tuple[AnswerCitation | None, ...] | None = Field(
        default=None,
        description=_POSITIONAL_CITATIONS_DESCRIPTION,
    )

    @field_validator("note")
    @classmethod
    def _validate_note(cls, value: str | None) -> str | None:
        return _normalize_response_note(value)

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


class Response(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_mode_override="validation",
        strict=True,
        str_strip_whitespace=False,
    )

    text: str | None = Field(default=None, max_length=80_000, exclude_if=lambda value: value is None)
    output: JsonValue | None = Field(default=None, exclude_if=lambda value: value is None)
    note: str | None = Field(
        default=None,
        max_length=80_000,
        description=_RESPONSE_NOTE_DESCRIPTION,
        exclude_if=lambda value: value is None,
    )
    citations: tuple[AnswerCitation | None, ...] | None = Field(
        default=None,
        description=_POSITIONAL_CITATIONS_DESCRIPTION,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_text_answer(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = cast(dict[str, object], value)
        if payload.get("text") is None or payload.get("output", ...) is not None:
            return value
        normalized = dict(payload)
        normalized.pop("output", None)
        return normalized

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("response text must not be blank")
        return stripped

    @field_validator("output")
    @classmethod
    def _validate_output(cls, value: JsonValue | None) -> JsonValue | None:
        if value is None:
            return None
        return validate_output_size(value)

    @field_validator("note")
    @classmethod
    def _validate_note(cls, value: str | None) -> str | None:
        return _normalize_response_note(value)

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

    @model_validator(mode="after")
    def _validate_answer_mode(self) -> Response:
        answer_fields = self.model_fields_set & {"text", "output"}
        if len(answer_fields) != 1:
            raise ValueError("response must include exactly one answer field")
        if "text" in answer_fields and self.text is None:
            raise ValueError("response text must not be null")
        return self

    @model_serializer(mode="wrap")
    def _serialize_answer_presence(  # noqa: ANN202 - return annotation changes Pydantic's published schema
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ):
        # A return annotation makes Pydantic publish distinct input/output schemas even
        # though json_schema_mode_override intentionally keeps this contract canonical.
        payload = cast(dict[str, object], handler(self))
        output_is_selected = (
            (info.include is None or "output" in info.include)
            and (info.exclude is None or "output" not in info.exclude)
        )
        if "output" in self.model_fields_set and output_is_selected:
            payload["output"] = self.output
        return payload

    @property
    def answer_text(self) -> str:
        if "text" in self.model_fields_set:
            assert self.text is not None
            return self.text
        return compact_json(self.output)


def _normalize_response_note(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError("response note must not be blank")
    return stripped


class ScorerReasoning(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    text: str | None = Field(default=None, min_length=1)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class FastScoreExpectedComponent(BaseModel):
    """One required answer component retained as fast-score evidence."""

    model_config = COMMONS_STRICT_CONFIG

    component_id: str = Field(min_length=1)
    is_correct: bool


class FastScoreExcessiveComponent(BaseModel):
    """One excessive answer component retained as fast-score evidence."""

    model_config = COMMONS_STRICT_CONFIG

    component_id: str = Field(min_length=1)


class FastScoreEvidence(BaseModel):
    """Persisted component judgment and deterministic metrics for one fast score."""

    model_config = COMMONS_STRICT_CONFIG

    expected_components: tuple[FastScoreExpectedComponent, ...] = Field(min_length=1)
    excessive_components: tuple[FastScoreExcessiveComponent, ...]
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)

    @field_validator("expected_components", "excessive_components", mode="before")
    @classmethod
    def _normalize_component_arrays(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_evidence(self) -> FastScoreEvidence:
        expected_ids = _normalized_fast_score_component_ids(
            component.component_id for component in self.expected_components
        )
        excessive_ids = _normalized_fast_score_component_ids(
            component.component_id for component in self.excessive_components
        )
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("fast score expected component identities must be unique")
        if len(excessive_ids) != len(set(excessive_ids)):
            raise ValueError("fast score excessive component identities must be unique")
        if set(expected_ids) & set(excessive_ids):
            raise ValueError("fast score expected and excessive component identities must be disjoint")

        correct = sum(component.is_correct for component in self.expected_components)
        expected_precision = 0.0 if correct == 0 else correct / (correct + len(self.excessive_components))
        expected_recall = correct / len(self.expected_components)
        if not math.isclose(self.precision, expected_precision, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("fast score evidence precision must match component outcomes")
        if not math.isclose(self.recall, expected_recall, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("fast score evidence recall must match component outcomes")
        return self

    @property
    def computed_f1(self) -> float:
        if self.precision == 0.0 or self.recall == 0.0:
            return 0.0
        return round(2 * self.precision * self.recall / (self.precision + self.recall), 6)


def _normalized_fast_score_component_ids(component_ids: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(component_id.strip().casefold() for component_id in component_ids)
    if any(not component_id for component_id in normalized):
        raise ValueError("fast score component identities must not be blank")
    return normalized


class ScoreBreakdown(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    comparison_score: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)
    scoring_version: str = Field(min_length=1)
    reasoning: ScorerReasoning | None = None
    fast_score_evidence: FastScoreEvidence | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

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
        if (
            self.fast_score_evidence is not None
            and self.total_score != self.fast_score_evidence.computed_f1
        ):
            raise ValueError("score breakdown total_score must match fast score evidence F1")
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
    PLATFORM_TOOL_PROXY_DENIED = "platform_tool_proxy_denied"
    PLATFORM_TOOL_PROXY_GRANT_FAILED = "platform_tool_proxy_grant_failed"
    PROGRESS_SNAPSHOT_FAILED = "progress_snapshot_failed"
    # Historical delivery failure code. Active validator runtime no longer emits it.
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
        MinerTaskErrorCode.PLATFORM_TOOL_PROXY_DENIED,
        MinerTaskErrorCode.PLATFORM_TOOL_PROXY_GRANT_FAILED,
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
    "FastScoreEvidence",
    "FastScoreExcessiveComponent",
    "FastScoreExpectedComponent",
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
