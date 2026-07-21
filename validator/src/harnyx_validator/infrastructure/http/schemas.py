"""Pydantic schemas for the validator HTTP API."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest, SimilarityJudgeResult
from harnyx_validator.domain.shared_config import VALIDATOR_STRICT_CONFIG

_VALIDATOR_TRANSPORT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    str_strip_whitespace=True,
)


def _validate_uuid_string(value: str) -> str:
    UUID(value)
    return value


class SimilarityJudgeRequestModel(BaseModel):
    model_config = _VALIDATOR_TRANSPORT_CONFIG

    candidate_artifact_id: str = Field(min_length=1)
    incumbent_artifact_id: str = Field(min_length=1)
    candidate_miner_uid: int = Field(ge=0)
    incumbent_miner_uid: int = Field(ge=0)
    incumbent_script: str = Field(min_length=1)
    candidate_diff: str = Field(min_length=1)

    @field_validator("candidate_artifact_id", "incumbent_artifact_id")
    @classmethod
    def _validate_artifact_id(cls, value: str) -> str:
        return _validate_uuid_string(value)

    def to_domain(self, *, batch_id: UUID) -> SimilarityJudgeRequest:
        return SimilarityJudgeRequest(
            batch_id=batch_id,
            candidate_artifact_id=UUID(self.candidate_artifact_id),
            reference_artifact_id=UUID(self.incumbent_artifact_id),
            candidate_miner_uid=self.candidate_miner_uid,
            reference_miner_uid=self.incumbent_miner_uid,
            reference_script=self.incumbent_script,
            candidate_diff=self.candidate_diff,
        )


class SimilarityJudgeResponseModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    classification: Literal["duplicate", "near_duplicate", "novel"]
    reasoning: str | None = None
    reasoning_tokens: int | None = Field(default=None, ge=0)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    judge_usage: JudgeUsageSummary | None = None

    @classmethod
    def from_domain(cls, result: SimilarityJudgeResult) -> SimilarityJudgeResponseModel:
        return cls(
            classification=result.classification,
            reasoning=result.reasoning,
            reasoning_tokens=result.reasoning_tokens,
            model=result.model,
            provider=result.provider,
            judge_usage=result.judge_usage,
        )


class SimilarityJudgeFailureResponseModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    error_code: Literal["similarity_judge_failed"]
    retryable: bool
    detail: str = Field(min_length=1)
    judge_usage: JudgeUsageSummary | None = None


class ValidatorInternalErrorResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    error_code: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    exception_type: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    traceback: str = Field(min_length=1)


class ValidatorResourceUsageResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    captured_at: str = Field(min_length=1)
    cpu_percent: float = Field(ge=0.0)
    cpu_capacity_cores: float = Field(ge=0.0)
    memory_used_bytes: int = Field(ge=0)
    memory_total_bytes: int = Field(ge=0)
    memory_percent: float = Field(ge=0.0)
    disk_used_bytes: int = Field(ge=0)
    disk_total_bytes: int = Field(ge=0)
    disk_percent: float = Field(ge=0.0)


class ValidatorStatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = Field(min_length=1)
    hotkey: str = Field(min_length=1)
    last_batch_id: str | None = None
    last_started_at: str | None = None
    last_completed_at: str | None = None
    running: bool = False
    queued_batches: int = Field(default=0, ge=0)
    last_error: str | None = None
    last_weight_submission_at: str | None = None
    last_weight_error: str | None = None
    is_chutes_configured: bool = False
    is_openrouter_configured: bool = False
    resource_usage: ValidatorResourceUsageResponse | None = None
    signature_hex: str | None = None


__all__ = [
    "SimilarityJudgeFailureResponseModel",
    "SimilarityJudgeRequestModel",
    "SimilarityJudgeResponseModel",
    "ValidatorResourceUsageResponse",
    "ValidatorInternalErrorResponse",
    "ValidatorStatusResponse",
]
