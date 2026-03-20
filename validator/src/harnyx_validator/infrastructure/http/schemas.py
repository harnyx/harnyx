"""Pydantic schemas for the validator HTTP API."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from harnyx_commons.domain.miner_task import (
    DEFAULT_MINER_TASK_BUDGET_USD,
    EvaluationDetails,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
)
from harnyx_commons.tools.http_models import ToolExecuteResponseDTO, ToolResultDTO
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec
from harnyx_validator.domain.shared_config import VALIDATOR_STRICT_CONFIG


class BatchAcceptResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    status: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    caller: str = Field(min_length=1)


class UsageModelEntry(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    call_count: int = Field(ge=0)


class UsageModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    call_count: int = Field(ge=0)
    by_provider: dict[str, dict[str, UsageModelEntry]] = Field(default_factory=dict)


class SessionModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    session_id: str = Field(min_length=1)
    uid: int = Field(ge=0)
    status: str = Field(min_length=1)
    issued_at: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)


class ValidatorModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    uid: int = Field(ge=0)


def _validate_uuid_string(value: str) -> str:
    UUID(value)
    return value


class ScriptArtifactRequestModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    uid: int = Field(ge=0)
    artifact_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, value: str) -> str:
        return _validate_uuid_string(value)

    def to_domain(self) -> ScriptArtifactSpec:
        return ScriptArtifactSpec(
            uid=self.uid,
            artifact_id=UUID(self.artifact_id),
            content_hash=self.content_hash,
            size_bytes=self.size_bytes,
        )


class MinerTaskRequestModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    task_id: str = Field(min_length=1)
    query: Query
    reference_answer: ReferenceAnswer
    budget_usd: float = Field(default=DEFAULT_MINER_TASK_BUDGET_USD, ge=0.0)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return _validate_uuid_string(value)

    def to_domain_task(self) -> MinerTask:
        return MinerTask(
            task_id=UUID(self.task_id),
            query=self.query,
            reference_answer=self.reference_answer,
            budget_usd=self.budget_usd,
        )


class MinerTaskBatchRequestModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    batch_id: str = Field(min_length=1)
    cutoff_at: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    tasks: list[MinerTaskRequestModel] = Field(min_length=1)
    artifacts: list[ScriptArtifactRequestModel] = Field(min_length=1)

    @field_validator("batch_id")
    @classmethod
    def _validate_batch_id(cls, value: str) -> str:
        return _validate_uuid_string(value)

    def to_domain(self) -> MinerTaskBatchSpec:
        return MinerTaskBatchSpec(
            batch_id=UUID(self.batch_id),
            cutoff_at=self.cutoff_at,
            created_at=self.created_at,
            tasks=tuple(task.to_domain_task() for task in self.tasks),
            artifacts=tuple(artifact.to_domain() for artifact in self.artifacts),
        )


class MinerTaskRunModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    uid: int = Field(ge=0)
    artifact_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    query: Query
    reference_answer: ReferenceAnswer
    response: Response | None = None


class MinerTaskRunSubmissionModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    batch_id: str = Field(min_length=1)
    validator: ValidatorModel
    run: MinerTaskRunModel
    score: float = Field(ge=0.0, le=1.0)
    usage: UsageModel
    session: SessionModel
    specifics: EvaluationDetails


class ProgressResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    batch_id: str = Field(min_length=1)
    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    remaining: int = Field(ge=0)
    miner_task_runs: list[MinerTaskRunSubmissionModel]


class ValidatorStatusResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    status: str = Field(min_length=1)
    last_batch_id: str | None = None
    last_started_at: str | None = None
    last_completed_at: str | None = None
    running: bool = False
    queued_batches: int = Field(default=0, ge=0)
    last_error: str | None = None
    last_weight_submission_at: str | None = None
    last_weight_error: str | None = None


__all__ = [
    "BatchAcceptResponse",
    "MinerTaskBatchRequestModel",
    "MinerTaskRequestModel",
    "MinerTaskRunModel",
    "MinerTaskRunSubmissionModel",
    "ProgressResponse",
    "SessionModel",
    "ScriptArtifactRequestModel",
    "ToolExecuteResponseDTO",
    "ToolResultDTO",
    "UsageModel",
    "UsageModelEntry",
    "ValidatorModel",
    "ValidatorStatusResponse",
]
