"""Pydantic schemas for the validator HTTP API."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from harnyx_commons.domain.miner_task import (
    DEFAULT_MINER_TASK_BUDGET_USD,
    EvaluationDetails,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
)
from harnyx_commons.domain.session import LlmUsageTotals, Session, SessionStatus
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_commons.tools.http_models import ToolExecuteResponseDTO, ToolResultDTO
from harnyx_commons.tools.types import TOOL_NAMES
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.application.ports.progress import ProviderFailureEvidence
from harnyx_validator.application.services.evaluation_runner import ValidatorBatchFailureDetail
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.domain.shared_config import VALIDATOR_STRICT_CONFIG

_VALIDATOR_TRANSPORT_CONFIG = ConfigDict(
    extra="ignore",
    frozen=True,
    strict=True,
    str_strip_whitespace=True,
)
_TOOL_CALLS_ADAPTER = TypeAdapter(tuple[ToolCall, ...])


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

    def to_domain(self) -> LlmUsageTotals:
        return LlmUsageTotals(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            call_count=self.call_count,
        )


class UsageModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    call_count: int = Field(ge=0)
    by_provider: dict[str, dict[str, UsageModelEntry]] = Field(default_factory=dict)

    def to_domain(self) -> TokenUsageSummary:
        return TokenUsageSummary.from_totals(
            {
                provider: {
                    model: entry.to_domain()
                    for model, entry in models.items()
                }
                for provider, models in self.by_provider.items()
            }
        )


class SessionModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    session_id: str = Field(min_length=1)
    uid: int = Field(ge=0)
    status: str = Field(min_length=1)
    issued_at: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)

    def to_domain(self, *, task: MinerTask, uid: int) -> Session:
        return Session(
            session_id=UUID(self.session_id),
            uid=uid,
            task_id=task.task_id,
            issued_at=datetime.fromisoformat(self.issued_at),
            expires_at=datetime.fromisoformat(self.expires_at),
            budget_usd=task.budget_usd,
            status=SessionStatus(self.status),
        )


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
    model_config = _VALIDATOR_TRANSPORT_CONFIG

    batch_id: str = Field(min_length=1)
    cutoff_at: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    tasks: list[MinerTaskRequestModel] = Field(min_length=1)
    artifacts: list[ScriptArtifactRequestModel] = Field(min_length=1)
    restore_runs: list[RestoreMinerTaskRunSubmissionModel] = Field(default_factory=list)
    restore_provider_evidence: list[ProviderEvidenceModel] = Field(default_factory=list)

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

    def to_domain_restore_runs(self) -> tuple[MinerTaskRunSubmission, ...]:
        batch = self.to_domain()
        return tuple(entry.to_domain(batch=batch) for entry in self.restore_runs)

    def to_domain_restore_provider_evidence(self) -> tuple[ProviderFailureEvidence, ...]:
        return tuple(entry.to_domain() for entry in self.restore_provider_evidence)


class MinerTaskRunModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    uid: int = Field(ge=0)
    artifact_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    query: Query
    reference_answer: ReferenceAnswer
    completed_at: str | None = None
    response: Response | None = None


class RestoreMinerTaskRunModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    artifact_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    completed_at: str | None = None
    response: Response | None = None

    @field_validator("artifact_id", "task_id")
    @classmethod
    def _validate_run_uuid(cls, value: str) -> str:
        return _validate_uuid_string(value)


class RestoreMinerTaskRunSubmissionModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    batch_id: str = Field(min_length=1)
    validator: ValidatorModel
    run: RestoreMinerTaskRunModel
    score: float = Field(ge=0.0, le=1.0)
    execution_log: tuple[ToolCall, ...] = ()
    usage: UsageModel
    session: SessionModel
    specifics: EvaluationDetails

    @field_validator("batch_id")
    @classmethod
    def _validate_batch_id(cls, value: str) -> str:
        return _validate_uuid_string(value)

    @field_validator("execution_log", mode="before")
    @classmethod
    def _validate_execution_log(cls, value: object) -> tuple[ToolCall, ...] | object:
        if isinstance(value, tuple) and all(isinstance(entry, ToolCall) for entry in value):
            return value
        return _TOOL_CALLS_ADAPTER.validate_python(_filter_unknown_execution_log_entries(value))

    def to_domain(self, *, batch: MinerTaskBatchSpec) -> MinerTaskRunSubmission:
        batch_id = UUID(self.batch_id)
        if batch_id != batch.batch_id:
            raise RuntimeError("restore run batch_id mismatch")
        artifact_id = UUID(self.run.artifact_id)
        task_id = UUID(self.run.task_id)
        tasks_by_id = {task.task_id: task for task in batch.tasks}
        artifacts_by_id = {artifact.artifact_id: artifact for artifact in batch.artifacts}
        task = tasks_by_id.get(task_id)
        if task is None:
            raise RuntimeError(f"restore run task missing from batch: {task_id}")
        artifact = artifacts_by_id.get(artifact_id)
        if artifact is None:
            raise RuntimeError(f"restore run artifact missing from batch: {artifact_id}")
        session = self.session.to_domain(task=task, uid=artifact.uid)
        return MinerTaskRunSubmission(
            batch_id=batch_id,
            validator_uid=self.validator.uid,
            run=MinerTaskRun(
                session_id=session.session_id,
                uid=artifact.uid,
                artifact_id=artifact_id,
                task_id=task_id,
                response=self.run.response,
                details=self.specifics,
                completed_at=(
                    datetime.fromisoformat(self.run.completed_at)
                    if self.run.completed_at is not None
                    else session.issued_at
                ),
            ),
            score=self.score,
            execution_log=self.execution_log,
            usage=self.usage.to_domain(),
            session=session,
        )


class MinerTaskRunSubmissionModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    batch_id: str = Field(min_length=1)
    validator: ValidatorModel
    run: MinerTaskRunModel
    score: float = Field(ge=0.0, le=1.0)
    execution_log: tuple[ToolCall, ...] = ()
    usage: UsageModel
    session: SessionModel
    specifics: EvaluationDetails


def _filter_unknown_execution_log_entries(value: object) -> object:
    if not isinstance(value, list | tuple):
        return value
    filtered: list[object] = []
    for entry in value:
        if _should_skip_execution_log_entry(entry):
            continue
        filtered.append(entry)
    return filtered


def _should_skip_execution_log_entry(entry: object) -> bool:
    if isinstance(entry, ToolCall):
        return False
    if not isinstance(entry, Mapping):
        return False
    mapping = cast(Mapping[str, object], entry)
    tool = mapping.get("tool")
    return isinstance(tool, str) and tool not in TOOL_NAMES


def _serialize_failure_detail_message(detail: ValidatorBatchFailureDetail) -> str:
    error_message = detail.error_message.strip()
    if error_message:
        return error_message
    if detail.exception_type is not None:
        exception_type = detail.exception_type.strip()
        if exception_type:
            return exception_type
    return detail.error_code


class FailureDetailResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    error_code: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    artifact_id: str | None = None
    task_id: str | None = None
    uid: int | None = Field(default=None, ge=0)
    exception_type: str | None = None
    traceback: str | None = None
    occurred_at: str = Field(min_length=1)

    @classmethod
    def from_domain(cls, detail: ValidatorBatchFailureDetail) -> FailureDetailResponse:
        return cls(
            error_code=detail.error_code,
            error_message=_serialize_failure_detail_message(detail),
            artifact_id=None if detail.artifact_id is None else str(detail.artifact_id),
            task_id=None if detail.task_id is None else str(detail.task_id),
            uid=detail.uid,
            exception_type=detail.exception_type,
            traceback=detail.traceback,
            occurred_at=detail.occurred_at.isoformat(),
        )


class ValidatorInternalErrorResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    error_code: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    exception_type: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    traceback: str | None = None


class ProviderEvidenceModel(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    total_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)

    def to_domain(self) -> ProviderFailureEvidence:
        return {
            "provider": self.provider,
            "model": self.model,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
        }


class ProgressResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    batch_id: str = Field(min_length=1)
    status: Literal["unknown", "queued", "processing", "completed", "failed"]
    error_code: str | None = None
    failure_detail: FailureDetailResponse | None = None
    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    remaining: int = Field(ge=0)
    miner_task_runs: list[MinerTaskRunSubmissionModel]
    provider_model_evidence: list[ProviderEvidenceModel] = Field(default_factory=list)


class ValidatorResourceUsageResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    captured_at: str = Field(min_length=1)
    cpu_percent: float = Field(ge=0.0)
    memory_used_bytes: int = Field(ge=0)
    memory_total_bytes: int = Field(ge=0)
    memory_percent: float = Field(ge=0.0)
    disk_used_bytes: int = Field(ge=0)
    disk_total_bytes: int = Field(ge=0)
    disk_percent: float = Field(ge=0.0)


class ValidatorStatusResponse(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

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
    resource_usage: ValidatorResourceUsageResponse | None = None
    signature_hex: str | None = None


__all__ = [
    "BatchAcceptResponse",
    "FailureDetailResponse",
    "MinerTaskBatchRequestModel",
    "MinerTaskRequestModel",
    "MinerTaskRunModel",
    "MinerTaskRunSubmissionModel",
    "ProviderEvidenceModel",
    "ProgressResponse",
    "SessionModel",
    "ScriptArtifactRequestModel",
    "ToolExecuteResponseDTO",
    "ToolResultDTO",
    "UsageModel",
    "UsageModelEntry",
    "ValidatorModel",
    "ValidatorResourceUsageResponse",
    "ValidatorInternalErrorResponse",
    "ValidatorStatusResponse",
]
