"""DTOs for validator miner-task query/run workflows."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harnyx_commons.domain.miner_task import MinerTask, Query, Response
from harnyx_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.domain.shared_config import VALIDATOR_STRICT_CONFIG


class TokenUsageSummary(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Aggregated LLM usage totals grouped by provider and model."""

    by_provider: dict[str, dict[str, LlmUsageTotals]] = Field(default_factory=dict)
    total_prompt_tokens: int = Field(default=0, ge=0)
    total_completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    call_count: int = Field(default=0, ge=0)

    @classmethod
    def empty(cls) -> TokenUsageSummary:
        return cls()

    @classmethod
    def from_totals(
        cls,
        totals: dict[str, dict[str, LlmUsageTotals]],
    ) -> TokenUsageSummary:
        providers, prompt, completion, total, calls = _aggregate_usage_totals(totals)
        return cls(
            by_provider=providers,
            total_prompt_tokens=prompt,
            total_completion_tokens=completion,
            total_tokens=total,
            call_count=calls,
        )

    @classmethod
    def from_usage(cls, usage: SessionUsage) -> TokenUsageSummary:
        if not usage.llm_usage_totals:
            return cls.empty()
        return cls.from_totals(usage.require_usage_totals())


def _aggregate_usage_totals(
    totals: dict[str, dict[str, LlmUsageTotals]],
) -> tuple[dict[str, dict[str, LlmUsageTotals]], int, int, int, int]:
    prompt = 0
    completion = 0
    total = 0
    calls = 0
    providers: dict[str, dict[str, LlmUsageTotals]] = {}

    for provider, models in totals.items():
        provider_models: dict[str, LlmUsageTotals] = {}
        for model, usage in models.items():
            provider_models[model] = usage
            prompt += usage.prompt_tokens
            completion += usage.completion_tokens
            total += usage.total_tokens
            calls += usage.call_count
        if provider_models:
            providers[provider] = provider_models

    return providers, prompt, completion, total, calls


class ScriptArtifactSpec(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Script artifact metadata supplied by the platform."""

    uid: int = Field(ge=0)
    artifact_id: UUID
    content_hash: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    miner_hotkey_ss58: str | None = None
    task_retry_count: int = Field(default=0, ge=0, le=3)

    def require_platform_tool_proxy_scope(self) -> None:
        if self.miner_hotkey_ss58 is None:
            raise ValueError("script artifact is missing miner hotkey for platform tool proxy")


class MinerTaskBatchSpec(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Miner-task batch supplied by the platform."""

    batch_id: UUID
    cutoff_at: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    tasks: tuple[MinerTask, ...] = Field(min_length=1)
    artifacts: tuple[ScriptArtifactSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_membership(self) -> Self:
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("batch tasks must be unique by task_id")

        artifact_ids = tuple(artifact.artifact_id for artifact in self.artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("batch artifacts must be unique by artifact_id")
        return self


class MinerTaskWorkAssignment(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """One platform-assigned miner-task attempt."""

    batch_id: UUID
    artifact: ScriptArtifactSpec
    task: MinerTask
    attempt_number: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    assignment_token: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_attempt_budget(self) -> MinerTaskWorkAssignment:
        if self.max_attempts < self.attempt_number:
            raise ValueError("max_attempts must be >= attempt_number")
        return self


class EntrypointInvocationRequest(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Input payload for invoking a miner query entrypoint."""

    session_id: UUID
    token: str = Field(min_length=1)
    uid: int = Field(ge=0)
    query: Query


class EntrypointInvocationResult(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Response returned by the sandbox query entrypoint."""

    response: Response
    tool_receipts: tuple[ToolCall, ...] = ()


class MinerTaskRunRequest(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Input payload for orchestrating a full miner task run."""

    batch_id: UUID
    session_id: UUID
    token: str = Field(min_length=1)
    uid: int = Field(ge=0)
    artifact_id: UUID
    task: MinerTask


class TaskRunOutcome(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Aggregate outcome of running a miner task query."""

    run: MinerTaskRun
    tool_receipts: tuple[ToolCall, ...] = ()
    usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary.empty)


class MinerTaskRunSubmission(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Payload persisted when a miner task run is recorded."""

    batch_id: UUID
    run: MinerTaskRun
    score: float = Field(ge=0.0, le=1.0)
    execution_log: tuple[ToolCall, ...] = ()
    usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary.empty)
    session: Session

    @model_validator(mode="after")
    def _validate_submission(self) -> MinerTaskRunSubmission:
        breakdown = self.run.details.score_breakdown
        error = self.run.details.error
        if error is None:
            if breakdown is None:
                raise ValueError("successful task runs must include score breakdown details")
            if self.run.response is None:
                raise ValueError("successful task runs must include a response")
            if breakdown.total_score != self.score:
                raise ValueError("score must match details.score_breakdown.total_score")
            return self

        if self.score != 0.0:
            raise ValueError("failed task runs must report score=0")
        return self


class MinerTaskAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MinerTaskAttemptRetryDecision(StrEnum):
    WILL_RETRY = "will_retry"
    WILL_NOT_RETRY = "will_not_retry"


class MinerTaskAttemptTerminalEffect(StrEnum):
    TASK_RESULT = "task_result"
    DELIVERY_FAILURE = "delivery_failure"
    ATTEMPT_FAILURE = "attempt_failure"


DIAGNOSTIC_ID_MAX_LENGTH = 512
DIAGNOSTIC_STATE_ERROR_MAX_LENGTH = 2048
DIAGNOSTIC_TEXT_MAX_LENGTH = 4096
DIAGNOSTIC_LOG_TAIL_MAX_LENGTH = 8192
DIAGNOSTIC_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    str_strip_whitespace=False,
)


class SandboxFailureDiagnostics(BaseModel):
    model_config = DIAGNOSTIC_STRICT_CONFIG

    image: str | None = Field(default=None, max_length=DIAGNOSTIC_ID_MAX_LENGTH)
    pull_policy: str | None = Field(default=None, max_length=DIAGNOSTIC_ID_MAX_LENGTH)
    container_name: str | None = Field(default=None, max_length=DIAGNOSTIC_ID_MAX_LENGTH)
    container_id: str | None = Field(default=None, max_length=DIAGNOSTIC_ID_MAX_LENGTH)
    status: str | None = Field(default=None, max_length=DIAGNOSTIC_ID_MAX_LENGTH)
    exit_code: int | None = None
    oom_killed: bool | None = None
    state_error: str | None = Field(default=None, max_length=DIAGNOSTIC_STATE_ERROR_MAX_LENGTH)
    error_text: str | None = Field(default=None, max_length=DIAGNOSTIC_TEXT_MAX_LENGTH)
    docker_logs_tail: str | None = Field(default=None, max_length=DIAGNOSTIC_LOG_TAIL_MAX_LENGTH)
    pull_returncode: int | None = None
    pull_stdout_tail: str | None = Field(default=None, max_length=DIAGNOSTIC_TEXT_MAX_LENGTH)
    pull_stderr_tail: str | None = Field(default=None, max_length=DIAGNOSTIC_TEXT_MAX_LENGTH)
    run_returncode: int | None = None
    run_stdout_tail: str | None = Field(default=None, max_length=DIAGNOSTIC_TEXT_MAX_LENGTH)
    run_stderr_tail: str | None = Field(default=None, max_length=DIAGNOSTIC_TEXT_MAX_LENGTH)


class ValidatorBatchFailureDetail(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    error_code: str
    error_message: str
    occurred_at: datetime
    artifact_id: UUID | None = None
    task_id: UUID | None = None
    uid: int | None = None
    exception_type: str | None = None
    traceback: str | None = None
    sandbox_diagnostics: SandboxFailureDiagnostics | None = None


class MinerTaskAttemptDiagnostics(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    phase: str = Field(min_length=1, max_length=128)
    timeout_owner: str | None = Field(default=None, max_length=128)
    failure_owner: str | None = Field(default=None, max_length=128)
    elapsed_ms: float | None = Field(default=None, ge=0.0)
    platform_tool_activity_observed: bool = False


class MinerTaskAttemptAuditRecord(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    validator_session_id: UUID
    batch_id: UUID
    artifact_id: UUID
    task_id: UUID
    attempt_number: int = Field(ge=1)
    uid: int = Field(ge=0)
    miner_hotkey_ss58: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    status: MinerTaskAttemptStatus
    error_code: str | None = Field(default=None, max_length=128)
    error_summary_code: str | None = Field(default=None, max_length=128)
    retry_decision: MinerTaskAttemptRetryDecision
    terminal_effect: MinerTaskAttemptTerminalEffect | None
    max_attempts: int = Field(ge=1)
    execution_log: tuple[ToolCall, ...] = ()
    diagnostics: MinerTaskAttemptDiagnostics | None = None
    delivery_failure_detail: ValidatorBatchFailureDetail | None = None

    @model_validator(mode="after")
    def _validate_attempt(self) -> MinerTaskAttemptAuditRecord:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be >= started_at")
        if self.max_attempts < self.attempt_number:
            raise ValueError("max_attempts must be >= attempt_number")
        if self.status is MinerTaskAttemptStatus.SUCCEEDED:
            if self.error_code is not None or self.error_summary_code is not None:
                raise ValueError("succeeded attempts must not include error fields")
            if self.retry_decision is not MinerTaskAttemptRetryDecision.WILL_NOT_RETRY:
                raise ValueError("succeeded attempts must not retry")
            if self.terminal_effect is not MinerTaskAttemptTerminalEffect.TASK_RESULT:
                raise ValueError("succeeded attempts must have task_result terminal effect")
        if self.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY:
            if self.terminal_effect is not None:
                raise ValueError("retrying attempts must not have terminal effect")
            if self.attempt_number >= self.max_attempts:
                raise ValueError("retrying attempts must have remaining retry budget")
        if (
            self.terminal_effect is MinerTaskAttemptTerminalEffect.ATTEMPT_FAILURE
            and self.attempt_number != self.max_attempts
        ):
            raise ValueError("attempt_failure requires final attempt")
        if self.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY and self.terminal_effect is None:
            raise ValueError("non-retrying attempts must have terminal effect")
        if (
            self.delivery_failure_detail is not None
            and self.terminal_effect is not MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
        ):
            raise ValueError("delivery_failure_detail requires delivery_failure terminal effect")
        return self


class PlatformOwnedTaskResult(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Completed payload for one platform-assigned miner-task attempt."""

    batch_id: UUID
    artifact_id: UUID
    task_id: UUID
    attempt_number: int = Field(ge=1)
    result: MinerTaskRunSubmission | None = None
    terminal_attempt: MinerTaskAttemptAuditRecord


__all__ = [
    "DIAGNOSTIC_ID_MAX_LENGTH",
    "DIAGNOSTIC_LOG_TAIL_MAX_LENGTH",
    "DIAGNOSTIC_STATE_ERROR_MAX_LENGTH",
    "DIAGNOSTIC_TEXT_MAX_LENGTH",
    "EntrypointInvocationRequest",
    "EntrypointInvocationResult",
    "MinerTaskAttemptAuditRecord",
    "MinerTaskAttemptDiagnostics",
    "MinerTaskAttemptRetryDecision",
    "MinerTaskAttemptStatus",
    "MinerTaskAttemptTerminalEffect",
    "MinerTaskBatchSpec",
    "MinerTaskWorkAssignment",
    "MinerTaskRunRequest",
    "MinerTaskRunSubmission",
    "PlatformOwnedTaskResult",
    "SandboxFailureDiagnostics",
    "ScriptArtifactSpec",
    "TaskRunOutcome",
    "TokenUsageSummary",
    "ValidatorBatchFailureDetail",
]
