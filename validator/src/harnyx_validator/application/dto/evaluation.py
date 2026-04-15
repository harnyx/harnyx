"""DTOs for validator miner-task query/run workflows."""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

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
    validator_uid: int = Field(ge=0)
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


class MinerTaskBatchRunResult(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    """Outcome of running a miner-task batch across the supplied artifacts."""

    batch_id: UUID
    tasks: tuple[MinerTask, ...]
    runs: tuple[MinerTaskRunSubmission, ...]


__all__ = [
    "EntrypointInvocationRequest",
    "EntrypointInvocationResult",
    "MinerTaskBatchRunResult",
    "MinerTaskBatchSpec",
    "MinerTaskRunRequest",
    "MinerTaskRunSubmission",
    "ScriptArtifactSpec",
    "TaskRunOutcome",
    "TokenUsageSummary",
]
