"""Dataclass schemas for the validator HTTP API."""

from __future__ import annotations

from dataclasses import dataclass

from caster_commons.domain.tool_usage import ToolUsageSummary
from caster_commons.tools.http_models import ToolExecuteResponseDTO, ToolResultDTO


@dataclass(frozen=True, slots=True)
class BatchAcceptResponse:
    status: str
    batch_id: str
    caller: str


@dataclass(frozen=True, slots=True)
class MinerTaskResultCitationModel:
    url: str | None = None
    note: str | None = None
    receipt_id: str | None = None
    result_id: str | None = None


@dataclass(frozen=True, slots=True)
class MinerTaskResultCriterionEvaluationModel:
    criterion_evaluation_id: str
    uid: int
    artifact_id: str
    claim_id: str
    verdict: int
    justification: str
    citations: list[MinerTaskResultCitationModel]


@dataclass(frozen=True, slots=True)
class MinerTaskResultScoreModel:
    verdict_score: float
    support_score: float
    justification_pass: bool
    failed_citation_ids: list[str]
    grader_rationale: str | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class UsageModelEntry:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    call_count: int


@dataclass(frozen=True, slots=True)
class UsageModel:
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    call_count: int
    by_provider: dict[str, dict[str, UsageModelEntry]]


@dataclass(frozen=True, slots=True)
class SessionModel:
    session_id: str
    uid: int
    status: str
    issued_at: str
    expires_at: str


@dataclass(frozen=True, slots=True)
class ValidatorModel:
    uid: int


@dataclass(frozen=True, slots=True)
class MinerTaskResultModel:
    batch_id: str
    validator: ValidatorModel
    criterion_evaluation: MinerTaskResultCriterionEvaluationModel
    score: MinerTaskResultScoreModel
    usage: UsageModel
    session: SessionModel
    total_tool_usage: ToolUsageSummary


@dataclass(frozen=True, slots=True)
class ProgressResponse:
    batch_id: str
    total: int
    completed: int
    remaining: int
    miner_task_results: list[MinerTaskResultModel]


@dataclass(frozen=True, slots=True)
class ValidatorStatusResponse:
    status: str
    last_batch_id: str | None = None
    last_started_at: str | None = None
    last_completed_at: str | None = None
    running: bool = False
    queued_batches: int = 0
    last_error: str | None = None
    last_weight_submission_at: str | None = None
    last_weight_error: str | None = None


__all__ = [
    "ToolResultDTO",
    "ToolExecuteResponseDTO",
    "BatchAcceptResponse",
    "MinerTaskResultCitationModel",
    "MinerTaskResultCriterionEvaluationModel",
    "MinerTaskResultScoreModel",
    "UsageModelEntry",
    "UsageModel",
    "SessionModel",
    "ValidatorModel",
    "MinerTaskResultModel",
    "ProgressResponse",
    "ValidatorStatusResponse",
]
