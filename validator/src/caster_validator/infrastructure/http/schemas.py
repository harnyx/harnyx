"""Dataclass schemas for the validator HTTP API."""

from __future__ import annotations

from dataclasses import dataclass

from caster_commons.tools.http_models import ToolExecuteResponseDTO, ToolResultDTO


@dataclass(frozen=True, slots=True)
class BatchAcceptResponse:
    status: str
    run_id: str
    caller: str


@dataclass(frozen=True, slots=True)
class CloseoutCitationModel:
    url: str | None = None
    note: str | None = None
    receipt_id: str | None = None
    result_id: str | None = None


@dataclass(frozen=True, slots=True)
class CloseoutEvaluationModel:
    evaluation_id: str
    uid: int
    claim_id: str
    verdict: int
    justification: str
    citations: list[CloseoutCitationModel]


@dataclass(frozen=True, slots=True)
class CloseoutScoreModel:
    verdict_score: float
    support_score: float
    justification_pass: bool
    failed_citation_ids: list[str]
    grader_rationale: str | None = None


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
class CloseoutValidatorModel:
    uid: int


@dataclass(frozen=True, slots=True)
class CloseoutModel:
    run_id: str
    validator: CloseoutValidatorModel
    evaluation: CloseoutEvaluationModel
    score: CloseoutScoreModel
    usage: UsageModel
    session: SessionModel


@dataclass(frozen=True, slots=True)
class ProgressResponse:
    run_id: str
    total: int
    completed: int
    remaining: int
    closeouts: list[CloseoutModel]


@dataclass(frozen=True, slots=True)
class ValidatorStatusResponse:
    status: str
    last_run_id: str | None = None
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
    "CloseoutCitationModel",
    "CloseoutEvaluationModel",
    "CloseoutScoreModel",
    "UsageModelEntry",
    "UsageModel",
    "SessionModel",
    "CloseoutValidatorModel",
    "CloseoutModel",
    "ProgressResponse",
    "ValidatorStatusResponse",
]
