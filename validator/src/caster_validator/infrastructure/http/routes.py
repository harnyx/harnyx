"""HTTP route definitions for the validator API."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from caster_commons.bittensor import VerificationError
from caster_commons.domain.session import Session
from caster_commons.errors import ConcurrencyLimitError
from caster_commons.tools.dto import ToolInvocationRequest
from caster_commons.tools.executor import ToolExecutor
from caster_commons.tools.http_models import (
    ToolExecuteRequestDTO,
    ToolExecuteResponseDTO,
)
from caster_commons.tools.http_serialization import serialize_tool_execute_response
from caster_commons.tools.token_semaphore import TokenSemaphore
from caster_validator.application.accept_batch import AcceptEvaluationBatch
from caster_validator.application.dto.evaluation import (
    MinerTaskBatchSpec,
    MinerTaskResult,
    TokenUsageSummary,
)
from caster_validator.application.services.evaluation_scoring import EvaluationScore
from caster_validator.application.status import StatusProvider
from caster_validator.domain.evaluation import MinerAnswer, MinerCriterionEvaluation
from caster_validator.infrastructure.http.schemas import (
    BatchAcceptResponse,
    MinerTaskResultCitationModel,
    MinerTaskResultCriterionEvaluationModel,
    MinerTaskResultModel,
    MinerTaskResultScoreModel,
    ProgressResponse,
    SessionModel,
    UsageModel,
    UsageModelEntry,
    ValidatorModel,
    ValidatorStatusResponse,
)
from caster_validator.infrastructure.state.run_progress import RunProgressSnapshot

logger = logging.getLogger("caster_validator.http")


@dataclass(frozen=True)
class ToolRouteDeps:
    tool_executor: ToolExecutor
    token_semaphore: TokenSemaphore


@dataclass(frozen=True)
class ValidatorControlDeps:
    accept_batch: AcceptEvaluationBatch
    status_provider: StatusProvider
    auth: Callable[[Request, bytes], str]

    progress_tracker: ProgressTracker


class ProgressTracker(Protocol):
    def snapshot(self, batch_id: UUID) -> RunProgressSnapshot:
        ...


def add_tool_routes(app: FastAPI, dependency_provider: Callable[[], ToolRouteDeps]) -> None:
    def get_dependencies() -> ToolRouteDeps:
        return dependency_provider()

    @app.post(
        "/v1/tools/execute",
        response_model=ToolExecuteResponseDTO,
        description="Execute a tool invocation and return the tool result and usage.",
    )
    async def execute_tool(
        payload: ToolExecuteRequestDTO,
        deps: ToolRouteDeps = Depends(get_dependencies),  # noqa: B008
    ) -> ToolExecuteResponseDTO:
        invocation = ToolInvocationRequest(
            session_id=payload.session_id,
            token=payload.token,
            tool=payload.tool,
            args=payload.args,
            kwargs=payload.kwargs,
        )
        try:
            result = await _execute_with_semaphore_async(invocation, deps)
        except (
            ConcurrencyLimitError,
            LookupError,
            PermissionError,
            RuntimeError,
            ValueError,
        ) as exc:
            _log_tool_error(invocation, invocation, exc)
            raise HTTPException(status_code=400, detail=_public_error_message(exc)) from exc
        return serialize_tool_execute_response(result)


def add_control_routes(
    app: FastAPI,
    control_deps_provider: Callable[[], ValidatorControlDeps],
) -> None:
    def get_control_deps() -> ValidatorControlDeps:
        return control_deps_provider()

    bittensor_header = APIKeyHeader(name="Authorization", scheme_name="BittensorAuth", auto_error=False)

    async def require_bittensor_caller(
        request: Request,
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        _auth_header: str | None = Security(bittensor_header),
    ) -> str:
        body = await request.body()
        try:
            return deps.auth(request, body)
        except VerificationError as exc:
            status_code = 403 if exc.code == "caller_not_allowed" else 401
            raise HTTPException(status_code=status_code, detail=exc.message) from exc

    @app.post(
        "/validator/miner-task-batches/batch",
        response_model=BatchAcceptResponse,
        description="Accept a miner task batch and start processing it.",
    )
    async def accept_batch(
        payload: MinerTaskBatchSpec,
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        caller: str = Security(require_bittensor_caller),
    ) -> BatchAcceptResponse:
        try:
            deps.accept_batch.execute(payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return BatchAcceptResponse(status="accepted", batch_id=str(payload.batch_id), caller=caller)

    @app.get(
        "/validator/miner-task-batches/{batch_id}/progress",
        response_model=ProgressResponse,
        description="Return progress and results for a miner task batch.",
    )
    def progress(
        batch_id: UUID,
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        _caller: str = Security(require_bittensor_caller),
    ) -> ProgressResponse:
        snapshot = deps.progress_tracker.snapshot(batch_id)
        results = [_serialize_result(c) for c in snapshot["miner_task_results"]]
        return ProgressResponse(
            batch_id=str(batch_id),
            total=snapshot["total"],
            completed=snapshot["completed"],
            remaining=snapshot["remaining"],
            miner_task_results=results,
        )

    @app.get(
        "/validator/status",
        response_model=ValidatorStatusResponse,
        description="Return a validator status snapshot for platform health checks.",
    )
    def status(
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        _caller: str = Security(require_bittensor_caller),
    ) -> ValidatorStatusResponse:
        snapshot = deps.status_provider.snapshot()
        return ValidatorStatusResponse(**snapshot)


# --- Helpers ---


async def _execute_with_semaphore_async(invocation: ToolInvocationRequest, deps: ToolRouteDeps) -> Any:
    token = invocation.token
    semaphore = deps.token_semaphore
    semaphore.acquire(token)
    try:
        return await deps.tool_executor.execute(invocation)
    finally:
        semaphore.release(token)


def _log_tool_error(
    payload: ToolInvocationRequest, invocation: ToolInvocationRequest, exc: Exception
) -> None:
    logger.exception(
        "tool execution failed (tool=%s session_id=%s request_session_id=%s args=%s kwargs=%s)",
        payload.tool,
        str(payload.session_id),
        str(invocation.session_id) if invocation else None,
        tuple(invocation.args) if invocation else None,
        dict(invocation.kwargs) if invocation else None,
        extra={"error_detail": str(exc)},
    )


def _public_error_message(exc: Exception) -> str:
    if isinstance(exc, PermissionError):
        return "session token rejected"
    if isinstance(exc, LookupError):
        return "session not found"
    if isinstance(exc, ConcurrencyLimitError):
        return "tool concurrency limit reached"
    if isinstance(exc, ValueError):
        return "tool response validation failed"
    return "tool execution failed"


def _serialize_result(result: MinerTaskResult) -> MinerTaskResultModel:
    evaluation = result.outcome.criterion_evaluation
    answer = evaluation.miner_answer
    return MinerTaskResultModel(
        batch_id=str(result.batch_id),
        validator=ValidatorModel(uid=result.validator_uid),
        criterion_evaluation=_serialize_evaluation_block(evaluation, answer),
        score=_serialize_score_block(
            result.outcome.score,
            error_code=result.error_code,
            error_message=result.error_message,
        ),
        usage=_serialize_usage_block(result.outcome.usage),
        session=_serialize_session_block(result.session),
    )


def _serialize_evaluation_block(
    evaluation: MinerCriterionEvaluation, answer: MinerAnswer
) -> MinerTaskResultCriterionEvaluationModel:
    return MinerTaskResultCriterionEvaluationModel(
        criterion_evaluation_id=str(evaluation.criterion_evaluation_id),
        uid=evaluation.uid,
        artifact_id=str(evaluation.artifact_id),
        claim_id=str(evaluation.claim_id),
        verdict=answer.verdict,
        justification=answer.justification,
        citations=_serialize_citations(answer),
    )


def _serialize_citations(answer: MinerAnswer) -> list[MinerTaskResultCitationModel]:
    return [
        MinerTaskResultCitationModel(
            url=citation.url,
            note=citation.note,
            receipt_id=citation.receipt_id,
            result_id=citation.result_id,
        )
        for citation in answer.citations
    ]


def _serialize_score_block(
    score: EvaluationScore,
    *,
    error_code: str | None,
    error_message: str | None,
) -> MinerTaskResultScoreModel:
    return MinerTaskResultScoreModel(
        verdict_score=score.verdict_score,
        support_score=score.support_score,
        justification_pass=score.justification_pass,
        failed_citation_ids=list(score.failed_citation_ids),
        grader_rationale=score.grader_rationale,
        error_code=error_code,
        error_message=error_message,
    )


def _serialize_usage_block(usage: TokenUsageSummary) -> UsageModel:
    return UsageModel(
        total_prompt_tokens=usage.total_prompt_tokens,
        total_completion_tokens=usage.total_completion_tokens,
        total_tokens=usage.total_tokens,
        call_count=usage.call_count,
        by_provider=_serialize_usage_providers(usage),
    )


def _serialize_usage_providers(usage: TokenUsageSummary) -> dict[str, dict[str, UsageModelEntry]]:
    return {
        provider: {
            model: UsageModelEntry(
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                total_tokens=entry.total_tokens,
                call_count=entry.call_count,
            )
            for model, entry in models.items()
        }
        for provider, models in usage.by_provider.items()
    }


def _serialize_session_block(session: Session) -> SessionModel:
    return SessionModel(
        session_id=str(session.session_id),
        uid=session.uid,
        status=session.status.value,
        issued_at=session.issued_at.isoformat(),
        expires_at=session.expires_at.isoformat(),
    )
