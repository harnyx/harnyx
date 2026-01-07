"""HTTP route definitions for the validator API."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request

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
    EvaluationBatchSpec,
    EvaluationCloseout,
    TokenUsageSummary,
)
from caster_validator.application.services.evaluation_scoring import EvaluationScore
from caster_validator.application.status import StatusProvider
from caster_validator.domain.evaluation import MinerAnswer, MinerEvaluation
from caster_validator.infrastructure.http.schemas import (
    BatchAcceptResponse,
    CloseoutCitationModel,
    CloseoutEvaluationModel,
    CloseoutModel,
    CloseoutScoreModel,
    CloseoutValidatorModel,
    ProgressResponse,
    SessionModel,
    UsageModel,
    UsageModelEntry,
    ValidatorStatusResponse,
)

logger = logging.getLogger("caster_validator.http")


@dataclass(frozen=True)
class RpcDependencies:
    tool_executor: ToolExecutor
    token_semaphore: TokenSemaphore


@dataclass(frozen=True)
class RpcControlDeps:
    accept_batch: AcceptEvaluationBatch
    status_provider: StatusProvider
    auth: Callable[[Request, bytes], str]
    progress_tracker: Any


def add_tool_routes(app: FastAPI, dependency_provider: Callable[[], RpcDependencies]) -> None:
    def get_dependencies() -> RpcDependencies:
        return dependency_provider()

    @app.post("/rpc/tools/execute", response_model=ToolExecuteResponseDTO)
    async def execute_tool(
        payload: ToolExecuteRequestDTO,
        deps: RpcDependencies = Depends(get_dependencies),  # noqa: B008
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
    control_deps_provider: Callable[[], RpcControlDeps],
) -> None:
    def get_control_deps() -> RpcControlDeps:
        return control_deps_provider()

    @app.post("/rpc/evaluations/batch", response_model=BatchAcceptResponse)
    async def accept_batch(
        payload: EvaluationBatchSpec,
        request: Request,
        deps: RpcControlDeps = Depends(get_control_deps),  # noqa: B008
    ) -> BatchAcceptResponse:
        body = await request.body()
        caller = deps.auth(request, body)
        try:
            deps.accept_batch.execute(payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return BatchAcceptResponse(status="accepted", run_id=str(payload.run_id), caller=caller)

    @app.get("/rpc/evaluations/{run_id}/progress", response_model=ProgressResponse)
    def progress(
        run_id: UUID,
        deps: RpcControlDeps = Depends(get_control_deps),  # noqa: B008
    ) -> ProgressResponse:
        snapshot = deps.progress_tracker.snapshot(run_id)
        closeouts = [_serialize_closeout(c) for c in snapshot.get("closeouts", ())]
        return ProgressResponse(
            run_id=str(run_id),
            total=int(snapshot.get("total", 0)),
            completed=int(snapshot.get("completed", 0)),
            remaining=int(snapshot.get("remaining", 0)),
            closeouts=closeouts,
        )

    @app.get("/rpc/status", response_model=ValidatorStatusResponse)
    def status(
        deps: RpcControlDeps = Depends(get_control_deps),  # noqa: B008
    ) -> ValidatorStatusResponse:
        snapshot = deps.status_provider.snapshot()
        return ValidatorStatusResponse(**snapshot)


# --- Helpers ---


async def _execute_with_semaphore_async(invocation: ToolInvocationRequest, deps: RpcDependencies) -> Any:
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


def _serialize_closeout(closeout: EvaluationCloseout) -> CloseoutModel:
    evaluation = closeout.outcome.evaluation
    answer = evaluation.miner_answer
    return CloseoutModel(
        run_id=str(closeout.run_id),
        validator=CloseoutValidatorModel(uid=closeout.validator_uid),
        evaluation=_serialize_evaluation_block(evaluation, answer),
        score=_serialize_score_block(closeout.outcome.score),
        usage=_serialize_usage_block(closeout.outcome.usage),
        session=_serialize_session_block(closeout.session),
    )


def _serialize_evaluation_block(evaluation: MinerEvaluation, answer: MinerAnswer) -> CloseoutEvaluationModel:
    return CloseoutEvaluationModel(
        evaluation_id=str(evaluation.evaluation_id),
        uid=evaluation.uid,
        claim_id=str(evaluation.claim_id),
        verdict=answer.verdict,
        justification=answer.justification,
        citations=_serialize_citations(answer),
    )


def _serialize_citations(answer: MinerAnswer) -> list[CloseoutCitationModel]:
    return [
        CloseoutCitationModel(
            url=citation.url,
            note=citation.note,
            receipt_id=citation.receipt_id,
            result_id=citation.result_id,
        )
        for citation in answer.citations
    ]


def _serialize_score_block(score: EvaluationScore) -> CloseoutScoreModel:
    return CloseoutScoreModel(
        verdict_score=score.verdict_score,
        support_score=score.support_score,
        justification_pass=score.justification_pass,
        failed_citation_ids=list(score.failed_citation_ids),
        grader_rationale=score.grader_rationale,
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
