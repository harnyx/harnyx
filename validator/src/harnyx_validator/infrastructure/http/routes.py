"""HTTP route definitions for the validator API."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from harnyx_commons.bittensor import VerificationError
from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.domain.session import Session
from harnyx_commons.errors import ConcurrencyLimitError
from harnyx_commons.protocol_headers import (
    SESSION_ID_HEADER,
)
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.http_models import (
    ToolExecuteRequestDTO,
    ToolExecuteResponseDTO,
)
from harnyx_commons.tools.http_serialization import serialize_tool_execute_response
from harnyx_commons.tools.token_semaphore import TokenSemaphore
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunSubmission,
    TokenUsageSummary,
)
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.infrastructure.http.schemas import (
    BatchAcceptResponse,
    MinerTaskBatchRequestModel,
    MinerTaskRunModel,
    MinerTaskRunSubmissionModel,
    ProgressResponse,
    SessionModel,
    UsageModel,
    UsageModelEntry,
    ValidatorModel,
    ValidatorStatusResponse,
)
from harnyx_validator.infrastructure.state.run_progress import RunProgressSnapshot

logger = logging.getLogger("harnyx_validator.http")


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

    tool_token_header = APIKeyHeader(name="x-platform-token", scheme_name="PlatformToken", auto_error=False)

    @app.post(
        "/v1/tools/execute",
        response_model=ToolExecuteResponseDTO,
        description="Execute a tool invocation and return the tool result and usage.",
    )
    async def execute_tool(
        payload: ToolExecuteRequestDTO,
        deps: ToolRouteDeps = Depends(get_dependencies),  # noqa: B008
        token_header: str | None = Security(tool_token_header),
        session_id: UUID = Header(alias=SESSION_ID_HEADER),  # noqa: B008
    ) -> ToolExecuteResponseDTO:
        if not token_header:
            raise HTTPException(status_code=401, detail="missing x-platform-token header")
        invocation = ToolInvocationRequest(
            session_id=session_id,
            token=token_header,
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
            _log_tool_error(session_id, invocation, exc)
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
        payload: MinerTaskBatchRequestModel,
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        caller: str = Security(require_bittensor_caller),
    ) -> BatchAcceptResponse:
        batch = payload.to_domain()
        try:
            deps.accept_batch.execute(batch)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return BatchAcceptResponse(status="accepted", batch_id=str(batch.batch_id), caller=caller)

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
        tasks_by_id = {task.task_id: task for task in snapshot["tasks"]}
        runs = [_serialize_run(result, tasks_by_id) for result in snapshot["miner_task_runs"]]
        return ProgressResponse(
            batch_id=str(batch_id),
            total=snapshot["total"],
            completed=snapshot["completed"],
            remaining=snapshot["remaining"],
            miner_task_runs=runs,
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


async def _execute_with_semaphore_async(invocation: ToolInvocationRequest, deps: ToolRouteDeps) -> Any:
    token = invocation.token
    semaphore = deps.token_semaphore
    semaphore.acquire(token)
    try:
        return await deps.tool_executor.execute(invocation)
    finally:
        semaphore.release(token)


def _log_tool_error(
    request_session_id: UUID,
    invocation: ToolInvocationRequest,
    exc: Exception,
) -> None:
    logger.exception(
        "tool execution failed (tool=%s session_id=%s request_session_id=%s args=%s kwargs=%s)",
        invocation.tool,
        str(invocation.session_id),
        str(request_session_id),
        tuple(invocation.args),
        dict(invocation.kwargs),
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


def _serialize_run(
    submission: MinerTaskRunSubmission,
    tasks_by_id: dict[UUID, MinerTask],
) -> MinerTaskRunSubmissionModel:
    task = tasks_by_id.get(submission.run.task_id)
    if task is None:
        raise RuntimeError(f"task {submission.run.task_id} missing from progress snapshot")
    return MinerTaskRunSubmissionModel(
        batch_id=str(submission.batch_id),
        validator=ValidatorModel(uid=submission.validator_uid),
        run=MinerTaskRunModel(
            uid=submission.run.uid,
            artifact_id=str(submission.run.artifact_id),
            task_id=str(submission.run.task_id),
            query=task.query,
            reference_answer=task.reference_answer,
            response=submission.run.response,
        ),
        score=submission.score,
        usage=_serialize_usage_block(submission.usage),
        session=_serialize_session_block(submission.session),
        specifics=submission.run.details,
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
