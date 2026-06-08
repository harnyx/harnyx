"""HTTP route definitions for the validator API."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from harnyx_commons.bittensor import VerificationError
from harnyx_commons.domain.miner_task import MinerTaskErrorCode
from harnyx_commons.domain.session import Session
from harnyx_commons.errors import ConcurrencyLimitError, ToolProviderError
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest, SimilarityJudgeResult
from harnyx_commons.protocol_headers import SESSION_ID_HEADER
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, execute_tool_with_concurrency_permit
from harnyx_commons.tools.http_models import ToolExecuteResponseDTO
from harnyx_commons.tools.http_serialization import serialize_tool_execute_response
from harnyx_commons.tools.token_semaphore import ToolConcurrencyLimiter
from harnyx_miner_sdk.tools.http_models import ToolExecuteRequestDTO
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskRunSubmission,
    TokenUsageSummary,
)
from harnyx_validator.application.platform_tool_proxy import PlatformToolProxyScopeRegistry
from harnyx_validator.application.ports.platform import PlatformToolProxyPlatformPort
from harnyx_validator.application.ports.progress import (
    ProviderFailureEvidence,
    RunProgressPage,
    RunProgressSummary,
    SequencedProgressDetail,
)
from harnyx_validator.application.restore_batch import RestoreEvaluationBatch
from harnyx_validator.application.services.evaluation_runner import ValidatorBatchFailureDetail
from harnyx_validator.application.status import BatchActivityTracker, StatusProvider
from harnyx_validator.infrastructure.http.schemas import (
    BatchAcceptResponse,
    BatchProgressRunsPageResponse,
    BatchProgressStatusResponse,
    FailureDetailResponse,
    MinerTaskAttemptAuditModel,
    MinerTaskBatchRequestModel,
    ProviderEvidenceModel,
    RestoreMinerTaskRunModel,
    RestoreMinerTaskRunSubmissionModel,
    SequencedProgressDetailModel,
    SessionModel,
    SimilarityJudgeRequestModel,
    SimilarityJudgeResponseModel,
    UsageModel,
    UsageModelEntry,
    ValidatorInternalErrorResponse,
    ValidatorModel,
    ValidatorResourceUsageResponse,
    ValidatorStatusResponse,
)
from harnyx_validator.infrastructure.observability.sentry import capture_exception
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageSnapshot
from harnyx_validator.runtime.restore_worker import RestoreWorker

logger = logging.getLogger("harnyx_validator.http")
_STATUS_PATH = "/validator/status"
_STATUS_TIMESTAMP_HEADER = "X-Harnyx-Status-Ts"


@dataclass(frozen=True)
class ToolRouteDeps:
    tool_executor: ToolExecutor
    tool_concurrency_limiter: ToolConcurrencyLimiter


ControlRouteAuth = Callable[[str, str, bytes, str | None], Awaitable[str]]


@dataclass(frozen=True)
class ValidatorControlDeps:
    accept_batch: AcceptEvaluationBatch
    restore_batch: RestoreEvaluationBatch
    restore_worker: RestoreWorker
    status_provider: StatusProvider
    auth: ControlRouteAuth
    progress_tracker: ProgressTracker
    validator_hotkey: StatusSigner
    resource_usage_provider: ResourceUsageProvider
    batch_activity: BatchActivityTracker
    platform_tool_proxy_platform: PlatformToolProxyPlatformPort | None = None
    platform_tool_proxy_scopes: PlatformToolProxyScopeRegistry | None = None
    similarity_judge: SimilarityJudgePort | None = None


class ProgressTracker(Protocol):
    def summary(self, batch_id: UUID) -> RunProgressSummary:
        ...

    def completed_run_page(
        self,
        batch_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> RunProgressPage:
        ...


class StatusSigner(Protocol):
    ss58_address: str

    def sign(self, payload: bytes) -> bytes:
        ...


class ResourceUsageProvider(Protocol):
    def snapshot(self) -> ValidatorResourceUsageSnapshot:
        ...


class SimilarityJudgePort(Protocol):
    async def judge(self, request: SimilarityJudgeRequest) -> SimilarityJudgeResult:
        ...


def _path_with_query(request: Request) -> str:
    path = request.url.path or "/"
    query = request.url.query
    if query:
        return f"{path}?{query}"
    return path


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
            result = await execute_tool_with_concurrency_permit(
                deps.tool_executor,
                deps.tool_concurrency_limiter,
                invocation,
            )
        except ToolProviderError as exc:
            _log_tool_error(session_id, invocation, exc)
            raise HTTPException(status_code=400, detail=_public_error_message(exc)) from exc
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


def add_system_routes(app: FastAPI, status_provider: StatusProvider) -> None:
    @app.get(
        "/healthz",
        description="Validator health check.",
    )
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/readyz",
        description="Validator readiness check.",
    )
    def readyz(response: Response) -> dict[str, str]:
        if status_provider.platform_registration_ready() and status_provider.auth_ready():
            return {"status": "ok"}
        if status_provider.platform_registration_ready():
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            error = status_provider.auth_error()
            if error:
                return {"status": "auth_unavailable", "detail": error}
            return {"status": "waiting_for_auth_warmup"}
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        error = status_provider.platform_registration_error()
        if error:
            return {"status": "registration_failed", "detail": error}
        return {"status": "waiting_for_platform_registration"}


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
        path_qs = _path_with_query(request)
        authorization_header = request.headers.get("authorization")
        try:
            caller = await deps.auth(
                request.method,
                path_qs,
                body,
                authorization_header,
            )
        except VerificationError as exc:
            if exc.code == "caller_not_allowed":
                status_code = 403
            elif exc.code == "auth_unavailable":
                status_code = 503
            else:
                status_code = 401
            raise HTTPException(status_code=status_code, detail=exc.message) from exc
        return caller

    @app.post(
        "/validator/miner-task-batches/batch",
        response_model=BatchAcceptResponse,
        responses={500: {"model": ValidatorInternalErrorResponse}},
        description="Accept a miner task batch and start processing it.",
    )
    async def accept_batch(
        request: Request,
        payload: MinerTaskBatchRequestModel,
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        caller: str = Security(require_bittensor_caller),
    ) -> BatchAcceptResponse | JSONResponse:
        try:
            batch = payload.to_domain()
            decision = deps.restore_batch.accept(batch)
            deps.restore_worker.request_restore(decision)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            return _control_route_internal_error_response(request, exc)

        return BatchAcceptResponse(status="accepted", batch_id=str(batch.batch_id), caller=caller)

    @app.post(
        "/validator/miner-task-batches/{batch_id}/similarity",
        response_model=SimilarityJudgeResponseModel,
        responses={500: {"model": ValidatorInternalErrorResponse}},
        description="Run a validator-owned similarity judge for a dethroning miner script candidate.",
    )
    async def judge_similarity(
        request: Request,
        batch_id: UUID,
        payload: SimilarityJudgeRequestModel,
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        _caller: str = Security(require_bittensor_caller),
    ) -> SimilarityJudgeResponseModel | JSONResponse:
        try:
            if deps.similarity_judge is None:
                raise HTTPException(status_code=503, detail="similarity judge is not configured")
            result = await deps.similarity_judge.judge(payload.to_domain(batch_id=batch_id))
        except HTTPException:
            raise
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            return _control_route_internal_error_response(request, exc)
        return SimilarityJudgeResponseModel.from_domain(result)

    @app.get(
        "/validator/miner-task-batches/{batch_id}/status",
        response_model=BatchProgressStatusResponse,
        responses={500: {"model": ValidatorInternalErrorResponse}},
        description="Return lightweight progress status for a miner task batch.",
    )
    def batch_status(
        request: Request,
        batch_id: UUID,
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        _caller: str = Security(require_bittensor_caller),
    ) -> BatchProgressStatusResponse | JSONResponse:
        try:
            lifecycle = deps.accept_batch.public_lifecycle_for(batch_id)
            if lifecycle is None:
                activity = deps.batch_activity.snapshot(batch_id)
                return BatchProgressStatusResponse(
                    batch_id=str(batch_id),
                    status="unknown",
                    error_code=None,
                    failure_detail=None,
                    total=0,
                    completed=0,
                    remaining=0,
                    latest_sequence=0,
                    provider_model_evidence=[],
                    activity_last_updated_at=activity.last_activity_at,
                    activity_stage=activity.last_activity_stage,
                    active_artifact_count=activity.active_artifact_count,
                    active_task_session_count=activity.active_task_session_count,
                )
            summary = deps.progress_tracker.summary(batch_id)
            activity = deps.batch_activity.snapshot(batch_id)
            provider_model_evidence = [
                _serialize_provider_evidence(entry) for entry in summary["provider_evidence"]
            ]
            if lifecycle == "failed":
                return BatchProgressStatusResponse(
                    batch_id=str(batch_id),
                    status="failed",
                    error_code=deps.accept_batch.error_code_for(batch_id),
                    failure_detail=_serialize_failure_detail(deps.accept_batch.failure_detail_for(batch_id)),
                    total=summary["total"],
                    completed=summary["completed"],
                    remaining=summary["remaining"],
                    latest_sequence=summary["latest_sequence"],
                    provider_model_evidence=provider_model_evidence,
                    activity_last_updated_at=activity.last_activity_at,
                    activity_stage=activity.last_activity_stage,
                    active_artifact_count=activity.active_artifact_count,
                    active_task_session_count=activity.active_task_session_count,
                )
            return BatchProgressStatusResponse(
                batch_id=str(batch_id),
                status=lifecycle,
                error_code=deps.accept_batch.error_code_for(batch_id),
                failure_detail=None,
                total=summary["total"],
                completed=summary["completed"],
                remaining=summary["remaining"],
                latest_sequence=summary["latest_sequence"],
                provider_model_evidence=provider_model_evidence,
                activity_last_updated_at=activity.last_activity_at,
                activity_stage=activity.last_activity_stage,
                active_artifact_count=activity.active_artifact_count,
                active_task_session_count=activity.active_task_session_count,
            )
        except Exception as exc:
            return _control_route_internal_error_response(request, exc)

    @app.get(
        "/validator/miner-task-batches/{batch_id}/runs",
        response_model=BatchProgressRunsPageResponse,
        responses={500: {"model": ValidatorInternalErrorResponse}},
        description="Return one bounded page of completed miner task runs for a batch.",
    )
    def completed_runs(
        request: Request,
        batch_id: UUID,
        after_sequence: int = Query(default=0, ge=0, description="Cursor; must be >= 0."),
        limit: int = Query(default=100, ge=1, le=500, description="Page size; must be 1..500."),
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        _caller: str = Security(require_bittensor_caller),
    ) -> BatchProgressRunsPageResponse | JSONResponse:
        try:
            lifecycle = deps.accept_batch.lifecycle_for(batch_id)
            if lifecycle is None:
                return BatchProgressRunsPageResponse(
                    batch_id=str(batch_id),
                    after_sequence=after_sequence,
                    limit=limit,
                    latest_sequence=0,
                    next_after_sequence=after_sequence,
                    has_more=False,
                    items=[],
                    failure_detail=None,
                )
            try:
                page = deps.progress_tracker.completed_run_page(
                    batch_id,
                    after_sequence=after_sequence,
                    limit=limit,
                )
                items = [_serialize_progress_detail(item) for item in page["items"]]
            except Exception as exc:
                summary = deps.progress_tracker.summary(batch_id)
                return _runs_page_internal_failure(
                    batch_id=batch_id,
                    after_sequence=after_sequence,
                    limit=limit,
                    latest_sequence=summary["latest_sequence"],
                    exc=exc,
                )
            return BatchProgressRunsPageResponse(
                batch_id=str(batch_id),
                after_sequence=page["after_sequence"],
                limit=page["limit"],
                latest_sequence=page["latest_sequence"],
                next_after_sequence=page["next_after_sequence"],
                has_more=page["has_more"],
                items=items,
                failure_detail=None,
            )
        except HTTPException:
            raise
        except Exception as exc:
            return _control_route_internal_error_response(request, exc)

    @app.get(
        _STATUS_PATH,
        response_model=ValidatorStatusResponse,
        responses={500: {"model": ValidatorInternalErrorResponse}},
        description="Return a validator status snapshot for platform health checks.",
    )
    def status(
        request: Request,
        deps: ValidatorControlDeps = Depends(get_control_deps),  # noqa: B008
        _caller: str = Security(require_bittensor_caller),
    ) -> ValidatorStatusResponse | JSONResponse:
        try:
            snapshot = deps.status_provider.snapshot()
            response = ValidatorStatusResponse(
                **snapshot,
                hotkey=deps.validator_hotkey.ss58_address,
                resource_usage=_safe_resource_usage_response(deps.resource_usage_provider),
            )
            request_ts = request.headers.get(_STATUS_TIMESTAMP_HEADER)
            if request_ts is None:
                return response
            proof_payload = _build_status_proof_payload(
                request_ts=request_ts,
                hotkey=response.hotkey,
                status=response.status,
                running=response.running,
            )
            return response.model_copy(
                update={"signature_hex": deps.validator_hotkey.sign(proof_payload).hex()}
            )
        except Exception as exc:
            return _control_route_internal_error_response(request, exc)


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


def _control_route_request_id(request: Request) -> str:
    state = request.scope.get("state", {})
    if isinstance(state, dict):
        request_id = state.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
    header_request_id = request.headers.get("x-request-id")
    if header_request_id:
        return header_request_id
    return uuid4().hex


def _unwrap_control_route_exception(exc: Exception) -> Exception:
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        first = exc.exceptions[0]
        if isinstance(first, Exception):
            return _unwrap_control_route_exception(first)
        return cast(Exception, first)
    return exc


def _control_route_internal_error_response(request: Request, exc: Exception) -> JSONResponse:
    exc = _unwrap_control_route_exception(exc)
    capture_exception(exc)
    payload = ValidatorInternalErrorResponse(
        error_code="internal_server_error",
        error_message=str(exc) or type(exc).__name__,
        exception_type=type(exc).__name__,
        request_id=_control_route_request_id(request),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
    return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))


def _serialize_restore_run(submission: MinerTaskRunSubmission) -> RestoreMinerTaskRunSubmissionModel:
    return RestoreMinerTaskRunSubmissionModel(
        batch_id=str(submission.batch_id),
        validator=ValidatorModel(uid=submission.validator_uid),
        run=RestoreMinerTaskRunModel(
            artifact_id=str(submission.run.artifact_id),
            task_id=str(submission.run.task_id),
            completed_at=submission.run.completed_at.isoformat(),
            response=submission.run.response,
        ),
        score=submission.score,
        execution_log=submission.execution_log,
        usage=_serialize_usage_block(submission.usage),
        session=_serialize_session_block(submission.session),
        specifics=submission.run.details,
    )


def _serialize_failure_detail(
    detail: ValidatorBatchFailureDetail | None,
) -> FailureDetailResponse | None:
    if detail is None:
        return None
    return FailureDetailResponse.from_domain(detail)


def _serialize_provider_evidence(entry: ProviderFailureEvidence) -> ProviderEvidenceModel:
    return ProviderEvidenceModel(
        provider=entry["provider"],
        model=entry["model"],
        total_calls=entry["total_calls"],
        failed_calls=entry["failed_calls"],
        failure_reason=entry.get("failure_reason"),
    )


def _serialize_progress_detail(item: object) -> SequencedProgressDetailModel:
    if not isinstance(item, dict):
        raise RuntimeError("progress detail item is invalid")
    detail = cast(SequencedProgressDetail, item)
    kind = detail["kind"]
    if kind == "completed_run":
        submission = detail["submission"]
        if submission is None:
            raise RuntimeError("completed progress detail missing submission")
        return SequencedProgressDetailModel(
            sequence=detail["sequence"],
            kind="completed_run",
            submission=_serialize_restore_run(submission),
            attempt=None,
        )
    if kind == "terminated_attempt":
        attempt = detail["attempt"]
        if not isinstance(attempt, MinerTaskAttemptAuditRecord):
            raise RuntimeError("terminated attempt progress detail missing attempt")
        return SequencedProgressDetailModel(
            sequence=detail["sequence"],
            kind="terminated_attempt",
            submission=None,
            attempt=_serialize_attempt(attempt),
        )
    raise RuntimeError("progress detail item has unsupported kind")


def _serialize_attempt(attempt: MinerTaskAttemptAuditRecord) -> MinerTaskAttemptAuditModel:
    return MinerTaskAttemptAuditModel(
        validator_session_id=str(attempt.validator_session_id),
        batch_id=str(attempt.batch_id),
        artifact_id=str(attempt.artifact_id),
        task_id=str(attempt.task_id),
        attempt_number=attempt.attempt_number,
        uid=attempt.uid,
        miner_hotkey_ss58=attempt.miner_hotkey_ss58,
        started_at=attempt.started_at,
        finished_at=attempt.finished_at,
        status=attempt.status.value,
        error_code=attempt.error_code,
        error_summary_code=attempt.error_summary_code,
        retry_decision=attempt.retry_decision.value,
        terminal_effect=attempt.terminal_effect.value,
        max_attempts=attempt.max_attempts,
    )


def _runs_page_internal_failure(
    *,
    batch_id: UUID,
    after_sequence: int,
    limit: int,
    latest_sequence: int,
    exc: Exception,
) -> BatchProgressRunsPageResponse:
    capture_exception(exc)
    return BatchProgressRunsPageResponse(
        batch_id=str(batch_id),
        after_sequence=after_sequence,
        limit=limit,
        latest_sequence=latest_sequence,
        next_after_sequence=after_sequence,
        has_more=after_sequence < latest_sequence,
        items=[],
        failure_detail=FailureDetailResponse(
            error_code=MinerTaskErrorCode.PROGRESS_SNAPSHOT_FAILED,
            error_message=str(exc) or type(exc).__name__,
            exception_type=type(exc).__name__,
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            occurred_at=datetime.now(UTC).isoformat(),
        ),
    )


def _serialize_usage_block(usage: TokenUsageSummary) -> UsageModel:
    return UsageModel(
        total_prompt_tokens=usage.total_prompt_tokens,
        total_completion_tokens=usage.total_completion_tokens,
        total_tokens=usage.total_tokens,
        call_count=usage.call_count,
        by_provider=_serialize_usage_providers(usage),
    )


def _serialize_resource_usage(
    snapshot: ValidatorResourceUsageSnapshot,
) -> ValidatorResourceUsageResponse:
    return ValidatorResourceUsageResponse(
        captured_at=snapshot.captured_at.isoformat(),
        cpu_percent=snapshot.cpu_percent,
        cpu_capacity_cores=snapshot.cpu_capacity_cores,
        memory_used_bytes=snapshot.memory_used_bytes,
        memory_total_bytes=snapshot.memory_total_bytes,
        memory_percent=snapshot.memory_percent,
        disk_used_bytes=snapshot.disk_used_bytes,
        disk_total_bytes=snapshot.disk_total_bytes,
        disk_percent=snapshot.disk_percent,
    )


def _safe_resource_usage_response(
    provider: ResourceUsageProvider,
) -> ValidatorResourceUsageResponse | None:
    try:
        return _serialize_resource_usage(provider.snapshot())
    except Exception:
        logger.exception("validator resource usage sampling failed")
        return None


def _build_status_proof_payload(
    *,
    request_ts: str,
    hotkey: str,
    status: str,
    running: bool,
) -> bytes:
    return "\n".join(
        (
            "validator-status-v1",
            f"path={_STATUS_PATH}",
            f"request_ts={request_ts}",
            f"hotkey={hotkey}",
            f"status={status}",
            f"running={running}",
        )
    ).encode("utf-8")


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


__all__ = [
    "ToolRouteDeps",
    "ValidatorControlDeps",
    "add_tool_routes",
    "add_system_routes",
    "add_control_routes",
]
