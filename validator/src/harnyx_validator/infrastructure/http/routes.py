"""HTTP route definitions for the validator API."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from harnyx_commons.bittensor import VerificationError
from harnyx_commons.errors import ConcurrencyLimitError, ToolProviderError
from harnyx_commons.llm.provider_error_summary import public_llm_provider_failure_summary
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest, SimilarityJudgeResult
from harnyx_commons.protocol_headers import SESSION_ID_HEADER
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, execute_tool_with_concurrency_permit
from harnyx_commons.tools.http_models import ToolExecuteResponseDTO
from harnyx_commons.tools.http_serialization import serialize_tool_execute_response
from harnyx_commons.tools.token_semaphore import ToolConcurrencyLimiter
from harnyx_miner_sdk.tools.http_models import ToolExecuteRequestDTO
from harnyx_validator.application.platform_tool_proxy import PlatformToolProxyScopeRegistry
from harnyx_validator.application.ports.platform import PlatformToolProxyPlatformPort
from harnyx_validator.application.status import BatchActivityTracker, StatusProvider
from harnyx_validator.infrastructure.http.schemas import (
    SimilarityJudgeFailureResponseModel,
    SimilarityJudgeRequestModel,
    SimilarityJudgeResponseModel,
    ValidatorInternalErrorResponse,
    ValidatorResourceUsageResponse,
    ValidatorStatusResponse,
)
from harnyx_validator.infrastructure.observability.sentry import capture_exception
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageSnapshot

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
    status_provider: StatusProvider
    auth: ControlRouteAuth
    validator_hotkey: StatusSigner
    resource_usage_provider: ResourceUsageProvider
    batch_activity: BatchActivityTracker
    is_chutes_configured: bool = False
    is_openrouter_configured: bool = False
    platform_tool_proxy_platform: PlatformToolProxyPlatformPort | None = None
    platform_tool_proxy_scopes: PlatformToolProxyScopeRegistry | None = None
    similarity_judge: SimilarityJudgePort | None = None


class StatusSigner(Protocol):
    ss58_address: str

    def sign(self, payload: bytes) -> bytes: ...


class ResourceUsageProvider(Protocol):
    def snapshot(self) -> ValidatorResourceUsageSnapshot: ...


class SimilarityJudgePort(Protocol):
    async def judge(self, request: SimilarityJudgeRequest) -> SimilarityJudgeResult: ...


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
        "/validator/miner-task-batches/{batch_id}/similarity",
        response_model=SimilarityJudgeResponseModel,
        responses={
            502: {"model": SimilarityJudgeFailureResponseModel},
            500: {"model": ValidatorInternalErrorResponse},
        },
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
            error_response = SimilarityJudgeFailureResponseModel(
                error_code="similarity_judge_failed",
                retryable=False,
                detail=str(exc),
                judge_usage=getattr(exc, "judge_usage", None),
            )
            return JSONResponse(
                status_code=502,
                content=error_response.model_dump(mode="json"),
            )
        except Exception as exc:
            return _control_route_internal_error_response(request, exc)
        return SimilarityJudgeResponseModel.from_domain(result)

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
                is_chutes_configured=deps.is_chutes_configured,
                is_openrouter_configured=deps.is_openrouter_configured,
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
            return response.model_copy(update={"signature_hex": deps.validator_hotkey.sign(proof_payload).hex()})
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
    platform_provider_message = _platform_tool_proxy_provider_failure_message(exc)
    if platform_provider_message is not None:
        return platform_provider_message
    if isinstance(exc, PermissionError):
        return "session token rejected"
    if isinstance(exc, LookupError):
        return "session not found"
    if isinstance(exc, ConcurrencyLimitError):
        return "tool concurrency limit reached"
    if isinstance(exc, ValueError):
        return "tool response validation failed"
    return "tool execution failed"


def _platform_tool_proxy_provider_failure_message(exc: Exception) -> str | None:
    if not isinstance(exc, ToolProviderError):
        return None
    error_code = getattr(exc, "error_code", None)
    if error_code != "provider_failed":
        return None
    return public_llm_provider_failure_summary(str(exc))


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


__all__ = [
    "ToolRouteDeps",
    "ValidatorControlDeps",
    "add_tool_routes",
    "add_system_routes",
    "add_control_routes",
]
