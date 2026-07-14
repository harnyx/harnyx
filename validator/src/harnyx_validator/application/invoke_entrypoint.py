"""Use case for invoking the miner query entrypoint under concurrency control."""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError

from harnyx_commons.application.miner_response_hydration import (
    MinerResponsePayloadError,
    hydrate_miner_response_payload,
)
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.ports.session_registry import SessionRegistryPort
from harnyx_commons.application.ports.token_registry import TokenRegistryPort
from harnyx_commons.domain.session import Session, SessionStatus
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.sandbox.client import SandboxClient, SandboxInvokeError
from harnyx_validator.application.dto.evaluation import EntrypointInvocationRequest, EntrypointInvocationResult

QUERY_ENTRYPOINT = "query"


class SandboxInvocationError(RuntimeError):
    """Raised when a sandbox entrypoint fails to execute."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        detail_code: str | None,
        detail_exception: str | None,
        detail_error: str | None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail_code = detail_code
        self.detail_exception = detail_exception
        self.detail_error = detail_error


class MinerResponseValidationError(RuntimeError):
    """Raised when miner output violates the response contract."""


class EntrypointInvoker:
    """Coordinates entrypoint invocation with token concurrency enforcement."""

    def __init__(
        self,
        session_registry: SessionRegistryPort,
        sandbox_client: SandboxClient,
        token_registry: TokenRegistryPort,
        receipt_log: ReceiptLogPort,
    ) -> None:
        self._sessions = session_registry
        self._sandbox = sandbox_client
        self._tokens = token_registry
        self._receipts = receipt_log

    async def invoke(self, request: EntrypointInvocationRequest) -> EntrypointInvocationResult:
        """Invoke the requested entrypoint after validating the session token."""
        session = self._load_session(request.session_id)
        self._validate_session(session, request)
        payload = await self._invoke_query(request=request, session=session)
        self._raise_if_session_exhausted(session.session_id)
        receipts = tuple(self._receipts.for_session(session.session_id))
        try:
            hydrated_response = hydrate_miner_response_payload(
                payload,
                query=request.query,
                session_id=session.session_id,
                receipt_log=self._receipts,
            )
        except (MinerResponsePayloadError, ValidationError) as exc:
            raise MinerResponseValidationError("miner returned invalid response payload") from exc
        return EntrypointInvocationResult(
            response=hydrated_response,
            tool_receipts=receipts,
        )

    async def _invoke_query(
        self,
        *,
        request: EntrypointInvocationRequest,
        session: Session,
    ) -> object:
        token = request.token
        try:
            return await self._sandbox.invoke(
                QUERY_ENTRYPOINT,
                payload=request.query.model_dump(mode="json"),
                context={},
                token=token,
                session_id=session.session_id,
            )
        except SandboxInvokeError as exc:
            self._raise_if_session_exhausted(session.session_id, cause=exc)
            identifier = f"session={session.session_id} uid={request.uid} entrypoint={QUERY_ENTRYPOINT}"
            message = f"sandbox invocation failed ({identifier}): {exc}"
            raise SandboxInvocationError(
                message,
                status_code=exc.status_code,
                detail_code=exc.detail_code,
                detail_exception=exc.detail_exception,
                detail_error=exc.detail_error,
            ) from exc
        except Exception as exc:
            self._raise_if_session_exhausted(session.session_id, cause=exc)
            identifier = f"session={session.session_id} uid={request.uid} entrypoint={QUERY_ENTRYPOINT}"
            message = f"sandbox invocation failed ({identifier}): {exc}"
            raise SandboxInvocationError(
                message,
                status_code=0,
                detail_code=None,
                detail_exception=exc.__class__.__name__,
                detail_error=str(exc),
            ) from exc

    def _load_session(self, session_id: UUID) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found")
        if session.status is not SessionStatus.ACTIVE:
            raise RuntimeError(f"session {session_id} is not active")
        return session

    def _validate_session(self, session: Session, request: EntrypointInvocationRequest) -> None:
        if session.uid != request.uid:
            raise PermissionError("session UID does not match invocation UID")
        if not self._tokens.verify(session.session_id, request.token):
            raise PermissionError("invalid session token presented for entrypoint invocation")

    def _raise_if_session_exhausted(
        self,
        session_id: UUID,
        *,
        cause: Exception | None = None,
    ) -> None:
        session = self._load_post_invoke_session(session_id)
        if session.status is not SessionStatus.EXHAUSTED:
            return
        message = f"session {session_id} exhausted during entrypoint invocation"
        if cause is None:
            raise SessionBudgetExhaustedError(message)
        raise SessionBudgetExhaustedError(message) from cause

    def _load_post_invoke_session(self, session_id: UUID) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found after entrypoint invocation")
        return session

__all__ = [
    "EntrypointInvoker",
    "MinerResponseValidationError",
    "SandboxClient",
    "SandboxInvocationError",
]
