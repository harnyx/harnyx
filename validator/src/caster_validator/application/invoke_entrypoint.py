"""Use case for invoking sandbox entrypoints under concurrency control."""

from __future__ import annotations

from uuid import UUID

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.application.ports.session_registry import SessionRegistryPort
from caster_commons.application.ports.token_registry import TokenRegistryPort
from caster_commons.domain.session import Session, SessionStatus
from caster_commons.sandbox.client import SandboxClient
from caster_validator.application.dto.evaluation import (
    EntrypointInvocationRequest,
    EntrypointInvocationResult,
)


class SandboxInvocationError(RuntimeError):
    """Raised when a sandbox entrypoint fails to execute."""


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

        token = request.token
        try:
            payload = await self._sandbox.invoke(
                request.entrypoint,
                payload=request.payload,
                context=request.context,
                token=token,
                session_id=session.session_id,
            )
        except Exception as exc:
            identifier = (
                f"session={session.session_id} uid={request.uid} entrypoint={request.entrypoint}"
            )
            message = f"sandbox invocation failed ({identifier}): {exc}"
            raise SandboxInvocationError(message) from exc

        receipts = tuple(self._receipts.for_session(session.session_id))
        return EntrypointInvocationResult(result=payload, tool_receipts=receipts)

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


__all__ = ["EntrypointInvoker", "SandboxClient", "SandboxInvocationError"]
