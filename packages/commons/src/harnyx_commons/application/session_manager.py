"""Session management use case shared across services."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from harnyx_commons.application.dto.session import (
    SessionEnvelope,
    SessionIssued,
    SessionTokenRequest,
)
from harnyx_commons.application.ports.session_registry import SessionRegistryPort
from harnyx_commons.application.ports.token_registry import TokenRegistryPort
from harnyx_commons.domain.session import Session, SessionFailureCode, SessionStatus


class SessionManager:
    """Coordinates session issuance and lifecycle transitions."""

    def __init__(
        self,
        sessions: SessionRegistryPort,
        tokens: TokenRegistryPort,
    ) -> None:
        self._sessions = sessions
        self._tokens = tokens

    def issue(self, request: SessionTokenRequest) -> SessionIssued:
        """Issue a new session and persist it."""
        session = Session(
            session_id=request.session_id,
            uid=request.uid,
            task_id=request.task_id,
            issued_at=request.issued_at,
            expires_at=request.expires_at,
            budget_usd=request.budget_usd,
            hard_limit_usd=request.hard_limit_usd,
            miner_hotkey_ss58=request.miner_hotkey_ss58,
        )
        self._sessions.create(session)
        token_hash = self._tokens.register(session.session_id, request.token)
        return SessionIssued(session=session, token=request.token, token_hash=token_hash)

    def load(self, session_id: UUID) -> SessionEnvelope | None:
        """Return the session envelope if it exists."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        token_hash = self._tokens.get_hash(session_id)
        if token_hash is None:
            return None
        return SessionEnvelope(session=session, token_hash=token_hash)

    def inspect(self, session_id: UUID) -> SessionEnvelope:
        """Return the session envelope or raise if it is missing."""
        envelope = self.load(session_id)
        if envelope is None:
            raise LookupError(f"session {session_id} not found")
        return envelope

    def mark_status(self, session_id: UUID, status: SessionStatus) -> SessionEnvelope:
        """Update the session status and persist it."""
        return self._mutate_session(session_id, lambda session: _transition_status(session, status))

    def begin_attempt(self, session_id: UUID, *, token: str) -> SessionIssued:
        """Advance the active retry attempt and clear stale failure markers."""
        token_hash = self._tokens.register(session_id, token)
        updated = self._sessions.mutate(session_id, lambda session: session.begin_attempt())
        return SessionIssued(session=updated, token=token, token_hash=token_hash)

    def mark_failure_code(self, session_id: UUID, failure_code: SessionFailureCode) -> SessionEnvelope:
        """Attach a transient execution failure marker to the session."""
        return self._mutate_session(session_id, lambda session: session.mark_failure_code(failure_code))

    def clear_failure_code(self, session_id: UUID) -> SessionEnvelope:
        """Clear any transient execution failure marker from the session."""
        return self._mutate_session(session_id, lambda session: session.clear_failure_code())

    def consume_failure_code(self, session_id: UUID) -> SessionFailureCode | None:
        """Return and clear the current-attempt failure marker, if present."""
        failure_code: SessionFailureCode | None = None

        def transition(session: Session) -> Session:
            nonlocal failure_code
            updated, failure_code = session.consume_failure_code()
            return updated

        self._mutate_session(session_id, transition)
        return failure_code

    def revoke(self, session_id: UUID) -> None:
        """Remove session/token metadata when no longer needed."""
        self._sessions.delete(session_id)
        self._tokens.revoke(session_id)

    def _mutate_session(
        self,
        session_id: UUID,
        transition: Callable[[Session], Session],
    ) -> SessionEnvelope:
        token_hash = self._tokens.get_hash(session_id)
        if token_hash is None:
            raise LookupError(f"session {session_id} not found")
        updated = self._sessions.mutate(session_id, transition)
        return SessionEnvelope(session=updated, token_hash=token_hash)


def _transition_status(session: Session, status: SessionStatus) -> Session:
    match status:
        case SessionStatus.COMPLETED:
            return session.mark_completed()
        case SessionStatus.EXHAUSTED:
            return session.mark_exhausted()
        case SessionStatus.ERROR:
            return session.mark_error()
        case SessionStatus.TIMED_OUT:
            return session.mark_timed_out()
        case SessionStatus.ACTIVE:
            return session


__all__ = ["SessionManager"]
