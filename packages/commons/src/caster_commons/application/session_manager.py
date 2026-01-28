"""Session management use case shared across services."""

from __future__ import annotations

from uuid import UUID

from caster_commons.application.dto.session import (
    SessionEnvelope,
    SessionIssued,
    SessionTokenRequest,
)
from caster_commons.application.ports.session_registry import SessionRegistryPort
from caster_commons.application.ports.token_registry import TokenRegistryPort
from caster_commons.domain.session import Session, SessionStatus


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
            claim_id=request.claim_id,
            issued_at=request.issued_at,
            expires_at=request.expires_at,
            budget_usd=request.budget_usd,
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
        envelope = self.load(session_id)
        if envelope is None:
            raise LookupError(f"session {session_id} not found")
        session = envelope.session
        match status:
            case SessionStatus.COMPLETED:
                updated = session.mark_completed()
            case SessionStatus.EXHAUSTED:
                updated = session.mark_exhausted()
            case SessionStatus.ERROR:
                updated = session.mark_error()
            case SessionStatus.TIMED_OUT:
                updated = session.mark_timed_out()
            case SessionStatus.ACTIVE:
                updated = session
        self._sessions.update(updated)
        return SessionEnvelope(session=updated, token_hash=envelope.token_hash)

    def revoke(self, session_id: UUID) -> None:
        """Remove session/token metadata when no longer needed."""
        self._sessions.delete(session_id)
        self._tokens.revoke(session_id)


__all__ = ["SessionManager"]
