"""DTOs for shared session management use cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from caster_commons.domain.session import Session


@dataclass(frozen=True)
class SessionTokenRequest:
    """Input payload for issuing a session token."""

    session_id: UUID
    uid: int
    claim_id: UUID
    issued_at: datetime
    expires_at: datetime
    token: str


@dataclass(frozen=True)
class SessionIssued:
    """Result of successfully issuing a session."""

    session: Session
    token: str
    token_hash: str


@dataclass(frozen=True)
class SessionEnvelope:
    """Envelope returned to callers when requesting session details."""

    session: Session
    token_hash: str


__all__ = [
    "SessionEnvelope",
    "SessionIssued",
    "SessionTokenRequest",
]
