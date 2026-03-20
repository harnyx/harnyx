"""DTOs for shared session management use cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from harnyx_commons.domain.session import Session


@dataclass(frozen=True)
class SessionTokenRequest:
    """Input payload for issuing a session token."""

    session_id: UUID
    uid: int
    task_id: UUID
    issued_at: datetime
    expires_at: datetime
    budget_usd: float
    token: str
    hard_limit_usd: float | None = None

    def __post_init__(self) -> None:
        if self.budget_usd < 0.0:
            raise ValueError("budget_usd must be non-negative")
        if self.hard_limit_usd is not None and self.hard_limit_usd < 0.0:
            raise ValueError("hard_limit_usd must be non-negative")
        effective_hard_limit_usd = (
            self.budget_usd if self.hard_limit_usd is None else self.hard_limit_usd
        )
        if effective_hard_limit_usd < self.budget_usd:
            raise ValueError("hard_limit_usd must be greater than or equal to budget_usd")


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
