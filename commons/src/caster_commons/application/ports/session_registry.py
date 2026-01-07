"""Port describing runtime session state access."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from caster_commons.domain.session import Session


class SessionRegistryPort(Protocol):
    """In-memory registry for issued sessions."""

    def create(self, session: Session) -> None:
        """Store a newly issued session."""

    def get(self, session_id: UUID) -> Session | None:
        """Return the session identified by ``session_id``."""

    def update(self, session: Session) -> None:
        """Persist an updated session snapshot."""

    def delete(self, session_id: UUID) -> None:
        """Remove the session, if present."""


__all__ = ["SessionRegistryPort"]
