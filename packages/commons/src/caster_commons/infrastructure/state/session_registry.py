"""In-memory session registry implementation."""

from __future__ import annotations

from threading import Lock
from uuid import UUID

from caster_commons.application.ports.session_registry import SessionRegistryPort
from caster_commons.domain.session import Session


class InMemorySessionRegistry(SessionRegistryPort):
    """Stores session snapshots in memory."""

    def __init__(self) -> None:
        self._sessions: dict[UUID, Session] = {}
        self._lock = Lock()

    def create(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: UUID) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
        return session

    def update(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def delete(self, session_id: UUID) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


__all__ = ["InMemorySessionRegistry"]
