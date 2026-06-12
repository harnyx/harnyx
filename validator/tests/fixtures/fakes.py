from __future__ import annotations

from collections.abc import Callable, Iterable
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.ports.session_registry import SessionRegistryPort
from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import StartedToolCall, ToolCall
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_validator.application.ports.agent_registry import AgentRegistryPort
from harnyx_validator.domain.agent import AgentRegistry, AgentStatus


class FakeAgentRegistry(AgentRegistryPort):
    """In-memory agent registry for tests."""

    def __init__(self) -> None:
        self._agents: dict[int, AgentRegistry] = {}

    def upsert(self, agent: AgentRegistry) -> None:
        self._agents[agent.uid] = agent

    def get(self, uid: int) -> AgentRegistry | None:
        return self._agents.get(uid)

    def list_by_status(self, status: AgentStatus) -> tuple[AgentRegistry, ...]:
        return tuple(agent for agent in self._agents.values() if agent.status is status)


class FakeSessionRegistry(SessionRegistryPort):
    """In-memory session registry for tests."""

    def __init__(self) -> None:
        self._sessions: dict[UUID, Session] = {}

    def create(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: UUID) -> Session | None:
        return self._sessions.get(session_id)

    def update(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def mutate(self, session_id: UUID, mutate: Callable[[Session], Session]) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found")
        updated = mutate(session)
        self._sessions[session_id] = updated
        return updated

    def delete(self, session_id: UUID) -> None:
        self._sessions.pop(session_id, None)

    def values(self) -> Iterable[Session]:
        return self._sessions.values()


class FakeReceiptLog(ReceiptLogPort):
    """In-memory receipt log for tests."""

    def __init__(self) -> None:
        self._delegate = InMemoryReceiptLog()

    def record(self, receipt: ToolCall) -> None:
        self._delegate.record(receipt)

    def start_pending_receipt(
        self,
        *,
        started_call: StartedToolCall,
    ) -> None:
        self._delegate.start_pending_receipt(
            started_call=started_call,
        )

    def complete_pending_receipt(
        self,
        receipt: ToolCall,
        settle_usage: Callable[[], tuple[Session, bool]],
    ) -> tuple[Session, bool] | None:
        return self._delegate.complete_pending_receipt(receipt, settle_usage)

    def abandon_pending_receipt(self, receipt_id: str) -> None:
        self._delegate.abandon_pending_receipt(receipt_id)

    def lookup(self, receipt_id: str) -> ToolCall | None:
        return self._delegate.lookup(receipt_id)

    def values(self) -> Iterable[ToolCall]:
        return self._delegate.values()

    def for_session(self, session_id: UUID) -> Iterable[ToolCall]:
        return self._delegate.for_session(session_id)

    def clear_session(self, session_id: UUID) -> None:
        self._delegate.clear_session(session_id)



__all__ = ["FakeAgentRegistry", "FakeReceiptLog", "FakeSessionRegistry"]
