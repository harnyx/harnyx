from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.application.ports.session_registry import SessionRegistryPort
from caster_commons.domain.session import Session
from caster_commons.domain.tool_call import ToolCall
from caster_commons.llm.grading import JustificationGrade
from caster_validator.application.ports.agent_registry import AgentRegistryPort
from caster_validator.domain.agent import AgentRegistry, AgentStatus


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

    def delete(self, session_id: UUID) -> None:
        self._sessions.pop(session_id, None)

    def values(self) -> Iterable[Session]:
        return self._sessions.values()


class FakeReceiptLog(ReceiptLogPort):
    """In-memory receipt log for tests."""

    def __init__(self) -> None:
        self._receipts: dict[str, ToolCall] = {}
        self._session_index: dict[UUID, set[str]] = {}

    def record(self, receipt: ToolCall) -> None:
        self._receipts[receipt.receipt_id] = receipt
        self._session_index.setdefault(receipt.session_id, set()).add(receipt.receipt_id)

    def lookup(self, receipt_id: str) -> ToolCall | None:
        return self._receipts.get(receipt_id)

    def values(self) -> Iterable[ToolCall]:
        return self._receipts.values()

    def for_session(self, session_id: UUID) -> Iterable[ToolCall]:
        receipt_ids = self._session_index.get(session_id, set())
        return tuple(self._receipts[receipt_id] for receipt_id in receipt_ids)

    def clear_session(self, session_id: UUID) -> None:
        receipt_ids = self._session_index.pop(session_id, set())
        for receipt_id in receipt_ids:
            self._receipts.pop(receipt_id, None)


class StubGrader:
    """Deterministic justification grader for tests."""

    def __init__(self, *, support_ok: bool = True, rationale: str = "ok") -> None:
        self.support_ok = support_ok
        self.rationale = rationale

    async def grade(self, **_: object) -> JustificationGrade:
        return JustificationGrade(rationale=self.rationale, support_ok=self.support_ok)


__all__ = ["FakeAgentRegistry", "FakeReceiptLog", "FakeSessionRegistry", "StubGrader"]
