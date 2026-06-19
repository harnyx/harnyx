from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from harnyx_commons.application.dto.session import SessionTokenRequest
from harnyx_commons.application.ports.session_registry import SessionRegistryPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.session import Session, SessionFailureCode, SessionStatus, SessionUsage
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry


class FakeSessionRegistry(SessionRegistryPort):
    def __init__(self) -> None:
        self._sessions: dict[UUID, Session] = {}

    def create(self, session: Session) -> None:  # pragma: no cover - trivial
        self._sessions[session.session_id] = session

    def get(self, session_id: UUID) -> Session | None:
        return self._sessions.get(session_id)

    def update(self, session: Session) -> None:  # pragma: no cover - trivial
        self._sessions[session.session_id] = session

    def mutate(self, session_id: UUID, mutate: Callable[[Session], Session]) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found")
        updated = mutate(session)
        self._sessions[session_id] = updated
        return updated

    def delete(self, session_id: UUID) -> None:  # pragma: no cover - trivial
        self._sessions.pop(session_id, None)

    def values(self) -> Iterable[Session]:  # pragma: no cover - helper
        return self._sessions.values()


class InterleavingSessionRegistry(FakeSessionRegistry):
    def __init__(self) -> None:
        super().__init__()
        self._pending: dict[UUID, Session] = {}

    def inject_before_next_mutate(self, session: Session) -> None:
        self._pending[session.session_id] = session

    def mutate(self, session_id: UUID, mutate: Callable[[Session], Session]) -> Session:
        injected = self._pending.pop(session_id, None)
        if injected is not None:
            self._sessions[session_id] = injected
        return super().mutate(session_id, mutate)


def make_request(token: str | None = None, *, budget_usd: float = 0.1) -> SessionTokenRequest:
    issued_at = datetime(2025, 10, 17, 12, tzinfo=UTC)
    expires_at = issued_at + timedelta(hours=1)
    return SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=issued_at,
        expires_at=expires_at,
        budget_usd=budget_usd,
        token=token or uuid4().hex,
    )


def test_issue_persists_session_and_token() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)

    request = make_request()
    issued = manager.issue(request)

    stored = sessions.get(request.session_id)
    assert stored is not None
    assert issued.session == stored
    assert issued.session.budget_usd == pytest.approx(request.budget_usd)
    assert issued.session.effective_hard_limit_usd == pytest.approx(request.budget_usd)
    assert tokens.verify(request.session_id, request.token)

    envelope = manager.load(request.session_id)
    assert envelope is not None
    assert envelope.token_hash == issued.token_hash
    assert envelope.session == stored


def test_mark_status_updates_session() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)

    envelope = manager.mark_status(request.session_id, SessionStatus.COMPLETED)
    assert envelope.session.status is SessionStatus.COMPLETED


def test_missing_session_raises_on_mark_status() -> None:
    manager = SessionManager(FakeSessionRegistry(), InMemoryTokenRegistry())
    with pytest.raises(LookupError):
        manager.mark_status(uuid4(), SessionStatus.COMPLETED)


def test_issue_preserves_explicit_hard_limit() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=uuid4().hex,
        hard_limit_usd=1.0,
    )

    issued = manager.issue(request)

    assert issued.session.budget_usd == pytest.approx(0.5)
    assert issued.session.effective_hard_limit_usd == pytest.approx(1.0)


def test_issue_preserves_miner_hotkey_metadata() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    request = replace(request, miner_hotkey_ss58="selected-hotkey")

    issued = manager.issue(request)

    assert issued.session.miner_hotkey_ss58 == "selected-hotkey"
    assert sessions.get(request.session_id) == issued.session


def test_session_token_request_rejects_blank_miner_hotkey() -> None:
    with pytest.raises(ValueError, match="miner_hotkey_ss58 must be non-empty"):
        replace(make_request(), miner_hotkey_ss58=" ")


def test_begin_attempt_advances_counter_and_clears_failure_marker() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)
    manager.mark_failure_code(request.session_id, SessionFailureCode.TOOL_PROVIDER_FAILED)
    next_token = uuid4().hex

    envelope = manager.begin_attempt(request.session_id, token=next_token)

    assert envelope.session.active_attempt == 1
    assert tokens.verify(request.session_id, next_token)
    assert not tokens.verify(request.session_id, request.token)
    assert envelope.session.failure_code is None
    assert envelope.session.failure_attempt is None


def test_consume_failure_code_returns_current_attempt_marker_and_clears_it() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)
    manager.begin_attempt(request.session_id, token=uuid4().hex)
    manager.mark_failure_code(request.session_id, SessionFailureCode.TOOL_PROVIDER_FAILED)

    assert manager.consume_failure_code(request.session_id) is SessionFailureCode.TOOL_PROVIDER_FAILED

    stored = sessions.get(request.session_id)
    assert stored is not None
    assert stored.failure_code is None
    assert stored.failure_attempt is None


def test_consume_failure_code_discards_stale_attempt_marker() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)
    manager.begin_attempt(request.session_id, token=uuid4().hex)
    stored = sessions.get(request.session_id)
    assert stored is not None
    sessions.update(
        replace(
            stored,
            active_attempt=2,
            failure_code=SessionFailureCode.TOOL_PROVIDER_FAILED,
            failure_attempt=1,
        )
    )

    assert manager.consume_failure_code(request.session_id) is None

    updated = sessions.get(request.session_id)
    assert updated is not None
    assert updated.failure_code is None
    assert updated.failure_attempt is None


def test_mark_failure_code_preserves_latest_concurrent_session_state() -> None:
    sessions = InterleavingSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)

    stored = sessions.get(request.session_id)
    assert stored is not None
    sessions.inject_before_next_mutate(
        replace(
            stored.mark_exhausted(),
            usage=SessionUsage(total_cost_usd=0.0001),
        )
    )

    envelope = manager.mark_failure_code(request.session_id, SessionFailureCode.TOOL_PROVIDER_FAILED)

    assert envelope.session.status is SessionStatus.EXHAUSTED
    assert envelope.session.usage.total_cost_usd == pytest.approx(0.0001)
    assert envelope.session.failure_code is SessionFailureCode.TOOL_PROVIDER_FAILED
    assert envelope.session.failure_attempt == 0


def test_consume_failure_code_preserves_latest_concurrent_session_state() -> None:
    sessions = InterleavingSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)

    stored = sessions.get(request.session_id)
    assert stored is not None
    sessions.inject_before_next_mutate(
        replace(
            stored.mark_exhausted(),
            active_attempt=1,
            usage=SessionUsage(total_cost_usd=0.0001),
            failure_code=SessionFailureCode.TOOL_PROVIDER_FAILED,
            failure_attempt=1,
        )
    )

    failure_code = manager.consume_failure_code(request.session_id)

    assert failure_code is SessionFailureCode.TOOL_PROVIDER_FAILED
    updated = sessions.get(request.session_id)
    assert updated is not None
    assert updated.status is SessionStatus.EXHAUSTED
    assert updated.active_attempt == 1
    assert updated.usage.total_cost_usd == pytest.approx(0.0001)
    assert updated.failure_code is None
    assert updated.failure_attempt is None


def test_begin_attempt_replaces_token_before_advancing_active_attempt() -> None:
    sessions = InterleavingSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    initial_token = uuid4().hex
    next_token = uuid4().hex
    request = make_request(token=initial_token)
    manager.issue(request)

    stored = sessions.get(request.session_id)
    assert stored is not None
    sessions.inject_before_next_mutate(stored)

    issued = manager.begin_attempt(request.session_id, token=next_token)

    assert issued.session.active_attempt == 1
    assert tokens.verify(request.session_id, next_token)
    assert not tokens.verify(request.session_id, initial_token)
