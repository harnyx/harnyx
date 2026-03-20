from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from harnyx_commons.application.dto.session import SessionTokenRequest
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import Query, Response
from harnyx_commons.domain.tool_call import ReceiptMetadata, ToolCall, ToolCallOutcome
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_validator.application.dto.evaluation import EntrypointInvocationRequest
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker, SandboxClient
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

pytestmark = pytest.mark.anyio("asyncio")


class RecordingSandbox(SandboxClient):
    def __init__(self) -> None:
        self.invocations: list[tuple[str, Mapping[str, object], Mapping[str, object], str, UUID]] = []
        self.response: Mapping[str, object] = {"text": "Answer"}
        self.raise_error: Exception | None = None
        self.on_invoke: Callable[[UUID], None] | None = None

    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: Mapping[str, object],
        context: Mapping[str, object],
        token: str,
        session_id: UUID,
    ) -> Mapping[str, object]:
        self.invocations.append((entrypoint, payload, context, token, session_id))
        if self.on_invoke is not None:
            self.on_invoke(session_id)
        if self.raise_error is not None:
            raise self.raise_error
        return self.response


def _make_session_request(token: str) -> SessionTokenRequest:
    issued_at = datetime(2025, 10, 17, 12, tzinfo=UTC)
    expires_at = issued_at + timedelta(hours=1)
    return SessionTokenRequest(
        session_id=uuid4(),
        uid=42,
        task_id=uuid4(),
        issued_at=issued_at,
        expires_at=expires_at,
        budget_usd=0.1,
        token=token,
    )


def _build_invoker(token: str) -> tuple[
    EntrypointInvoker,
    RecordingSandbox,
    UUID,
    FakeSessionRegistry,
    SessionManager,
    InMemoryTokenRegistry,
    FakeReceiptLog,
]:
    session_registry = FakeSessionRegistry()
    token_registry = InMemoryTokenRegistry()
    receipt_log = FakeReceiptLog()

    manager = SessionManager(session_registry, token_registry)
    request = _make_session_request(token)
    manager.issue(request)

    sandbox = RecordingSandbox()
    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )
    return invoker, sandbox, request.session_id, session_registry, manager, token_registry, receipt_log


async def test_invoke_entrypoint_calls_query_with_query_payload() -> None:
    token = uuid4().hex
    invoker, sandbox, session_id, _, _, _, _ = _build_invoker(token)

    response = await invoker.invoke(
        EntrypointInvocationRequest(
            session_id=session_id,
            token=token,
            uid=42,
            query=Query(text="harnyx subnet"),
        ),
    )

    assert response.response == Response(text="Answer")
    assert sandbox.invocations == [
        (
            "query",
            {"text": "harnyx subnet"},
            {},
            token,
            session_id,
        ),
    ]


async def test_invoke_entrypoint_returns_tool_receipts() -> None:
    token = uuid4().hex
    invoker, _, session_id, _, _, _, receipt_log = _build_invoker(token)

    receipt = ToolCall(
        receipt_id="receipt-1",
        session_id=session_id,
        uid=42,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        metadata=ReceiptMetadata(request_hash="req", response_hash="res"),
    )
    receipt_log.record(receipt)

    result = await invoker.invoke(
        EntrypointInvocationRequest(
            session_id=session_id,
            token=token,
            uid=42,
            query=Query(text="hello"),
        ),
    )

    assert result.tool_receipts == (receipt,)


async def test_invoke_entrypoint_rejects_invalid_token() -> None:
    valid_token = uuid4().hex
    invalid_token = uuid4().hex
    invoker, _, session_id, _, _, _, _ = _build_invoker(valid_token)

    with pytest.raises(PermissionError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=invalid_token,
                uid=42,
                query=Query(text="demo"),
            ),
        )


async def test_invoke_entrypoint_recovers_after_sandbox_error() -> None:
    token = uuid4().hex
    invoker, sandbox, session_id, _, _, _, _ = _build_invoker(token)
    sandbox.raise_error = RuntimeError("sandbox failure")

    with pytest.raises(RuntimeError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=token,
                uid=42,
                query=Query(text="demo"),
            ),
        )

    sandbox.raise_error = None
    response = await invoker.invoke(
        EntrypointInvocationRequest(
            session_id=session_id,
            token=token,
            uid=42,
            query=Query(text="demo"),
        ),
    )
    assert response.response == Response(text="Answer")


async def test_invoke_entrypoint_rejects_inactive_session() -> None:
    token = uuid4().hex
    invoker, _, session_id, session_registry, _, _, _ = _build_invoker(token)

    session = session_registry.get(session_id)
    assert session is not None
    session_registry.update(session.mark_completed())

    with pytest.raises(RuntimeError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=token,
                uid=42,
                query=Query(text="demo"),
            ),
        )


async def test_invoke_entrypoint_raises_when_session_exhausts_after_successful_response() -> None:
    token = uuid4().hex
    invoker, sandbox, session_id, session_registry, _, _, _ = _build_invoker(token)

    def exhaust_session(invoked_session_id: UUID) -> None:
        session = session_registry.get(invoked_session_id)
        assert session is not None
        session_registry.update(session.mark_exhausted())

    sandbox.on_invoke = exhaust_session

    with pytest.raises(SessionBudgetExhaustedError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=token,
                uid=42,
                query=Query(text="demo"),
            ),
        )


async def test_invoke_entrypoint_raises_exhausted_when_sandbox_errors_after_exhaustion() -> None:
    token = uuid4().hex
    invoker, sandbox, session_id, session_registry, _, _, _ = _build_invoker(token)

    def exhaust_session(invoked_session_id: UUID) -> None:
        session = session_registry.get(invoked_session_id)
        assert session is not None
        session_registry.update(session.mark_exhausted())

    sandbox.on_invoke = exhaust_session
    sandbox.raise_error = RuntimeError("tool budget exceeded")

    with pytest.raises(SessionBudgetExhaustedError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=token,
                uid=42,
                query=Query(text="demo"),
            ),
        )
