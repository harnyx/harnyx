from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from caster_commons.application.dto.session import SessionTokenRequest
from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.tool_call import ReceiptMetadata, ToolCall, ToolCallOutcome
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_validator.application.dto.evaluation import EntrypointInvocationRequest
from caster_validator.application.invoke_entrypoint import EntrypointInvoker, SandboxClient
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

pytestmark = pytest.mark.anyio("asyncio")


def generate_token() -> str:
    return uuid4().hex


class RecordingSandbox(SandboxClient):
    def __init__(self) -> None:
        self.invocations: list[
            tuple[str, Mapping[str, object], Mapping[str, object], str, UUID]
        ] = []
        self.response: Mapping[str, object] = {"ok": True}
        self.raise_error: Exception | None = None

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
        if self.raise_error is not None:
            raise self.raise_error
        return self.response


def make_session_request(token: str) -> SessionTokenRequest:
    issued_at = datetime(2025, 10, 17, 12, tzinfo=UTC)
    expires_at = issued_at + timedelta(hours=1)
    return SessionTokenRequest(
        session_id=uuid4(),
        uid=42,
        claim_id=uuid4(),
        issued_at=issued_at,
        expires_at=expires_at,
        budget_usd=0.1,
        token=token,
    )


def build_entrypoint_invoker(token: str) -> tuple[
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
    request = make_session_request(token)
    manager.issue(request)

    sandbox = RecordingSandbox()
    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )
    return (
        invoker,
        sandbox,
        request.session_id,
        session_registry,
        manager,
        token_registry,
        receipt_log,
    )


async def test_invoke_entrypoint_calls_sandbox_and_releases_permit() -> None:
    token = generate_token()
    invoker, sandbox, session_id, _, _, _, _ = build_entrypoint_invoker(token)

    response = await invoker.invoke(
        EntrypointInvocationRequest(
            session_id=session_id,
            token=token,
            uid=42,
            entrypoint="evaluate_criterion",
            payload={"query": "caster subnet"},
            context={"claim_id": str(session_id)},
        ),
    )

    assert response.result == {"ok": True}
    assert sandbox.invocations == [
        (
            "evaluate_criterion",
            {"query": "caster subnet"},
            {"claim_id": str(session_id)},
            token,
            session_id,
        ),
    ]


async def test_invoke_entrypoint_returns_tool_receipts() -> None:
    token = generate_token()
    invoker, _, session_id, _, _, _, receipt_log = build_entrypoint_invoker(token)

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
            entrypoint="evaluate_criterion",
            payload={},
            context={},
        ),
    )

    assert result.tool_receipts == (receipt,)


async def test_invoke_entrypoint_rejects_invalid_token() -> None:
    valid_token = generate_token()
    invalid_token = generate_token()
    invoker, _, session_id, _, _, _, _ = build_entrypoint_invoker(valid_token)

    with pytest.raises(PermissionError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=invalid_token,
                uid=42,
                entrypoint="evaluate_criterion",
                payload={},
                context={},
            ),
        )


async def test_invoke_entrypoint_recovers_after_sandbox_error() -> None:
    token = generate_token()
    invoker, sandbox, session_id, _, _, _, _ = build_entrypoint_invoker(token)
    sandbox.raise_error = RuntimeError("sandbox failure")

    with pytest.raises(RuntimeError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=token,
                uid=42,
                entrypoint="evaluate_criterion",
                payload={},
                context={},
            ),
        )

    sandbox.raise_error = None
    response = await invoker.invoke(
        EntrypointInvocationRequest(
            session_id=session_id,
            token=token,
            uid=42,
            entrypoint="evaluate_criterion",
            payload={},
            context={},
        ),
    )
    assert response.result == {"ok": True}


async def test_invoke_entrypoint_rejects_inactive_session() -> None:
    token = generate_token()
    invoker, _, session_id, session_registry, _, _, _ = build_entrypoint_invoker(token)

    session = session_registry.get(session_id)
    assert session is not None
    session_registry.update(session.mark_completed())

    with pytest.raises(RuntimeError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=token,
                uid=42,
                entrypoint="evaluate_criterion",
                payload={},
                context={},
            ),
        )


async def test_invoke_entrypoint_rejects_mismatched_uid() -> None:
    token = generate_token()
    invoker, _, session_id, _, _, _, _ = build_entrypoint_invoker(token)

    with pytest.raises(PermissionError):
        await invoker.invoke(
            EntrypointInvocationRequest(
                session_id=session_id,
                token=token,
                uid=999,
                entrypoint="evaluate_criterion",
                payload={},
                context={},
            ),
        )
