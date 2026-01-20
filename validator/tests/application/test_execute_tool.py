from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from caster_commons.domain.session import Session, SessionStatus, SessionUsage
from caster_commons.domain.tool_call import ToolCallOutcome
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmMessageContentPart, LlmResponse, LlmUsage
from caster_commons.tools.dto import ToolInvocationRequest
from caster_commons.tools.executor import ToolExecutor, ToolInvoker
from caster_commons.tools.usage_tracker import UsageTracker
from caster_validator.domain.exceptions import BudgetExceededError
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

pytestmark = pytest.mark.anyio("asyncio")


def generate_token() -> str:
    return uuid4().hex


class RecordingToolInvoker(ToolInvoker):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def invoke(
        self,
        tool_name: str,
        *,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((tool_name, args, kwargs))
        return {"data": [], "query": kwargs.get("query", "")}


def make_session() -> Session:
    return Session(
        session_id=uuid4(),
        uid=7,
        claim_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        usage=SessionUsage(),
        status=SessionStatus.ACTIVE,
    )


def make_request(session: Session, *, token: str) -> ToolInvocationRequest:
    return ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="search_web",
        args=("caster subnet",),
        kwargs={"query": "caster subnet"},
    )


def build_executor(
    session: Session,
    *,
    token: str,
    budget_usd: float | None = None,
    clock: Callable[[], datetime] | None = None,
) -> tuple[
    ToolExecutor,
    RecordingToolInvoker,
    FakeReceiptLog,
    FakeSessionRegistry,
    InMemoryTokenRegistry,
]:
    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    invoker = RecordingToolInvoker()
    usage_tracker = UsageTracker(budget_usd if budget_usd is not None else 0.1)
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=invoker,
        token_registry=token_registry,
        clock=clock or (lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC)),
    )
    return executor, invoker, receipt_log, session_registry, token_registry


async def test_execute_tool_records_receipt_and_updates_budget() -> None:
    session = make_session()
    token = generate_token()
    executor, invoker, receipt_log, session_registry, token_registry = build_executor(
        session,
        token=token,
    )
    request = make_request(session, token=token)

    result = await executor.execute(request)

    assert invoker.calls == [
        ("search_web", ("caster subnet",), {"query": "caster subnet"})
    ]
    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0025)

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert receipt.outcome is ToolCallOutcome.OK

    assert token_registry.verify(session.session_id, token)
    assert result.response_payload["data"] == []


async def test_execute_tool_prices_search_ai_by_referenceable_results() -> None:
    session = make_session()
    token = generate_token()

    class SearchAiInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> dict[str, object]:
            assert tool_name == "search_ai"
            return {
                "data": [
                    {"url": "https://a.example", "note": "A"},
                    {"url": None, "note": "missing"},
                    {"url": "https://b.example", "note": "B"},
                ]
            }

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker(cost_limit_usd=1.0)
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=SearchAiInvoker(),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="search_ai",
        args=(),
        kwargs={"prompt": "caster subnet", "tools": ["web"], "count": 3},
    )

    result = await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.008)

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert receipt.metadata.cost_usd == pytest.approx(0.008)
    assert len(receipt.metadata.results) == 2


async def test_execute_tool_logs_response_preview(caplog: pytest.LogCaptureFixture) -> None:
    session = make_session()
    token = generate_token()
    executor, *_ = build_executor(session, token=token)
    request = make_request(session, token=token)

    with caplog.at_level("INFO", logger="caster_commons.tools"):
        await executor.execute(request)

    completed = next(
        record
        for record in caplog.records
        if record.message.startswith("tool call completed:")
    )
    assert "response_preview={'data': [], 'query': 'caster subnet'}" in completed.message
    assert completed.response_preview == "{'data': [], 'query': 'caster subnet'}"
    assert completed.results_preview == "()"


async def test_execute_tool_rejects_unknown_session() -> None:
    session = make_session()
    token = generate_token()
    executor, *_ = build_executor(session, token=token)

    request = ToolInvocationRequest(
        session_id=uuid4(),
        token=token,
        tool="search_web",
        args=(),
        kwargs={},
    )

    with pytest.raises(LookupError):
        await executor.execute(request)


async def test_execute_tool_rejects_invalid_token() -> None:
    session = make_session()
    valid_token = generate_token()
    invalid_token = generate_token()
    executor, *_ = build_executor(session, token=valid_token)
    request = make_request(session, token=invalid_token)

    with pytest.raises(PermissionError):
        await executor.execute(request)


async def test_execute_tool_enforces_budget() -> None:
    session = make_session()
    limit = 0.003
    token = generate_token()
    executor, *_ = build_executor(session, token=token, budget_usd=limit)
    first = make_request(session, token=token)
    await executor.execute(first)

    with pytest.raises(BudgetExceededError):
        await executor.execute(make_request(session, token=token))


async def test_execute_tool_rejects_expired_session() -> None:
    session = make_session()
    token = generate_token()
    def expired_clock() -> datetime:
        return session.expires_at + timedelta(seconds=1)
    executor, *_ = build_executor(session, token=token, clock=expired_clock)
    request = make_request(session, token=token)

    with pytest.raises(RuntimeError, match="expired at"):
        await executor.execute(request)


async def test_execute_tool_records_llm_tokens_for_llm_chat() -> None:
    session = make_session()
    token = generate_token()

    class UsageToolInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> dict[str, object]:
            response = LlmResponse(
                id="offline-chutes",
                choices=(
                    LlmChoice(
                        index=0,
                        message=LlmChoiceMessage(
                            role="assistant",
                            content=(LlmMessageContentPart(type="text", text="ok"),),
                        ),
                    ),
                ),
                usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
            payload = response.to_payload()
            payload["_caster_provider"] = "llm"
            payload["_caster_model"] = "openai/gpt-oss-20b"
            return payload

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker(cost_limit_usd=1.0)
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=UsageToolInvoker(),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="llm_chat",
        args=(),
        kwargs={
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    result = await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.llm_tokens_last_call == 15
    usage_totals = stored_session.usage.llm_usage_totals["chutes"]["openai/gpt-oss-20b"]
    assert usage_totals.prompt_tokens == 10
    assert usage_totals.completion_tokens == 5
    assert usage_totals.total_tokens == 15
    assert usage_totals.call_count == 1
    assert result.response_payload["usage"]["total_tokens"] == 15
