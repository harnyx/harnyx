from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import ToolCallOutcome
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.pricing import parse_tool_model, price_llm
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmMessageContentPart, LlmResponse, LlmUsage
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, ToolInvoker
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.application.evaluate_task_run import UsageSummarizer
from harnyx_validator.domain.exceptions import BudgetExceededError
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
        return {"data": [], "search_queries": kwargs.get("search_queries", [])}


def make_session(*, budget_usd: float = 0.1, hard_limit_usd: float | None = None) -> Session:
    return Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=budget_usd,
        hard_limit_usd=hard_limit_usd,
        usage=SessionUsage(),
        status=SessionStatus.ACTIVE,
    )


def make_request(session: Session, *, token: str) -> ToolInvocationRequest:
    return ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="search_web",
        args=(),
        kwargs={"search_queries": ["harnyx", "subnet"]},
    )


def build_executor(
    session: Session,
    *,
    token: str,
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
    usage_tracker = UsageTracker()
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
        ("search_web", (), {"search_queries": ["harnyx", "subnet"]})
    ]
    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert receipt.outcome is ToolCallOutcome.OK

    assert token_registry.verify(session.session_id, token)
    assert result.response_payload["data"] == []

async def test_execute_tool_supports_tooling_info_without_consuming_budget() -> None:
    session = make_session()
    token = generate_token()
    executor, invoker, _, session_registry, _ = build_executor(session, token=token)

    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="tooling_info",
        args=(),
        kwargs={},
    )

    result = await executor.execute(request)

    assert invoker.calls == [
        ("tooling_info", (), {})
    ]
    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)
    assert result.budget.session_budget_usd == pytest.approx(0.1)
    assert result.budget.session_hard_limit_usd == pytest.approx(0.1)
    assert result.budget.session_used_budget_usd == pytest.approx(0.0)
    assert result.budget.session_remaining_budget_usd == pytest.approx(0.1)


async def test_execute_tool_budget_is_session_scoped() -> None:
    session_a = make_session(budget_usd=0.2)
    token_a = generate_token()
    executor_a, _, _, _, _ = build_executor(session_a, token=token_a)
    result_a = await executor_a.execute(
        ToolInvocationRequest(
            session_id=session_a.session_id,
            token=token_a,
            tool="tooling_info",
            args=(),
            kwargs={},
        )
    )

    session_b = make_session(budget_usd=0.7)
    token_b = generate_token()
    executor_b, _, _, _, _ = build_executor(session_b, token=token_b)
    result_b = await executor_b.execute(
        ToolInvocationRequest(
            session_id=session_b.session_id,
            token=token_b,
            tool="tooling_info",
            args=(),
            kwargs={},
        )
    )

    assert result_a.budget.session_budget_usd == pytest.approx(0.2)
    assert result_b.budget.session_budget_usd == pytest.approx(0.7)


async def test_execute_tool_budget_snapshot_clamps_remaining_after_soft_budget_is_exhausted() -> None:
    session = make_session(budget_usd=0.0001, hard_limit_usd=0.01)
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> dict[str, object]:
            assert tool_name == "search_web"
            return {
                "data": [
                    {"link": "https://a.example", "snippet": "A"},
                    {"link": "https://b.example", "snippet": "B"},
                ]
            }

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker()
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=SearchWebInvoker(),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    result = await executor.execute(make_request(session, token=token))

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0002)
    assert result.budget.session_budget_usd == pytest.approx(0.0001)
    assert result.budget.session_hard_limit_usd == pytest.approx(0.01)
    assert result.budget.session_used_budget_usd == pytest.approx(0.0002)
    assert result.budget.session_remaining_budget_usd == pytest.approx(0.0)


async def test_execute_tool_prices_search_web_by_referenceable_results() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> dict[str, object]:
            assert tool_name == "search_web"
            return {
                "data": [
                    {"link": "https://a.example", "snippet": "A"},
                    {"link": "", "snippet": "ignored"},
                    {"link": "https://b.example", "snippet": "B"},
                ]
            }

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker()
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=SearchWebInvoker(),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    result = await executor.execute(make_request(session, token=token))

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0002)

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert receipt.metadata.cost_usd == pytest.approx(0.0002)
    assert len(receipt.metadata.results) == 2


async def test_execute_tool_prices_search_ai_by_referenceable_results() -> None:
    session = make_session(budget_usd=1.0)
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
    usage_tracker = UsageTracker()
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
        kwargs={"prompt": "harnyx subnet", "count": 3},
    )

    result = await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0008)

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert receipt.metadata.cost_usd == pytest.approx(0.0008)
    assert len(receipt.metadata.results) == 2


async def test_execute_tool_logs_response_preview(caplog: pytest.LogCaptureFixture) -> None:
    session = make_session()
    token = generate_token()
    executor, *_ = build_executor(session, token=token)
    request = make_request(session, token=token)

    with caplog.at_level("INFO", logger="harnyx_commons.tools"):
        await executor.execute(request)

    completed = next(
        record
        for record in caplog.records
        if record.message.startswith("tool call completed:")
    )
    assert "response_preview={'data': [], 'search_queries': ['harnyx', 'subnet']}" in completed.message
    assert completed.response_preview == "{'data': [], 'search_queries': ['harnyx', 'subnet']}"
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
    limit = 0.0001
    session = make_session(budget_usd=limit, hard_limit_usd=0.00015)
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> dict[str, object]:
            assert tool_name == "search_web"
            return {
                "data": [
                    {"link": "https://a.example", "snippet": "A"},
                ]
            }

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker()
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=SearchWebInvoker(),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

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


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-oss-20b",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
    ],
)
async def test_execute_tool_records_llm_tokens_for_llm_chat(model: str) -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    usage = LlmUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        reasoning_tokens=7,
    )

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
                usage=usage,
            )
            return response.to_payload()

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker()
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
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    result = await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.llm_tokens_last_call == 15
    usage_totals = stored_session.usage.llm_usage_totals["chutes"][model]
    assert usage_totals.prompt_tokens == 10
    assert usage_totals.completion_tokens == 5
    assert usage_totals.total_tokens == 15
    assert usage_totals.call_count == 1
    assert result.response_payload["usage"]["total_tokens"] == 15
    assert result.response_payload["usage"]["reasoning_tokens"] == 7
    assert stored_session.usage.total_cost_usd == pytest.approx(
        price_llm(parse_tool_model(model), usage)
    )
    assert stored_session.usage.cost_by_provider["chutes"] == pytest.approx(
        price_llm(parse_tool_model(model), usage)
    )


async def test_execute_tool_ignores_stale_response_model_metadata_for_llm_chat() -> None:
    request_model = "openai/gpt-oss-20b"
    stale_payload_model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    session = make_session(budget_usd=1.0)
    token = generate_token()
    usage = LlmUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        reasoning_tokens=7,
    )

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
                usage=usage,
            )
            payload = response.to_payload()
            payload["harnyx_model"] = stale_payload_model
            return payload

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker()
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
            "model": request_model,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stale_payload_model not in stored_session.usage.llm_usage_totals["chutes"]
    usage_totals = stored_session.usage.llm_usage_totals["chutes"][request_model]
    assert usage_totals.prompt_tokens == 10
    assert usage_totals.completion_tokens == 5
    assert usage_totals.total_tokens == 15
    assert usage_totals.call_count == 1
    assert stored_session.usage.total_cost_usd == pytest.approx(
        price_llm(parse_tool_model(request_model), usage)
    )
    assert stored_session.usage.cost_by_provider["chutes"] == pytest.approx(
        price_llm(parse_tool_model(request_model), usage)
    )


async def test_execute_tool_allows_settlement_after_sibling_exhausts_session() -> None:
    session = make_session(budget_usd=0.0001, hard_limit_usd=0.00015)
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        def __init__(self, sessions: FakeSessionRegistry) -> None:
            self._sessions = sessions

        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> dict[str, object]:
            stored = self._sessions.get(session.session_id)
            assert stored is not None
            self._sessions.update(stored.mark_exhausted())
            return {
                "data": [
                    {"link": "https://a.example", "snippet": "A"},
                ]
            }

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker()
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=SearchWebInvoker(session_registry),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    result = await executor.execute(make_request(session, token=token))

    stored = session_registry.get(session.session_id)
    assert stored is not None
    assert stored.status is SessionStatus.EXHAUSTED
    assert stored.usage.total_cost_usd == pytest.approx(0.0001)
    assert receipt_log.lookup(result.receipt.receipt_id) is not None


async def test_execute_tool_records_receipt_for_search_call_that_exhausts_budget() -> None:
    session = make_session(budget_usd=0.00005, hard_limit_usd=0.00005)
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> dict[str, object]:
            return {
                "data": [
                    {"link": "https://a.example", "snippet": "A"},
                ]
            }

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker()
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=SearchWebInvoker(),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    with pytest.raises(BudgetExceededError):
        await executor.execute(make_request(session, token=token))

    stored = session_registry.get(session.session_id)
    assert stored is not None
    assert stored.status is SessionStatus.EXHAUSTED
    assert stored.hard_limit_usd == pytest.approx(0.00005)
    assert stored.usage.total_cost_usd == pytest.approx(0.0001)

    receipts = receipt_log.for_session(session.session_id)
    assert len(receipts) == 1

    _, total_tool_usage = UsageSummarizer().summarize(stored, receipts)
    assert total_tool_usage.search_tool.call_count == 1
    assert total_tool_usage.search_tool_cost == pytest.approx(0.0001)


async def test_execute_tool_rejects_settlement_after_terminal_session_transition() -> None:
    session = make_session()
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        def __init__(self, sessions: FakeSessionRegistry) -> None:
            self._sessions = sessions

        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> dict[str, object]:
            stored = self._sessions.get(session.session_id)
            assert stored is not None
            self._sessions.update(stored.mark_error())
            return {"data": []}

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = FakeReceiptLog()
    usage_tracker = UsageTracker()
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=SearchWebInvoker(session_registry),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    with pytest.raises(RuntimeError, match="became error during tool accounting"):
        await executor.execute(make_request(session, token=token))
