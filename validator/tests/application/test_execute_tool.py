from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

import harnyx_commons.tools.executor as tool_executor_module
from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import (
    ToolCall,
    ToolCallOutcome,
    ToolExecutionFacts,
)
from harnyx_commons.errors import ToolInvocationTimeoutError, ToolProviderError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmMessageContentPart, LlmResponse, LlmUsage
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationContext, ToolInvocationOutput, ToolInvoker
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.application.evaluate_task_run import UsageSummarizer
from harnyx_validator.domain.exceptions import BudgetExceededError
from harnyx_validator.infrastructure.tools.platform_client import (
    PlatformToolProxyBudgetExceededError,
    PlatformToolProxyInterruptedError,
    PlatformToolProxyInvocationError,
    PlatformToolProxyProviderError,
    PlatformToolProxyToolTimeoutError,
)
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

pytestmark = pytest.mark.anyio("asyncio")

TEST_SEARCH_COST_USD = 0.005
TEST_LLM_COST_USD = 0.0042


def generate_token() -> str:
    return uuid4().hex


def search_output(payload: dict[str, object], *, cost_usd: float = TEST_SEARCH_COST_USD) -> ToolInvocationOutput:
    return ToolInvocationOutput(
        public_payload=payload,
        actual_cost_usd=cost_usd,
        actual_cost_provider="parallel",
    )


def llm_output(
    response: LlmResponse,
    *,
    cost_usd: float = TEST_LLM_COST_USD,
    execution: ToolExecutionFacts | None = None,
) -> ToolInvocationOutput:
    return ToolInvocationOutput(
        public_payload=response.to_payload(),
        actual_cost_usd=cost_usd,
        actual_cost_provider="openrouter",
        execution=execution,
    )


class RecordingToolInvoker(ToolInvoker):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def invoke(
        self,
        tool_name: str,
        *,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        context: ToolInvocationContext | None = None,
    ) -> ToolInvocationOutput:
        self.calls.append((tool_name, args, kwargs))
        return search_output({"data": [], "search_queries": kwargs.get("search_queries", [])})


class RaisingReceiptLog(FakeReceiptLog):
    def __init__(self) -> None:
        super().__init__()
        self.attempted_receipts: list[ToolCall] = []

    def complete_pending_receipt(
        self,
        receipt: ToolCall,
        settle_usage: Callable[[], tuple[Session, bool]],
    ) -> tuple[Session, bool] | None:
        _ = settle_usage
        self.attempted_receipts.append(receipt)
        raise RuntimeError("receipt log write failed")


class BlockingLlmInvoker(ToolInvoker):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._release = asyncio.Event()

    async def invoke(
        self,
        tool_name: str,
        *,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        context: ToolInvocationContext | None = None,
    ) -> ToolInvocationOutput:
        assert tool_name == "llm_chat"
        self.started.set()
        await self._release.wait()
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
            usage=LlmUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
        return llm_output(response)

    def release(self) -> None:
        self._release.set()


class BlockingProviderErrorInvoker(ToolInvoker):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._release = asyncio.Event()

    async def invoke(
        self,
        tool_name: str,
        *,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        context: ToolInvocationContext | None = None,
    ) -> dict[str, object]:
        assert tool_name == "llm_chat"
        self.started.set()
        await self._release.wait()
        raise ToolProviderError("tool provider failed")

    def release(self) -> None:
        self._release.set()


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


def build_executor_with_invoker(
    session: Session,
    *,
    token: str,
    invoker: ToolInvoker,
) -> tuple[ToolExecutor, FakeReceiptLog, FakeSessionRegistry]:
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
        tool_invoker=invoker,
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )
    return executor, receipt_log, session_registry


def require_log_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    return next(record for record in caplog.records if record.message == message)


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
    assert stored_session.usage.total_cost_usd == pytest.approx(TEST_SEARCH_COST_USD)
    assert stored_session.usage.reference_total_cost_usd == pytest.approx(TEST_SEARCH_COST_USD)
    assert stored_session.usage.actual_total_cost_usd == pytest.approx(TEST_SEARCH_COST_USD)

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert receipt.outcome is ToolCallOutcome.OK
    assert receipt.details.extra is not None
    assert receipt.details.extra["session_active_attempt"] == "0"

    assert token_registry.verify(session.session_id, token)
    assert result.response_payload["data"] == []


async def test_execute_tool_rejects_nonfinite_provider_cost_before_settling_usage() -> None:
    session = make_session()
    token = generate_token()

    class NonFiniteCostInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            _ = tool_name, args, kwargs, context
            return search_output({"data": []}, cost_usd=float("nan"))

    executor, receipt_log, session_registry = build_executor_with_invoker(
        session,
        token=token,
        invoker=NonFiniteCostInvoker(),
    )

    with pytest.raises(ValueError, match="actual_cost_usd must be finite"):
        await executor.execute(make_request(session, token=token))

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)
    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.INTERNAL_ERROR


async def test_execute_tool_rejects_boolean_provider_cost_before_settling_usage() -> None:
    session = make_session()
    token = generate_token()

    class BooleanCostInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            _ = tool_name, args, kwargs, context
            return ToolInvocationOutput(
                public_payload={"data": []},
                actual_cost_usd=cast(float, True),
                actual_cost_provider="parallel",
            )

    executor, receipt_log, session_registry = build_executor_with_invoker(
        session,
        token=token,
        invoker=BooleanCostInvoker(),
    )

    with pytest.raises(ValueError, match="actual_cost_usd must be numeric"):
        await executor.execute(make_request(session, token=token))

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)
    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.INTERNAL_ERROR


async def test_execute_tool_does_not_settle_late_completion_after_pending_receipt_is_abandoned() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    invoker = BlockingLlmInvoker()
    executor, receipt_log, session_registry = build_executor_with_invoker(
        session,
        token=token,
        invoker=invoker,
    )
    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="llm_chat",
        args=(),
        kwargs={
            "provider": "chutes",
            "model": "zai-org/GLM-5-TEE",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    task = asyncio.create_task(executor.execute(request))
    await invoker.started.wait()
    receipt_log.clear_session(session.session_id)
    invoker.release()

    with pytest.raises(RuntimeError, match="pending receipt was abandoned"):
        await task

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert receipts == ()
    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)
    assert stored_session.usage.llm_usage_totals == {}


async def test_execute_tool_does_not_record_late_provider_failure_after_pending_receipt_is_abandoned() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    invoker = BlockingProviderErrorInvoker()
    executor, receipt_log, session_registry = build_executor_with_invoker(
        session,
        token=token,
        invoker=invoker,
    )
    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="llm_chat",
        args=(),
        kwargs={
            "provider": "chutes",
            "model": "zai-org/GLM-5-TEE",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    task = asyncio.create_task(executor.execute(request))
    await invoker.started.wait()
    receipt_log.clear_session(session.session_id)
    invoker.release()

    with pytest.raises(ToolProviderError, match="tool provider failed"):
        await task

    assert tuple(receipt_log.for_session(session.session_id)) == ()
    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)
    assert stored_session.usage.llm_usage_totals == {}


async def test_execute_tool_records_timeout_receipt_when_cancelled() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    invoker = BlockingLlmInvoker()
    executor, receipt_log, _ = build_executor_with_invoker(
        session,
        token=token,
        invoker=invoker,
    )
    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="llm_chat",
        args=(),
        kwargs={
            "provider": "chutes",
            "model": "zai-org/GLM-5-TEE",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    task = asyncio.create_task(executor.execute(request))
    await invoker.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.TIMEOUT
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["error_type"] == "CancelledError"


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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            assert tool_name == "search_web"
            return search_output(
                {
                    "data": [
                        {"link": "https://a.example", "snippet": "A"},
                        {"link": "https://b.example", "snippet": "B"},
                    ]
                },
                cost_usd=0.0002,
            )

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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            assert tool_name == "search_web"
            return ToolInvocationOutput(
                public_payload={
                    "data": [
                        {"link": "https://a.example", "snippet": "A"},
                        {"link": "", "snippet": "ignored"},
                        {"link": "https://b.example", "snippet": "B"},
                    ]
                },
                actual_cost_usd=0.005,
                actual_cost_provider="parallel",
            )

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
    assert stored_session.usage.total_cost_usd == pytest.approx(0.005)
    assert stored_session.usage.reference_total_cost_usd == pytest.approx(0.005)
    assert stored_session.usage.actual_total_cost_usd == pytest.approx(0.005)
    assert stored_session.usage.cost_by_provider == {"parallel": pytest.approx(0.005)}
    assert stored_session.usage.reference_cost_by_provider == {"parallel": pytest.approx(0.005)}
    assert stored_session.usage.actual_cost_by_provider == {"parallel": pytest.approx(0.005)}
    assert result.budget.session_used_budget_usd == pytest.approx(0.005)
    assert result.budget.session_remaining_budget_usd == pytest.approx(0.995)
    assert "actual_cost_usd" not in result.response_payload

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert receipt.details.cost_usd == pytest.approx(0.005)
    assert receipt.details.reference_cost_usd == pytest.approx(0.005)
    assert receipt.details.actual_cost_usd == pytest.approx(0.005)
    assert receipt.details.actual_cost_provider == "parallel"
    assert len(receipt.details.results) == 2


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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            assert tool_name == "search_ai"
            return search_output(
                {
                    "data": [
                        {"url": "https://a.example", "note": "A"},
                        {"url": None, "note": "missing"},
                        {"url": "https://b.example", "note": "B"},
                    ]
                },
                cost_usd=0.0008,
            )

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
        kwargs={"prompt": "harnyx subnet", "count": 10},
    )

    result = await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0008)

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert receipt.details.cost_usd == pytest.approx(0.0008)
    assert len(receipt.details.results) == 2


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


async def test_execute_tool_debug_log_includes_full_request_response_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = make_session()
    token = generate_token()
    executor, *_ = build_executor(session, token=token)
    request = make_request(session, token=token)

    with caplog.at_level("DEBUG", logger="harnyx_commons.tools"):
        await executor.execute(request)

    started = require_log_record(caplog, "miner_tool_call.started")
    completed = require_log_record(caplog, "miner_tool_call.completed")
    assert started.data["call_id"] == completed.data["call_id"]
    assert completed.data["tool_name"] == "search_web"
    assert completed.data["session_id"] == str(session.session_id)
    assert completed.data["task_id"] == str(session.task_id)
    assert completed.data["attempt"] == session.active_attempt
    assert completed.data["request"] == {
        "args": [],
        "kwargs": {"search_queries": ["harnyx", "subnet"]},
    }
    assert completed.data["response"] == {
        "data": [],
        "search_queries": ["harnyx", "subnet"],
    }
    assert completed.data["budget"]["session_budget_usd"] == pytest.approx(session.budget_usd)
    assert completed.data["error"] is None


async def test_execute_tool_debug_log_preserves_raw_payload_and_error_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = make_session()
    token = generate_token()

    class FailingInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> dict[str, object]:
            raise RuntimeError(
                'access_token=access-secret Authorization: Bearer bearer-secret '
                '{"api_key":"json-secret"} password: "password-secret" '
                '{"authorization":"Bearer quoted-bearer-secret"} token=plain-token-secret '
                '"token":"json-token-secret" '
                "provider failed GET /oauth?access_token=query-secret&error=invalid_grant status=400"
            )

    executor, _, _ = build_executor_with_invoker(
        session,
        token=token,
        invoker=FailingInvoker(),
    )
    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="search_web",
        args=(),
        kwargs={"api_key": "secret", "prompt": "visible"},
    )

    with caplog.at_level("DEBUG", logger="harnyx_commons.tools"):
        with pytest.raises(RuntimeError):
            await executor.execute(request)

    started = require_log_record(caplog, "miner_tool_call.started")
    failed = require_log_record(caplog, "miner_tool_call.failed")
    normal_failure = require_log_record(caplog, "tool call failed")
    assert started.data["request"]["kwargs"]["api_key"] == "secret"
    assert started.data["request"]["kwargs"]["prompt"] == "visible"
    assert "access-secret" in failed.data["error"]["message"]
    assert "bearer-secret" in failed.data["error"]["message"]
    assert "json-secret" in failed.data["error"]["message"]
    assert "password-secret" in failed.data["error"]["message"]
    assert "quoted-bearer-secret" in failed.data["error"]["message"]
    assert "plain-token-secret" in failed.data["error"]["message"]
    assert "json-token-secret" in failed.data["error"]["message"]
    assert "query-secret" in failed.data["error"]["message"]
    assert "&error=invalid_grant" in failed.data["error"]["message"]
    assert "access-secret" in normal_failure.error
    assert "bearer-secret" in normal_failure.error
    assert "json-secret" in normal_failure.error
    assert "password-secret" in normal_failure.error
    assert "quoted-bearer-secret" in normal_failure.error
    assert "plain-token-secret" in normal_failure.error
    assert "json-token-secret" in normal_failure.error
    assert "query-secret" in normal_failure.error
    assert "&error=invalid_grant" in normal_failure.error
    assert normal_failure.exc_info is None


async def test_execute_tool_records_failed_receipt_before_reraising_provider_error() -> None:
    session = make_session()
    token = generate_token()

    class FailingInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> dict[str, object]:
            assert tool_name == "search_web"
            raise ToolProviderError("tool provider failed") from ValueError("upstream detail")

    executor, receipt_log, session_registry = build_executor_with_invoker(
        session,
        token=token,
        invoker=FailingInvoker(),
    )

    with pytest.raises(ToolProviderError, match="tool provider failed"):
        await executor.execute(make_request(session, token=token))

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.tool == "search_web"
    assert receipt.outcome is ToolCallOutcome.PROVIDER_ERROR
    assert receipt.details.request_payload == {
        "args": [],
        "kwargs": {"search_queries": ["harnyx", "subnet"]},
    }
    assert receipt.details.response_hash is None
    assert receipt.details.response_payload is None
    assert receipt.details.results == ()
    assert receipt.details.cost_usd is None
    assert receipt.details.extra is not None
    assert receipt.details.extra["error_type"] == "ToolProviderError"
    assert receipt.details.extra["error_message"] == "upstream detail"
    assert receipt.details.extra["error_cause_type"] == "ValueError"
    assert receipt.details.extra["error_cause_message"] == "upstream detail"
    assert receipt.details.execution is not None
    assert receipt.details.execution.started_at is not None
    assert receipt.details.execution.finished_at is not None
    assert receipt.details.execution.elapsed_ms == pytest.approx(0.0)
    stored = session_registry.get(session.session_id)
    assert stored is not None
    assert stored.usage.total_cost_usd == pytest.approx(0.0)


async def test_execute_tool_records_failed_receipt_before_reraising_generic_error() -> None:
    session = make_session()
    token = generate_token()

    class FailingInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> dict[str, object]:
            raise ValueError("model openai/gpt-oss-120b is not allowed")

    executor, receipt_log, session_registry = build_executor_with_invoker(
        session,
        token=token,
        invoker=FailingInvoker(),
    )

    with pytest.raises(ValueError, match="not allowed"):
        await executor.execute(make_request(session, token=token))

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.tool == "search_web"
    assert receipt.outcome is ToolCallOutcome.INTERNAL_ERROR
    assert receipt.details.response_hash is None
    assert receipt.details.response_payload is None
    assert receipt.details.results == ()
    assert receipt.details.cost_usd is None
    assert receipt.details.extra is not None
    assert receipt.details.extra["error_type"] == "ValueError"
    assert receipt.details.extra["error_message"] == "model openai/gpt-oss-120b is not allowed"
    stored = session_registry.get(session.session_id)
    assert stored is not None
    assert stored.usage.total_cost_usd == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("exc", "expected_outcome", "expected_error_code", "expected_status_code"),
    [
        (
            PlatformToolProxyProviderError(status_code=400, message="provider failed"),
            ToolCallOutcome.PROVIDER_ERROR,
            "provider_failed",
            "400",
        ),
        (
            PlatformToolProxyInvocationError(
                status_code=502,
                error_code="platform_error",
                message="platform proxy failed",
            ),
            ToolCallOutcome.INTERNAL_ERROR,
            "platform_error",
            "502",
        ),
        (
            PlatformToolProxyInvocationError(
                status_code=403,
                error_code="platform_tool_proxy_denied",
                message="platform tool proxy denied",
            ),
            ToolCallOutcome.INTERNAL_ERROR,
            "platform_tool_proxy_denied",
            "403",
        ),
        (
            PlatformToolProxyInvocationError(
                status_code=400,
                error_code="platform_tool_proxy_grant_failed",
                message="platform tool proxy grant failed",
            ),
            ToolCallOutcome.INTERNAL_ERROR,
            "platform_tool_proxy_grant_failed",
            "400",
        ),
        (
            PlatformToolProxyToolTimeoutError(status_code=408, message="tool timed out"),
            ToolCallOutcome.TIMEOUT,
            "tool_timeout",
            "408",
        ),
        (
            PlatformToolProxyInterruptedError("platform tool proxy execution interrupted before a response"),
            ToolCallOutcome.INTERNAL_ERROR,
            "platform_interrupted",
            "0",
        ),
        (
            PlatformToolProxyBudgetExceededError(status_code=400, message="budget exhausted"),
            ToolCallOutcome.BUDGET_EXCEEDED,
            "budget_exhausted",
            "400",
        ),
    ],
)
async def test_execute_tool_records_platform_tool_proxy_category_in_failed_receipt_and_log(
    caplog: pytest.LogCaptureFixture,
    exc: Exception,
    expected_outcome: ToolCallOutcome,
    expected_error_code: str,
    expected_status_code: str,
) -> None:
    session = make_session()
    token = generate_token()

    class FailingInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> dict[str, object]:
            raise exc

    executor, receipt_log, _ = build_executor_with_invoker(
        session,
        token=token,
        invoker=FailingInvoker(),
    )

    with caplog.at_level("INFO", logger="harnyx_commons.tools"):
        with pytest.raises(type(exc)):
            await executor.execute(make_request(session, token=token))

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.outcome is expected_outcome
    assert receipt.details.extra is not None
    assert receipt.details.extra["platform_tool_proxy_error_code"] == expected_error_code
    assert receipt.details.extra["platform_tool_proxy_status_code"] == expected_status_code
    failure_log = require_log_record(caplog, "tool call failed")
    assert failure_log.platform_tool_proxy_error_code == expected_error_code
    assert failure_log.platform_tool_proxy_status_code == expected_status_code


async def test_execute_tool_records_failed_timeout_receipt_as_timeout() -> None:
    session = make_session()
    token = generate_token()

    class TimeoutInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> dict[str, object]:
            raise ToolInvocationTimeoutError("tool timed out")

    executor, receipt_log, _ = build_executor_with_invoker(
        session,
        token=token,
        invoker=TimeoutInvoker(),
    )

    with pytest.raises(ToolInvocationTimeoutError, match="tool timed out"):
        await executor.execute(make_request(session, token=token))

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.TIMEOUT
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["error_type"] == "ToolInvocationTimeoutError"


async def test_execute_tool_does_not_record_receipt_for_invalid_session_token() -> None:
    session = make_session()
    valid_token = generate_token()
    invalid_token = generate_token()
    executor, _, receipt_log, _, _ = build_executor(session, token=valid_token)

    with pytest.raises(PermissionError):
        await executor.execute(make_request(session, token=invalid_token))

    assert tuple(receipt_log.for_session(session.session_id)) == ()


async def test_execute_tool_records_failed_receipt_for_invalid_invoker_output() -> None:
    session = make_session()
    token = generate_token()

    class InvalidOutputInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> object:
            return object()

    executor, receipt_log, _ = build_executor_with_invoker(
        session,
        token=token,
        invoker=InvalidOutputInvoker(),
    )

    with pytest.raises(ValueError, match="tool invoker must return"):
        await executor.execute(make_request(session, token=token))

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.INTERNAL_ERROR
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["error_type"] == "ValueError"


async def test_execute_tool_skips_failure_debug_payload_when_debug_disabled(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session()
    token = generate_token()

    class FailingInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> dict[str, object]:
            raise RuntimeError("authorization=secret")

    def fail_debug_error_data(exc: Exception) -> dict[str, str]:
        raise AssertionError("_debug_error_data should only run when DEBUG is enabled")

    executor, _, _ = build_executor_with_invoker(
        session,
        token=token,
        invoker=FailingInvoker(),
    )
    monkeypatch.setattr(tool_executor_module, "_debug_error_data", fail_debug_error_data)

    with caplog.at_level("INFO", logger="harnyx_commons.tools"):
        with pytest.raises(RuntimeError):
            await executor.execute(make_request(session, token=token))

    assert not any(record.message == "miner_tool_call.failed" for record in caplog.records)


async def test_execute_tool_debug_logs_completion_before_budget_exhausted_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = make_session(budget_usd=0.00005, hard_limit_usd=0.00005)
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            assert tool_name == "search_web"
            return search_output({"data": [{"link": "https://a.example", "snippet": "A"}]}, cost_usd=0.0001)

    executor, _, _ = build_executor_with_invoker(
        session,
        token=token,
        invoker=SearchWebInvoker(),
    )

    with caplog.at_level("DEBUG", logger="harnyx_commons.tools"):
        with pytest.raises(BudgetExceededError):
            await executor.execute(make_request(session, token=token))

    completed = require_log_record(caplog, "miner_tool_call.completed")
    failed = require_log_record(caplog, "miner_tool_call.failed")
    assert completed.data["call_id"] == failed.data["call_id"]
    assert completed.data["response"] == {
        "data": [{"link": "https://a.example", "snippet": "A"}]
    }
    assert completed.data["cost_usd"] == pytest.approx(0.0001)
    assert completed.data["budget"]["session_used_budget_usd"] == pytest.approx(0.0001)


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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            assert tool_name == "search_web"
            return search_output(
                {
                    "data": [
                        {"link": "https://a.example", "snippet": "A"},
                    ]
                },
                cost_usd=0.0001,
            )

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


async def test_execute_tool_exhausts_budget_from_provider_actual_cost() -> None:
    session = make_session(budget_usd=0.001, hard_limit_usd=0.001)
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            assert tool_name == "search_web"
            return ToolInvocationOutput(
                public_payload={"data": [{"link": "https://a.example", "snippet": "A"}]},
                actual_cost_usd=0.005,
                actual_cost_provider="parallel",
            )

    executor, _, session_registry = build_executor_with_invoker(
        session,
        token=token,
        invoker=SearchWebInvoker(),
    )

    with pytest.raises(BudgetExceededError):
        await executor.execute(make_request(session, token=token))

    stored = session_registry.get(session.session_id)
    assert stored is not None
    assert stored.status is SessionStatus.EXHAUSTED
    assert stored.usage.total_cost_usd == pytest.approx(0.005)
    assert stored.usage.reference_total_cost_usd == pytest.approx(0.005)
    assert stored.usage.actual_total_cost_usd == pytest.approx(0.005)


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
        "zai-org/GLM-5-TEE",
        "Qwen/Qwen3.6-27B-TEE",
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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
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
            return ToolInvocationOutput(
                public_payload=response.to_payload(),
                actual_cost_usd=0.003,
                actual_cost_provider="openrouter",
            )

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
            "provider": "chutes",
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "thinking": {"enabled": True, "effort": "medium"},
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
    assert "actual_cost_usd" not in result.response_payload
    assert stored_session.usage.total_cost_usd == pytest.approx(0.003)
    assert stored_session.usage.reference_total_cost_usd == pytest.approx(0.003)
    assert stored_session.usage.actual_total_cost_usd == pytest.approx(0.003)
    assert stored_session.usage.cost_by_provider["openrouter"] == pytest.approx(0.003)
    assert stored_session.usage.reference_cost_by_provider["openrouter"] == pytest.approx(0.003)
    assert stored_session.usage.actual_cost_by_provider["openrouter"] == pytest.approx(0.003)
    assert result.budget.session_used_budget_usd == pytest.approx(0.003)


async def test_execute_tool_rejects_llm_chat_usage_when_provider_is_missing() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    model = "zai-org/GLM-5-TEE"

    class UsageToolInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
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
                usage=LlmUsage(total_tokens=15),
            )
            return llm_output(response)

    executor, receipt_log, session_registry = build_executor_with_invoker(
        session,
        token=token,
        invoker=UsageToolInvoker(),
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

    with pytest.raises(ValueError, match="llm tool request must include a 'provider' payload value"):
        await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)
    assert stored_session.usage.llm_usage_totals == {}
    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.INTERNAL_ERROR


async def test_execute_tool_records_zero_token_llm_usage_when_counters_are_missing() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    model = "deepseek/deepseek-v3.2"

    class UsageToolInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            response = LlmResponse(
                id="offline-openrouter",
                choices=(
                    LlmChoice(
                        index=0,
                        message=LlmChoiceMessage(
                            role="assistant",
                            content=(LlmMessageContentPart(type="text", text="ok"),),
                        ),
                    ),
                ),
                usage=LlmUsage(),
            )
            return llm_output(response, cost_usd=0.003)

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

    result = await executor.execute(
        ToolInvocationRequest(
            session_id=session.session_id,
            token=token,
            tool="llm_chat",
            args=(),
            kwargs={
                "provider": "openrouter",
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
    )

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.llm_tokens_last_call == 0
    usage_totals = stored_session.usage.llm_usage_totals["openrouter"][model]
    assert usage_totals.call_count == 1
    assert usage_totals.prompt_tokens == 0
    assert usage_totals.completion_tokens == 0
    assert usage_totals.total_tokens == 0
    assert usage_totals.reasoning_tokens == 0
    assert result.usage is not None
    assert result.usage.total_tokens == 0
    assert stored_session.usage.total_cost_usd == pytest.approx(0.003)


async def test_execute_tool_records_llm_tokens_for_first_positional_llm_chat_payload() -> None:
    model = "zai-org/GLM-5-TEE"
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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
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
            return llm_output(response)

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
        args=({"provider": "chutes", "model": model, "messages": [{"role": "user", "content": "ping"}]},),
        kwargs={},
    )

    result = await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.llm_tokens_last_call == 15
    usage_totals = stored_session.usage.llm_usage_totals["chutes"][model]
    assert usage_totals.total_tokens == 15
    assert usage_totals.call_count == 1
    assert result.response_payload["usage"]["total_tokens"] == 15
    assert stored_session.usage.total_cost_usd == pytest.approx(TEST_LLM_COST_USD)


async def test_execute_tool_records_selected_llm_provider_for_llm_chat_usage() -> None:
    model = "deepseek/deepseek-v3.2"
    session = make_session(budget_usd=1.0)
    token = generate_token()
    usage = LlmUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    class UsageToolInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            assert tool_name == "llm_chat"
            assert kwargs["provider"] == "openrouter"
            response = LlmResponse(
                id="offline-openrouter",
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
            return llm_output(response)

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

    await executor.execute(
        ToolInvocationRequest(
            session_id=session.session_id,
            token=token,
            tool="llm_chat",
            args=(),
            kwargs={
                "provider": "openrouter",
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
    )

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert "chutes" not in stored_session.usage.llm_usage_totals
    usage_totals = stored_session.usage.llm_usage_totals["openrouter"][model]
    assert usage_totals.prompt_tokens == 10
    assert usage_totals.completion_tokens == 5
    assert usage_totals.total_tokens == 15
    assert usage_totals.call_count == 1
    assert stored_session.usage.cost_by_provider == {
        "openrouter": pytest.approx(TEST_LLM_COST_USD),
    }


async def test_execute_tool_counts_reasoning_tokens_when_total_is_missing() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    usage = LlmUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=None,
        reasoning_tokens=7,
    )

    class UsageToolInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
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
            return llm_output(response)

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
            "provider": "chutes",
            "model": "deepseek-ai/DeepSeek-V3.2-TEE",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    result = await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.llm_tokens_last_call == 22
    usage_totals = stored_session.usage.llm_usage_totals["chutes"]["deepseek-ai/DeepSeek-V3.2-TEE"]
    assert usage_totals.prompt_tokens == 10
    assert usage_totals.completion_tokens == 5
    assert usage_totals.reasoning_tokens == 7
    assert usage_totals.total_tokens == 22
    assert result.usage is not None
    assert result.usage.total_tokens == 22
    assert result.response_payload["usage"]["total_tokens"] is None


async def test_execute_tool_counts_reasoning_only_usage_when_total_is_missing() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    usage = LlmUsage(reasoning_tokens=7)

    class UsageToolInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
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
            return llm_output(response)

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
            "provider": "chutes",
            "model": "deepseek-ai/DeepSeek-V3.2-TEE",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    result = await executor.execute(request)

    stored_session = session_registry.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.llm_tokens_last_call == 7
    usage_totals = stored_session.usage.llm_usage_totals["chutes"]["deepseek-ai/DeepSeek-V3.2-TEE"]
    assert usage_totals.prompt_tokens == 0
    assert usage_totals.completion_tokens == 0
    assert usage_totals.reasoning_tokens == 7
    assert usage_totals.total_tokens == 7
    assert result.usage is not None
    assert result.usage.total_tokens == 7
    assert result.response_payload["usage"]["total_tokens"] is None
    assert result.response_payload["usage"]["reasoning_tokens"] == 7


async def test_execute_tool_ignores_stale_response_model_metadata_for_llm_chat() -> None:
    request_model = "zai-org/GLM-5-TEE"
    stale_payload_model = "provider-returned-stale-model"
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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
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
            return ToolInvocationOutput(
                public_payload=payload,
                actual_cost_usd=TEST_LLM_COST_USD,
                actual_cost_provider="openrouter",
            )

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
            "provider": "chutes",
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
    assert stored_session.usage.total_cost_usd == pytest.approx(TEST_LLM_COST_USD)
    assert stored_session.usage.cost_by_provider["openrouter"] == pytest.approx(TEST_LLM_COST_USD)


async def test_execute_tool_records_llm_elapsed_ms_only_in_receipt_details() -> None:
    session = make_session(budget_usd=1.0)
    token = generate_token()
    precheck_at = datetime(2025, 10, 17, 12, 4, 59, tzinfo=UTC)
    started_at = datetime(2025, 10, 17, 12, 5, tzinfo=UTC)
    finished_at = datetime(2025, 10, 17, 12, 5, 2, tzinfo=UTC)
    clock_values = iter((precheck_at, started_at, finished_at, finished_at))

    class UsageToolInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
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
                usage=LlmUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            )
            return llm_output(response, execution=ToolExecutionFacts(elapsed_ms=1250.0))

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
        clock=lambda: next(clock_values),
    )
    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="llm_chat",
        args=(),
        kwargs={
            "provider": "chutes",
            "model": "zai-org/GLM-5-TEE",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    result = await executor.execute(request)

    receipt = receipt_log.lookup(result.receipt.receipt_id)
    assert receipt is not None
    assert "elapsed_ms" not in result.response_payload
    assert receipt.details.request_payload == {
        "args": [],
        "kwargs": {
            "provider": "chutes",
            "model": "zai-org/GLM-5-TEE",
            "messages": [{"role": "user", "content": "ping"}],
        },
    }
    assert receipt.details.execution == ToolExecutionFacts(
        elapsed_ms=1250.0,
        started_at=started_at,
        finished_at=finished_at,
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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            stored = self._sessions.get(session.session_id)
            assert stored is not None
            self._sessions.update(stored.mark_exhausted())
            return search_output(
                {
                    "data": [
                        {"link": "https://a.example", "snippet": "A"},
                    ]
                },
                cost_usd=0.0001,
            )

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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            return search_output(
                {
                    "data": [
                        {"link": "https://a.example", "snippet": "A"},
                    ]
                },
                cost_usd=0.0001,
            )

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


async def test_execute_tool_budget_exhaustion_records_one_successful_receipt_only() -> None:
    session = make_session(budget_usd=0.00005, hard_limit_usd=0.00005)
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            return search_output({"data": [{"link": "https://a.example", "snippet": "A"}]}, cost_usd=0.0001)

    executor, receipt_log, _ = build_executor_with_invoker(
        session,
        token=token,
        invoker=SearchWebInvoker(),
    )

    with pytest.raises(BudgetExceededError):
        await executor.execute(make_request(session, token=token))

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.outcome is ToolCallOutcome.OK
    assert receipt.details.response_payload == {
        "data": [{"link": "https://a.example", "snippet": "A"}]
    }
    assert receipt.details.extra is not None
    assert "error_type" not in receipt.details.extra


async def test_execute_tool_attempts_failed_receipt_for_receipt_persistence_failure() -> None:
    session = make_session()
    token = generate_token()

    class SearchWebInvoker(ToolInvoker):
        async def invoke(
            self,
            tool_name: str,
            *,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            return search_output({"data": []})

    session_registry = FakeSessionRegistry()
    session_registry.create(session)
    receipt_log = RaisingReceiptLog()
    token_registry = InMemoryTokenRegistry()
    token_registry.register(session.session_id, token)
    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=UsageTracker(),
        tool_invoker=SearchWebInvoker(),
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    with pytest.raises(RuntimeError, match="receipt log write failed"):
        await executor.execute(make_request(session, token=token))

    assert [receipt.outcome for receipt in receipt_log.attempted_receipts] == [
        ToolCallOutcome.OK,
        ToolCallOutcome.INTERNAL_ERROR,
    ]
    assert receipt_log.attempted_receipts[1].details.extra is not None
    assert receipt_log.attempted_receipts[1].details.extra["error_type"] == "RuntimeError"


async def test_execute_tool_preserves_successful_receipt_for_usage_settlement_failure() -> None:
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
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            stored = self._sessions.get(session.session_id)
            assert stored is not None
            self._sessions.update(stored.mark_error())
            return search_output({"data": []})

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

    receipts = tuple(receipt_log.for_session(session.session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.OK
    assert receipts[0].details.response_payload == {"data": []}
    assert receipts[0].details.extra is not None
    assert "error_type" not in receipts[0].details.extra
