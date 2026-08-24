from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from harnyx_commons.domain.session import (
    ProviderCredentialSource,
    Session,
    SessionStatus,
    SessionUsage,
)
from harnyx_commons.domain.tool_call import (
    ToolCall,
    ToolCallOutcome,
    ToolExecutionFacts,
)
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmMessageContentPart, LlmResponse, LlmUsage
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationContext, ToolInvocationOutput, ToolInvoker
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.application.evaluate_task_run import UsageSummarizer
from harnyx_validator.domain.exceptions import BudgetExceededError
from harnyx_validator.infrastructure.tools.platform_client import (
    PlatformToolProxyBudgetExceededError,
    PlatformToolProxyInvocationError,
    PlatformToolProxyProviderError,
    PlatformToolProxyToolTimeoutError,
)
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

pytestmark = pytest.mark.anyio("asyncio")

TEST_SEARCH_COST_USD = 0.005
TEST_LLM_COST_USD = 0.0042
TEST_BYOK_COST_USD = 0.40935002
TEST_BYOK_EVIDENCE = {
    "settlement_source": "provider_returned",
    "pricing_origin": "openrouter_usage_cost_and_upstream_inference_cost",
    "provider": "openrouter",
    "model": "openai/gpt-oss-120b",
    "usage": {
        "is_byok": True,
        "cost": 0.29335727,
        "cost_details": {"upstream_inference_cost": 0.11599275},
    },
}


def generate_token() -> str:
    return uuid4().hex


def search_output(payload: dict[str, object], *, cost_usd: float = TEST_SEARCH_COST_USD) -> ToolInvocationOutput:
    return ToolInvocationOutput(
        public_payload=payload,
        actual_cost_usd=cost_usd,
        actual_cost_provider="parallel",
        actual_cost_evidence={"settlement_source": "provider_returned"},
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


class ByokLlmInvoker(ToolInvoker):
    async def invoke(
        self,
        tool_name: str,
        *,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        context: ToolInvocationContext | None = None,
    ) -> ToolInvocationOutput:
        _ = args, kwargs, context
        assert tool_name == "llm_chat"
        response = LlmResponse(
            id="openrouter-byok",
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
        return ToolInvocationOutput(
            public_payload=response.to_payload(),
            actual_cost_usd=TEST_BYOK_COST_USD,
            actual_cost_provider="openrouter",
            actual_cost_evidence=TEST_BYOK_EVIDENCE,
        )


def make_session(
    *,
    budget_usd: float = 0.1,
    hard_limit_usd: float | None = None,
    provider_credential_source: ProviderCredentialSource = ProviderCredentialSource.MINER,
) -> Session:
    return Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=budget_usd,
        hard_limit_usd=hard_limit_usd,
        provider_credential_source=provider_credential_source,
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


def make_llm_request(session: Session, *, token: str) -> ToolInvocationRequest:
    return ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="llm_chat",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )


def build_executor(
    session: Session,
    *,
    token: str,
    clock: Callable[[], datetime] | None = None,
    tool_call_observer: Callable[[Session, ToolCall], Awaitable[None]] | None = None,
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
        tool_call_observer=tool_call_observer,
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
                status_code=403,
                error_code="platform_tool_proxy_denied",
                message="platform tool proxy denied",
            ),
            ToolCallOutcome.INTERNAL_ERROR,
            "platform_tool_proxy_denied",
            "403",
        ),
        (
            PlatformToolProxyToolTimeoutError(status_code=408, message="tool timed out"),
            ToolCallOutcome.TIMEOUT,
            "tool_timeout",
            "408",
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
