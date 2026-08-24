from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.domain.session import ProviderCredentialSource, Session, SessionStatus
from harnyx_commons.domain.tool_call import SearchToolResult, ToolCallOutcome, ToolResultPolicy
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmMessageContentPart, LlmResponse, LlmUsage
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.types import SearchToolName, ToolName
from harnyx_commons.tools.usage_tracker import UsageTracker

pytestmark = pytest.mark.anyio("asyncio")


class StaticSearchInvoker:
    def __init__(self, payload: JsonObject, *, actual_cost_usd: float = 0.001) -> None:
        self._payload = payload
        self._actual_cost_usd = actual_cost_usd

    async def invoke(
        self,
        tool_name: ToolName,
        *,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
        context: ToolInvocationContext | None = None,
    ) -> ToolInvocationOutput:
        _ = tool_name, args, kwargs, context
        return ToolInvocationOutput(
            public_payload=self._payload,
            actual_cost_usd=self._actual_cost_usd,
            actual_cost_provider="test-provider",
        )


class BlockingLlmInvoker:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._release = asyncio.Event()

    async def invoke(
        self,
        tool_name: ToolName,
        *,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
        context: ToolInvocationContext | None = None,
    ) -> ToolInvocationOutput:
        _ = args, kwargs, context
        assert tool_name == "llm_chat"
        self.started.set()
        await self._release.wait()
        response = LlmResponse(
            id="late-completion",
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
            actual_cost_usd=0.0042,
            actual_cost_provider="chutes",
        )

    def release(self) -> None:
        self._release.set()


def _build_search_executor(
    *,
    now: datetime,
    expires_at: datetime,
    payload: JsonObject,
    provider_credential_source: ProviderCredentialSource = ProviderCredentialSource.MINER,
) -> tuple[ToolExecutor, InMemoryReceiptLog, Session, str]:
    token = uuid4().hex
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=now - timedelta(hours=1),
        expires_at=expires_at,
        budget_usd=1.0,
        provider_credential_source=provider_credential_source,
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    receipts = InMemoryReceiptLog()
    tokens = InMemoryTokenRegistry()
    tokens.register(session.session_id, token)
    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=receipts,
        usage_tracker=UsageTracker(),
        tool_invoker=StaticSearchInvoker(payload),
        token_registry=tokens,
        clock=lambda: now,
    )
    return executor, receipts, session, token


async def test_tool_executor_rejects_an_expired_session() -> None:
    """Future failure: a miner must not invoke paid tools after its session expires."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    executor, _, session, token = _build_search_executor(
        now=now,
        expires_at=now - timedelta(seconds=1),
        payload={"data": []},
    )

    with pytest.raises(RuntimeError, match="expired at"):
        await executor.execute(
            ToolInvocationRequest(
                session_id=session.session_id,
                token=token,
                tool="search_web",
            )
        )


async def test_tool_executor_rejects_invalid_token_without_recording_receipt() -> None:
    """Future failure: an unauthorized call pollutes another session's audit trail."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    executor, receipts, session, _ = _build_search_executor(
        now=now,
        expires_at=now + timedelta(hours=1),
        payload={"data": []},
    )

    with pytest.raises(PermissionError):
        await executor.execute(
            ToolInvocationRequest(
                session_id=session.session_id,
                token=uuid4().hex,
                tool="search_web",
            )
        )

    assert tuple(receipts.for_session(session.session_id)) == ()


async def test_platform_provider_success_log_omits_response_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Future failure: Platform-funded provider content must never enter INFO logs."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    response_sentinel = "platform-provider-response-sentinel"
    payload: JsonObject = {"data": [{"snippet": response_sentinel}]}
    executor, receipts, session, token = _build_search_executor(
        now=now,
        expires_at=now + timedelta(hours=1),
        payload=payload,
        provider_credential_source=ProviderCredentialSource.PLATFORM,
    )

    with caplog.at_level(logging.INFO, logger="harnyx_commons.tools"):
        result = await executor.execute(
            ToolInvocationRequest(
                session_id=session.session_id,
                token=token,
                tool="search_web",
            )
        )

    assert result.response_payload == payload
    assert tuple(receipts.for_session(session.session_id))[0].details.response_payload == payload
    completed = next(record for record in caplog.records if record.message == "tool call completed")
    assert not hasattr(completed, "response_preview")
    assert not hasattr(completed, "results_preview")
    assert response_sentinel not in "\n".join(str(record.__dict__) for record in caplog.records)


@pytest.mark.parametrize(
    ("actual_cost_usd", "error_match"),
    [
        pytest.param(float("nan"), "actual_cost_usd must be finite", id="nonfinite"),
        pytest.param(cast(float, True), "actual_cost_usd must be numeric", id="boolean"),
    ],
)
async def test_tool_executor_rejects_invalid_provider_cost_before_settling_usage(
    actual_cost_usd: float,
    error_match: str,
) -> None:
    """Future failure: invalid provider costs must not corrupt session settlement."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    token = uuid4().hex
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=now,
        expires_at=now + timedelta(hours=1),
        budget_usd=1.0,
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    receipts = InMemoryReceiptLog()
    tokens = InMemoryTokenRegistry()
    tokens.register(session.session_id, token)
    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=receipts,
        usage_tracker=UsageTracker(),
        tool_invoker=StaticSearchInvoker({"data": []}, actual_cost_usd=actual_cost_usd),
        token_registry=tokens,
        clock=lambda: now,
    )

    with pytest.raises(ValueError, match=error_match):
        await executor.execute(
            ToolInvocationRequest(session_id=session.session_id, token=token, tool="search_web")
        )

    stored_session = sessions.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)
    stored_receipts = tuple(receipts.for_session(session.session_id))
    assert len(stored_receipts) == 1
    assert stored_receipts[0].outcome is ToolCallOutcome.INTERNAL_ERROR


@pytest.mark.parametrize(
    ("tool_name", "payload", "expected"),
    [
        pytest.param(
            "search_web",
            {"data": [{"link": "https://example.com/search", "snippet": "Search note", "title": "Search"}]},
            ((0, "https://example.com/search", "Search note", "Search"),),
            id="search-web",
        ),
        pytest.param(
            "fetch_page",
            {"data": [{"url": "https://example.com/page", "content": "Page note", "title": "Page"}]},
            ((0, "https://example.com/page", "Page note", "Page"),),
            id="fetch-page",
        ),
        pytest.param(
            "search_ai",
            {"data": [{"url": "https://example.com/answer", "note": "AI note", "title": "Answer"}]},
            ((0, "https://example.com/answer", "AI note", "Answer"),),
            id="search-ai",
        ),
    ],
)
async def test_search_tool_results_preserve_referenceable_fields(
    tool_name: SearchToolName,
    payload: JsonObject,
    expected: tuple[tuple[int, str, str | None, str | None], ...],
) -> None:
    """Future failure: successful search payloads must remain usable as miner citations."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    token = uuid4().hex
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=now,
        expires_at=now + timedelta(hours=1),
        budget_usd=1.0,
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    tokens = InMemoryTokenRegistry()
    tokens.register(session.session_id, token)
    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=InMemoryReceiptLog(),
        usage_tracker=UsageTracker(),
        tool_invoker=StaticSearchInvoker(payload),
        token_registry=tokens,
        clock=lambda: now,
    )

    result = await executor.execute(
        ToolInvocationRequest(
            session_id=session.session_id,
            token=token,
            tool=tool_name,
        )
    )

    assert result.receipt.details.result_policy is ToolResultPolicy.REFERENCEABLE
    assert all(isinstance(item, SearchToolResult) for item in result.receipt.details.results)
    search_results = cast(tuple[SearchToolResult, ...], result.receipt.details.results)
    assert tuple((item.index, item.url, item.note, item.title) for item in search_results) == expected
    assert len({item.result_id for item in search_results}) == len(search_results)


async def test_tool_executor_rejects_late_completion_after_pending_receipt_is_abandoned() -> None:
    now = datetime(2026, 8, 21, tzinfo=UTC)
    token = uuid4().hex
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=now,
        expires_at=now + timedelta(hours=1),
        budget_usd=1.0,
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    receipts = InMemoryReceiptLog()
    tokens = InMemoryTokenRegistry()
    tokens.register(session.session_id, token)
    invoker = BlockingLlmInvoker()
    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=receipts,
        usage_tracker=UsageTracker(),
        tool_invoker=invoker,
        token_registry=tokens,
        clock=lambda: now,
    )
    request = ToolInvocationRequest(
        session_id=session.session_id,
        token=token,
        tool="llm_chat",
        kwargs={
            "provider": "chutes",
            "model": "zai-org/GLM-5-TEE",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    execution = asyncio.create_task(executor.execute(request))
    await invoker.started.wait()
    receipts.clear_session(session.session_id)
    invoker.release()

    with pytest.raises(RuntimeError, match="pending receipt was abandoned"):
        await execution

    assert tuple(receipts.for_session(session.session_id)) == ()
    stored_session = sessions.get(session.session_id)
    assert stored_session is not None
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0)
    assert stored_session.usage.llm_usage_totals == {}


async def test_tool_executor_settles_completed_call_after_session_is_exhausted() -> None:
    """Future failure: a paid sibling that already completed must settle after session exhaustion."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    token = uuid4().hex
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=now,
        expires_at=now + timedelta(hours=1),
        budget_usd=0.0001,
        hard_limit_usd=0.00015,
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    receipts = InMemoryReceiptLog()
    tokens = InMemoryTokenRegistry()
    tokens.register(session.session_id, token)

    class ExhaustedSiblingSearchInvoker:
        async def invoke(
            self,
            tool_name: ToolName,
            *,
            args: Sequence[JsonValue],
            kwargs: Mapping[str, JsonValue],
            context: ToolInvocationContext | None = None,
        ) -> ToolInvocationOutput:
            _ = tool_name, args, kwargs, context
            stored_session = sessions.get(session.session_id)
            assert stored_session is not None
            sessions.update(stored_session.mark_exhausted())
            return ToolInvocationOutput(
                public_payload={"data": []},
                actual_cost_usd=0.0001,
                actual_cost_provider="test-provider",
            )

    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=receipts,
        usage_tracker=UsageTracker(),
        tool_invoker=ExhaustedSiblingSearchInvoker(),
        token_registry=tokens,
        clock=lambda: now,
    )

    result = await executor.execute(
        ToolInvocationRequest(session_id=session.session_id, token=token, tool="search_web")
    )

    stored_session = sessions.get(session.session_id)
    assert stored_session is not None
    assert stored_session.status is SessionStatus.EXHAUSTED
    assert stored_session.usage.total_cost_usd == pytest.approx(0.0001)
    assert receipts.lookup(result.receipt.receipt_id) is not None
