from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.domain.session import ProviderCredentialSource, Session, SessionStatus
from harnyx_commons.domain.tool_call import (
    SearchToolResult,
    StartedToolCall,
    ToolCall,
    ToolCallOutcome,
    ToolResultPolicy,
)
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmMessageContentPart, LlmResponse, LlmUsage
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolCallRecorder, ToolExecutor, ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker
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
    recorder: ToolCallRecorder | None = None,
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
        tool_call_recorder=recorder,
    )
    return executor, receipts, session, token


class RecordingEvidence:
    def __init__(self) -> None:
        self.starts: list[StartedToolCall] = []
        self.terminals: list[ToolCall] = []
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()
        self.release_start.set()
        self.fail_start = False
        self.fail_terminal = False

    async def record_started(self, session: Session, call: StartedToolCall, *, on_recorded: Callable[[], None]) -> None:
        self.start_entered.set()
        await self.release_start.wait()
        if self.fail_start:
            raise RuntimeError("start storage failed")
        self.starts.append(call)
        on_recorded()

    async def record_terminal(self, session: Session, call: ToolCall) -> None:
        self.terminals.append(call)
        if self.fail_terminal:
            raise RuntimeError("terminal acknowledgement lost")


@pytest.mark.parametrize("rejection", ["expiry", "revocation", "budget"])
async def test_durable_start_revalidates_admission_before_dispatch(rejection: str) -> None:
    """A delayed storage write must not permit a paid call after admission ends."""
    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    recorder.release_start.clear()
    executor, _, session, token = _build_search_executor(
        now=now,
        expires_at=now + timedelta(minutes=1),
        payload={"data": []},
        recorder=recorder,
    )
    invoked = False

    async def provider(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("provider must not run")

    executor._tool_invoker.invoke = provider
    work = asyncio.create_task(
        executor.execute(
            ToolInvocationRequest(
                session_id=session.session_id,
                token=token,
                tool="search_web",
                args=(),
                kwargs={"query": "q"},
            )
        )
    )
    await recorder.start_entered.wait()
    if rejection == "expiry":
        executor._clock = lambda: now + timedelta(minutes=2)
    elif rejection == "revocation":
        executor._tokens.revoke(session.session_id)
    else:
        executor._sessions.mutate(session.session_id, lambda current: current.mark_exhausted())
    recorder.release_start.set()
    with pytest.raises((RuntimeError, PermissionError)):
        await work
    assert not invoked
    assert len(recorder.terminals) == 1
    assert recorder.terminals[0].details.actual_cost_usd == 0


async def test_malformed_decision_payload_finishes_started_receipt_with_unattributed_zero() -> None:
    """Malformed arguments must not strand a started call or invent provider charges during error recording."""
    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    executor, receipts, session, token = _build_search_executor(
        now=now,
        expires_at=now + timedelta(minutes=1),
        payload={},
        recorder=recorder,
    )
    executor._tool_invoker = RuntimeToolInvoker(receipts)
    with pytest.raises(ValueError, match="expected JSON object payload as first positional argument"):
        await executor.execute(ToolInvocationRequest(session.session_id, token, "decision_query", ("invalid",), {}))

    assert len(recorder.starts) == 1
    assert len(recorder.terminals) == 1
    receipt = recorder.terminals[0]
    assert receipt.receipt_id == recorder.starts[0].receipt_id
    assert receipts.for_session(session.session_id) == (receipt,)
    assert receipt.outcome is ToolCallOutcome.INTERNAL_ERROR
    assert receipt.details.actual_cost_usd == 0.0
    assert receipt.details.actual_cost_provider is None
    updated = executor._sessions.get(session.session_id)
    assert updated is not None
    assert updated.usage.total_cost_usd == 0.0
    assert updated.usage.actual_total_cost_usd == 0.0
    assert updated.usage.cost_by_provider == {}
    assert updated.usage.actual_cost_by_provider == {}


async def test_start_storage_failure_prevents_paid_dispatch() -> None:
    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    recorder.fail_start = True
    executor, _, session, token = _build_search_executor(
        now=now,
        expires_at=now + timedelta(minutes=1),
        payload={"data": []},
        recorder=recorder,
    )
    with pytest.raises(RuntimeError, match="start storage failed"):
        await executor.execute(ToolInvocationRequest(session.session_id, token, "search_web", (), {"query": "q"}))
    assert recorder.terminals == []
    assert executor._sessions.get(session.session_id).status is SessionStatus.ERROR


async def test_charged_parse_failure_retains_cost_and_original_local_failure() -> None:
    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    executor, receipts, session, token = _build_search_executor(
        now=now,
        expires_at=now + timedelta(minutes=1),
        payload={},
        recorder=recorder,
    )
    with pytest.raises(ValueError):
        await executor.execute(
            ToolInvocationRequest(session.session_id, token, "llm_chat", (), {"provider": "test", "model": "model"})
        )
    assert len(recorder.terminals) == 1
    assert recorder.terminals[0].details.actual_cost_usd == 0.001
    assert recorder.terminals[0].outcome is ToolCallOutcome.INTERNAL_ERROR
    assert receipts.lookup(recorder.terminals[0].receipt_id).details.cost_usd is None


async def test_ambiguous_terminal_commit_never_replaces_evidence() -> None:
    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    recorder.fail_terminal = True
    executor, _, session, token = _build_search_executor(
        now=now,
        expires_at=now + timedelta(minutes=1),
        payload={"data": []},
        recorder=recorder,
    )
    with pytest.raises(RuntimeError, match="terminal acknowledgement lost"):
        await executor.execute(ToolInvocationRequest(session.session_id, token, "search_web", (), {"query": "q"}))
    assert len(recorder.starts) == len(recorder.terminals) == 1
    assert recorder.terminals[0].outcome is ToolCallOutcome.OK
    assert recorder.terminals[0].details.cost_usd == 0.001


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
        await executor.execute(ToolInvocationRequest(session_id=session.session_id, token=token, tool="search_web"))

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
            "fetch_page",
            {
                "data": [
                    {"url": "https://example.com/page", "content": "Markdown evidence", "title": "Page"},
                    {"url": "https://example.com/page", "content": '{"json": []}', "title": "Page"},
                ]
            },
            (
                (0, "https://example.com/page", "Markdown evidence", "Page"),
                (1, "https://example.com/page", '{"json": []}', "Page"),
            ),
            id="fetch-same-url-two-representations",
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


@pytest.mark.parametrize("dispatched", [False, True])
async def test_opt_in_provider_failure_settles_zero_or_unknown_once(dispatched) -> None:
    from harnyx_commons.errors import ToolProviderError

    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    executor, receipts, session, token = _build_search_executor(
        now=now, expires_at=now + timedelta(minutes=1), payload={}, recorder=recorder
    )
    executor._account_provider_failures = True

    async def failed(tool_name, *, args, kwargs, context):
        if dispatched:
            context.on_provider_dispatch()
        raise ToolProviderError("failed", provider="openai")

    executor._tool_invoker.invoke = failed
    with pytest.raises(ToolProviderError):
        await executor.execute(
            ToolInvocationRequest(
                session.session_id, token, "llm_chat", (), {"provider": "openai", "model": "gpt-6-luna"}
            )
        )
    recorded = receipts.for_session(session.session_id)
    assert len(recorded) == len(recorder.terminals) == 1
    assert recorded[0] == recorder.terminals[0]
    expected = None if dispatched else 0.0
    assert recorded[0].details.actual_cost_usd == expected
    assert executor._sessions.get(session.session_id).usage.actual_total_cost_usd == expected


async def test_opt_in_failure_after_returned_output_retains_payload_tokens_and_cost_once() -> None:
    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    response = LlmResponse(
        id="charged",
        choices=(),
        usage=LlmUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=17, prompt_cached_tokens=3, reasoning_tokens=2
        ),
        metadata={"provider_status": "incomplete"},
    )
    executor, receipts, session, token = _build_search_executor(
        now=now, expires_at=now + timedelta(minutes=1), payload=response.to_payload(), recorder=recorder
    )
    executor._account_provider_failures = True
    original_invoke = executor._tool_invoker.invoke

    async def invoke(tool_name, *, args, kwargs, context):
        context.on_provider_dispatch()
        return await original_invoke(tool_name, args=args, kwargs=kwargs, context=context)

    executor._tool_invoker.invoke = invoke

    def fail_local_parse(*args):
        raise ValueError("local failure after provider result")

    executor._build_results = fail_local_parse
    with pytest.raises(ValueError):
        await executor.execute(
            ToolInvocationRequest(
                session.session_id, token, "llm_chat", (), {"provider": "openai", "model": "gpt-6-luna"}
            )
        )
    updated = executor._sessions.get(session.session_id)
    assert updated.usage.total_cost_usd == pytest.approx(0.001)
    assert updated.usage.llm_usage_totals["openai"]["gpt-6-luna"].total_tokens == 17
    retained = receipts.for_session(session.session_id)[0].details.response_payload
    assert retained["metadata"] == {"provider_status": "incomplete"}
    assert retained["id"] == "charged"
    assert retained["choices"] == []
    assert retained["usage"] == response.to_payload()["usage"]
    assert len(recorder.terminals) == 1
    assert receipts.for_session(session.session_id) == tuple(recorder.terminals)


@pytest.mark.parametrize("stage", ["success_terminal", "failure_terminal", "start"])
async def test_opt_in_storage_failure_retains_paid_cost_without_reactivating_session(stage) -> None:
    """Future failure: receipt storage errors erase a paid call or permit another dispatch."""
    from harnyx_commons.errors import ToolProviderError

    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    recorder.fail_start = stage == "start"
    recorder.fail_terminal = stage != "start"
    response = LlmResponse(
        id="charged", choices=(), usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    executor, receipts, session, token = _build_search_executor(
        now=now, expires_at=now + timedelta(minutes=1), payload=response.to_payload(), recorder=recorder
    )
    executor._account_provider_failures = True
    original = executor._tool_invoker.invoke

    async def invoke(tool_name, *, args, kwargs, context):
        context.on_provider_dispatch()
        if stage == "failure_terminal":
            raise ToolProviderError("provider failed", provider="openai")
        return await original(tool_name, args=args, kwargs=kwargs, context=context)

    executor._tool_invoker.invoke = invoke
    with pytest.raises(RuntimeError, match="storage failed|acknowledgement lost"):
        await executor.execute(
            ToolInvocationRequest(
                session.session_id, token, "llm_chat", (), {"provider": "openai", "model": "gpt-6-luna"}
            )
        )
    updated = executor._sessions.get(session.session_id)
    assert updated.status is SessionStatus.ERROR
    expected = 0.0 if stage == "start" else None if stage == "failure_terminal" else 0.001
    assert updated.usage.actual_total_cost_usd == expected
    assert len(receipts.for_session(session.session_id)) == 1
    assert len(recorder.terminals) == (0 if stage == "start" else 1)
    if stage == "success_terminal":
        assert updated.usage.llm_usage_totals["openai"]["gpt-6-luna"].total_tokens == 15


async def test_opt_in_typed_failure_billing_keeps_available_tokens() -> None:
    from harnyx_commons.errors import ToolProviderError
    from harnyx_commons.tools.decision_models import DecisionUsage
    from harnyx_commons.tools.provider_billing import ProviderBillingMetadata

    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    executor, receipts, session, token = _build_search_executor(
        now=now, expires_at=now + timedelta(minutes=1), payload={}, recorder=recorder
    )
    executor._account_provider_failures = True

    async def invoke(tool_name, *, args, kwargs, context):
        context.on_provider_dispatch()
        raise ToolProviderError(
            "failed with usage",
            provider="openai",
            billing=ProviderBillingMetadata(
                actual_cost_provider="openai",
                source="response_body",
                actual_cost_usd=0.003,
                usage=DecisionUsage(input_tokens=10, output_tokens=5),
            ),
        )

    executor._tool_invoker.invoke = invoke
    with pytest.raises(ToolProviderError):
        await executor.execute(
            ToolInvocationRequest(
                session.session_id, token, "llm_chat", (), {"provider": "openai", "model": "gpt-6-luna"}
            )
        )
    usage = executor._sessions.get(session.session_id).usage
    assert usage.total_cost_usd == pytest.approx(0.003)
    assert usage.llm_usage_totals["openai"]["gpt-6-luna"].total_tokens == 15
    assert receipts.for_session(session.session_id) == tuple(recorder.terminals)


@pytest.mark.parametrize("confirmed_rejection", [True, False])
async def test_deepseek_failure_receipt_distinguishes_zero_from_unknown(confirmed_rejection):
    from harnyx_commons.errors import ToolProviderError
    from harnyx_commons.tools.provider_billing import ProviderBillingMetadata

    now = datetime.now(UTC)
    recorder = RecordingEvidence()
    executor, receipts, session, token = _build_search_executor(
        now=now,
        expires_at=now + timedelta(minutes=1),
        payload={},
        recorder=recorder,
    )
    executor._account_provider_failures = True

    async def invoke(tool_name, *, args, kwargs, context):
        context.on_provider_dispatch()
        raise ToolProviderError(
            "rejected or interrupted",
            provider="deepseek",
            billing=ProviderBillingMetadata(
                actual_cost_provider="deepseek",
                actual_cost_usd=0.0,
                source="response_headers",
            )
            if confirmed_rejection
            else None,
        )

    executor._tool_invoker.invoke = invoke
    with pytest.raises(ToolProviderError):
        await executor.execute(
            ToolInvocationRequest(
                session.session_id,
                token,
                "search_web",
                (),
                {"provider": "deepseek", "search_queries": ["query"]},
            )
        )
    receipt = receipts.for_session(session.session_id)[0]
    assert receipt.details.actual_cost_usd == (0.0 if confirmed_rejection else None)
    assert receipt.details.actual_cost_provider == "deepseek"
    assert receipts.for_session(session.session_id) == tuple(recorder.terminals)


@pytest.mark.parametrize("provider,reference", [("openrouter", 0.008), ("openrouter", None), ("cloudflare", 0.004)])
@pytest.mark.parametrize("account_failures", [False, True])
async def test_billed_failure_preserves_charge_and_allows_miner_recovery(
    provider: str, reference: float | None, account_failures: bool,
) -> None:
    from harnyx_commons.errors import ToolProviderError
    from harnyx_commons.llm.schema import LlmUsage
    from harnyx_commons.tools.provider_billing import ProviderBillingMetadata
    from harnyx_commons.tools.receipt_accounting import usage_from_receipts

    now = datetime.now(UTC)
    executor, receipts, session, token = _build_search_executor(
        now=now, expires_at=now + timedelta(minutes=1), payload={}
    )

    executor._account_provider_failures = account_failures
    model = "inception/mercury-3" if provider == "openrouter" else "@cf/openai/gpt-oss-20b"

    async def invoke(tool_name, *, args, kwargs, context):
        context.on_provider_dispatch()
        raise ToolProviderError(
            "unpriceable" if reference is None else "rejected output",
            provider=provider,
            billing=ProviderBillingMetadata(
                actual_cost_provider=provider,
                source="response_body",
                actual_cost_usd=0.004,
                reference_cost_usd=reference,
                reference_cost_unavailable=reference is None,
                llm_usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=19, reasoning_tokens=4),
                provider_usage={"inference": "attempted"},
            ),
        )

    executor._tool_invoker.invoke = invoke
    request = ToolInvocationRequest(
        session.session_id, token, "llm_chat", (), {"provider": provider, "model": model}
    )
    with pytest.raises(ToolProviderError):
        await executor.execute(request)
    receipt = receipts.for_session(session.session_id)[0]
    assert receipt.details.cost_usd == reference
    assert receipt.details.actual_cost_usd == pytest.approx(0.004)
    updated = executor._sessions.get(session.session_id)
    assert updated.usage.total_cost_usd == pytest.approx(reference or 0)
    assert updated.usage.actual_total_cost_usd == pytest.approx(0.004)
    assert updated.usage == usage_from_receipts((receipt,))
    totals = updated.usage.llm_usage_totals[provider][model]
    assert (totals.total_tokens, totals.reasoning_tokens, totals.call_count) == (19, 4, 1)
    assert updated.status is SessionStatus.ACTIVE

    fallback_reference = 0.004 if provider == "openrouter" else 0.002

    async def fallback(tool_name, *, args, kwargs, context):
        return ToolInvocationOutput(
            public_payload=LlmResponse(
                id="miner-fallback",
                choices=(),
                usage=LlmUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            ).to_payload(),
            actual_cost_usd=0.002,
            reference_cost_usd=fallback_reference,
            actual_cost_provider=provider,
            actual_cost_evidence={"inference": "attempted"},
        )

    executor._tool_invoker.invoke = fallback
    await executor.execute(request)
    recovered = executor._sessions.get(session.session_id)
    assert recovered.status is SessionStatus.ACTIVE
    assert recovered.usage.total_cost_usd == pytest.approx((reference or 0) + fallback_reference)
    assert recovered.usage.actual_total_cost_usd == pytest.approx(0.006)
    assert len(receipts.for_session(session.session_id)) == 2


@pytest.mark.parametrize("account_failures", [False, True])
@pytest.mark.parametrize("inference", ["not_dispatched", "attempted"])
async def test_openrouter_zero_charge_failure_distinguishes_rejection_from_inference(
    account_failures: bool, inference: str
) -> None:
    from harnyx_commons.errors import ToolProviderError
    from harnyx_commons.tools.provider_billing import ProviderBillingMetadata
    from harnyx_commons.tools.receipt_accounting import usage_from_receipts

    now = datetime.now(UTC)
    executor, receipts, session, token = _build_search_executor(
        now=now, expires_at=now + timedelta(minutes=1), payload={}
    )
    executor._account_provider_failures = account_failures

    async def invoke(tool_name, *, args, kwargs, context):
        context.on_provider_dispatch()  # The validator contacted Platform, which rejected before inference.
        raise ToolProviderError(
            "rejected before inference",
            provider="openrouter",
            billing=ProviderBillingMetadata(
                actual_cost_provider="openrouter",
                source="response_body",
                actual_cost_usd=0.0,
                reference_cost_usd=0.0,
                provider_usage={"inference": inference},
            ),
        )

    executor._tool_invoker.invoke = invoke
    with pytest.raises(ToolProviderError):
        await executor.execute(
            ToolInvocationRequest(
                session.session_id, token, "llm_chat", (), {"provider": "openrouter", "model": "inception/mercury-3"}
            )
        )
    usage = executor._sessions.get(session.session_id).usage
    assert usage == usage_from_receipts(receipts.for_session(session.session_id))
    if inference == "not_dispatched":
        assert usage.llm_usage_totals == {}
    else:
        assert usage.llm_usage_totals["openrouter"]["inception/mercury-3"].call_count == 1
    assert usage.total_cost_usd == usage.actual_total_cost_usd == 0.0
    assert executor._sessions.get(session.session_id).status is SessionStatus.ACTIVE


async def test_openrouter_success_missing_reference_fails_and_retains_observed_charge() -> None:
    from harnyx_commons.errors import ToolProviderError

    now = datetime.now(UTC)
    executor, receipts, session, token = _build_search_executor(
        now=now, expires_at=now + timedelta(minutes=1), payload={}
    )

    async def invoke(tool_name, *, args, kwargs, context):
        context.on_provider_dispatch()
        return ToolInvocationOutput(
            public_payload={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            actual_cost_usd=0.004,
            actual_cost_provider="openrouter",
            actual_cost_evidence={
                "inference": "attempted",
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    executor._tool_invoker.invoke = invoke
    with pytest.raises(ToolProviderError, match="reference cost is missing"):
        await executor.execute(
            ToolInvocationRequest(
                session.session_id, token, "llm_chat", (), {"provider": "openrouter", "model": "inception/mercury-3"}
            )
        )
    receipt = receipts.for_session(session.session_id)[0]
    assert not receipt.is_successful()
    assert receipt.details.reference_cost_usd is None
    assert receipt.details.actual_cost_usd == pytest.approx(0.004)
    assert executor._sessions.get(session.session_id).status is SessionStatus.ACTIVE
