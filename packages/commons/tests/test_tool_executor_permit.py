from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolResultPolicy
from harnyx_commons.tools.dto import ToolBudgetSnapshot, ToolInvocationRequest, ToolInvocationResult
from harnyx_commons.tools.executor import execute_tool_with_concurrency_permit
from harnyx_commons.tools.token_semaphore import (
    DEFAULT_TOOL_CONCURRENCY_LIMITS,
    ToolConcurrencyLimiter,
    ToolConcurrencyLimits,
    max_parallel_provider_tool_calls,
)
from harnyx_commons.tools.types import ToolName

pytestmark = pytest.mark.anyio("asyncio")
TEST_TOKEN = "token"  # noqa: S105
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"


class RecordingExecutor:
    def __init__(self, result: ToolInvocationResult) -> None:
        self.result = result
        self.invocations: list[ToolInvocationRequest] = []

    async def execute(self, invocation: ToolInvocationRequest) -> ToolInvocationResult:
        self.invocations.append(invocation)
        return self.result


class FailingExecutor:
    async def execute(self, _: ToolInvocationRequest) -> ToolInvocationResult:
        raise RuntimeError("expected failure")


def _invocation(token: str, tool: ToolName = "search_web", *, model: str = DEFAULT_LLM_MODEL) -> ToolInvocationRequest:
    kwargs = {"model": model, "messages": [{"role": "user", "content": "demo"}]} if tool == "llm_chat" else {}
    return ToolInvocationRequest(
        session_id=uuid4(),
        token=token,
        tool=tool,
        args=(),
        kwargs=kwargs,
    )


def _mixed_invocations(count: int, *, token: str = TEST_TOKEN) -> list[ToolInvocationRequest]:
    tools: tuple[ToolName, ...] = ("search_web", "fetch_page", "tooling_info", "test_tool", "llm_chat")
    return [_invocation(token, tools[index % len(tools)], model=f"model-{index}") for index in range(count)]


def _result(session_id, tool: ToolName = "search_web") -> ToolInvocationResult:
    return ToolInvocationResult(
        receipt=ToolCall(
            receipt_id="receipt-1",
            session_id=session_id,
            uid=7,
            tool=tool,
            issued_at=datetime(2026, 5, 7, tzinfo=UTC),
            outcome=ToolCallOutcome.OK,
            details=ToolCallDetails(
                request_hash="request-hash",
                response_hash="response-hash",
                response_payload={"data": []},
                result_policy=ToolResultPolicy.REFERENCEABLE,
            ),
        ),
        response_payload={"data": []},
        budget=ToolBudgetSnapshot(
            session_budget_usd=1.0,
            session_hard_limit_usd=1.0,
            session_used_budget_usd=0.0,
            session_remaining_budget_usd=1.0,
        ),
    )


async def test_execute_tool_with_concurrency_permit_waits_for_released_token_permit() -> None:
    invocation = _invocation(TEST_TOKEN, "llm_chat")
    expected = _result(invocation.session_id, invocation.tool)
    executor = RecordingExecutor(expected)
    limiter = ToolConcurrencyLimiter(ToolConcurrencyLimits(max_parallel_calls=1))

    limiter.acquire(invocation)
    waiter = asyncio.create_task(execute_tool_with_concurrency_permit(executor, limiter, invocation))
    await asyncio.sleep(0.05)

    assert not waiter.done()
    assert executor.invocations == []

    limiter.release(invocation)
    result = await asyncio.wait_for(waiter, timeout=1.0)

    assert result == expected
    assert executor.invocations == [invocation]
    assert limiter.in_flight(invocation) == 0


async def test_execute_tool_with_concurrency_permit_releases_token_permit_after_executor_failure() -> None:
    invocation = _invocation(TEST_TOKEN, "llm_chat")
    limiter = ToolConcurrencyLimiter(ToolConcurrencyLimits(max_parallel_calls=2))

    with pytest.raises(RuntimeError, match="expected failure"):
        await execute_tool_with_concurrency_permit(FailingExecutor(), limiter, invocation)

    assert limiter.in_flight(invocation) == 0


async def test_all_tool_calls_share_one_token_cap() -> None:
    llm_invocation = _invocation(TEST_TOKEN, "llm_chat")
    search_invocation = _invocation(TEST_TOKEN, "search_web")
    waiter_invocation = _invocation(TEST_TOKEN, "fetch_page")
    expected = _result(waiter_invocation.session_id, waiter_invocation.tool)
    executor = RecordingExecutor(expected)
    limiter = ToolConcurrencyLimiter(ToolConcurrencyLimits(max_parallel_calls=2))

    limiter.acquire(llm_invocation)
    limiter.acquire(search_invocation)
    waiter = asyncio.create_task(execute_tool_with_concurrency_permit(executor, limiter, waiter_invocation))
    await asyncio.sleep(0.05)

    assert not waiter.done()
    assert executor.invocations == []

    limiter.release(search_invocation)
    result = await asyncio.wait_for(waiter, timeout=1.0)

    assert result == expected
    assert executor.invocations == [waiter_invocation]
    limiter.release(llm_invocation)
    assert limiter.in_flight(waiter_invocation) == 0


async def test_default_limits_wait_on_twenty_first_mixed_call_until_release() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = _mixed_invocations(20)
    waiter_invocation = _invocation(TEST_TOKEN, "search_web")
    expected = _result(waiter_invocation.session_id, waiter_invocation.tool)
    executor = RecordingExecutor(expected)

    for invocation in held:
        limiter.acquire(invocation)
    try:
        waiter = asyncio.create_task(execute_tool_with_concurrency_permit(executor, limiter, waiter_invocation))
        await asyncio.sleep(0.05)
        assert not waiter.done()
        assert executor.invocations == []

        limiter.release(held.pop())
        result = await asyncio.wait_for(waiter, timeout=1.0)

        assert result == expected
        assert executor.invocations == [waiter_invocation]
    finally:
        for invocation in held:
            limiter.release(invocation)

    assert limiter.in_flight(waiter_invocation) == 0


def test_provider_tool_parallel_call_cap_matches_single_token_cap() -> None:
    assert max_parallel_provider_tool_calls(DEFAULT_TOOL_CONCURRENCY_LIMITS) == 20


def test_provider_tool_parallel_call_cap_respects_custom_limit() -> None:
    assert max_parallel_provider_tool_calls(ToolConcurrencyLimits(max_parallel_calls=7)) == 7


async def test_stream_deadline_includes_blocked_terminal_send() -> None:
    from harnyx_commons.tools.llm_chat_stream import LlmChatExecutionStream
    from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatErrorFrame

    committed = asyncio.Event()
    terminal_started = asyncio.Event()

    async def execute(_sink):
        committed.set()
        return {"receipt_id": "committed"}

    stream = LlmChatExecutionStream(execute, total=0.05, error_frame=lambda exc: LlmChatErrorFrame(detail=str(exc)))

    async def stalled_transport():
        async for frame in stream.body():
            assert '"type":"completed"' in frame
            terminal_started.set()
            await asyncio.Event().wait()

    with pytest.raises(TimeoutError, match="total timeout"):
        await stream.deliver(stalled_transport)
    assert committed.is_set()
    assert terminal_started.is_set()


async def test_stream_disconnect_holds_permit_until_repeated_cancel_safe_finalization() -> None:
    from harnyx_commons.tools.llm_chat_stream import open_tool_stream
    from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatChunk, LlmChatErrorFrame, TextUpdate

    invocation = _invocation(TEST_TOKEN, "llm_chat")
    limiter = ToolConcurrencyLimiter(ToolConcurrencyLimits(max_parallel_calls=1))
    settling = asyncio.Event()
    release_settlement = asyncio.Event()
    delivered = asyncio.Event()
    settlements = []

    class Executor:
        def validate_request(self, request):
            pass

        async def execute(self, request, *, on_output):
            try:
                await on_output(LlmChatChunk(choice_index=0, text=TextUpdate(operation="append", value="first")))
                await asyncio.Event().wait()
            finally:
                settling.set()
                await release_settlement.wait()
                settlements.append("settled")

    stream = open_tool_stream(
        Executor(), limiter, invocation, error_frame=lambda exc: LlmChatErrorFrame(detail=str(exc))
    )

    async def transport():
        async for _ in stream.body():
            delivered.set()

    owner = asyncio.create_task(stream.deliver(transport))
    await asyncio.wait_for(delivered.wait(), 1)
    assert limiter.in_flight(invocation) == 1
    owner.cancel()
    await asyncio.wait_for(settling.wait(), 1)
    owner.cancel()
    await asyncio.sleep(0)
    assert limiter.in_flight(invocation) == 1
    release_settlement.set()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert settlements == ["settled"]
    assert limiter.in_flight(invocation) == 0


async def test_stream_bounded_backpressure_times_out_without_consumer_reads() -> None:
    from harnyx_commons.tools.llm_chat_stream import LlmChatExecutionStream
    from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatChunk, LlmChatErrorFrame, TextUpdate

    produced = []
    cleaned = asyncio.Event()

    async def execute(sink):
        try:
            for index in range(10_000):
                await sink(LlmChatChunk(choice_index=0, text=TextUpdate(operation="append", value=str(index))))
                produced.append(index)
            return {}
        finally:
            cleaned.set()

    stream = LlmChatExecutionStream(execute, total=0.05, error_frame=lambda exc: LlmChatErrorFrame(detail=str(exc)))
    with pytest.raises(TimeoutError):
        await stream.deliver(asyncio.Event().wait)
    assert 0 < len(produced) < 10_000
    assert cleaned.is_set()


@pytest.mark.parametrize("cleanup_result", ["return", "raise"])
async def test_stream_cleanup_never_publishes_into_full_abandoned_queue(cleanup_result) -> None:
    """Cancellation settlement can return or fail; neither may wait on a disconnected consumer."""
    from harnyx_commons.tools.llm_chat_stream import LlmChatExecutionStream
    from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatChunk, LlmChatErrorFrame, TextUpdate

    full, settling, release, settled = (asyncio.Event() for _ in range(4))

    async def execute(sink):
        try:
            for _ in range(8):
                await sink(LlmChatChunk(choice_index=0, text=TextUpdate(operation="append", value="output")))
            full.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            settling.set()
            await release.wait()
            settled.set()
            if cleanup_result == "raise":
                raise RuntimeError("settlement failed") from None
            return {}

    stream = LlmChatExecutionStream(execute, total=2, error_frame=lambda exc: LlmChatErrorFrame(detail=str(exc)))
    owner = asyncio.create_task(stream.deliver(asyncio.Event().wait))
    await asyncio.wait_for(full.wait(), 1)
    owner.cancel()
    await asyncio.wait_for(settling.wait(), 1)
    owner.cancel()
    await asyncio.sleep(0)
    assert not owner.done()
    release.set()
    done, _ = await asyncio.wait({owner}, timeout=1)
    # Stop a broken producer explicitly so this regression cannot hang pytest cleanup.
    if not done:
        stream._producer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert done and settled.is_set()
