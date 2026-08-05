from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolResultPolicy
from harnyx_commons.errors import ConcurrencyLimitError
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


def test_default_limit_allows_twenty_mixed_calls_and_blocks_twenty_first() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = _mixed_invocations(20)

    for invocation in held:
        limiter.acquire(invocation)
    try:
        with pytest.raises(ConcurrencyLimitError):
            limiter.acquire(_invocation(TEST_TOKEN, "llm_chat", model="openrouter/native-alias"))
    finally:
        for invocation in held:
            limiter.release(invocation)


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
