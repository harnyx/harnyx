"""One hosted LLM call: bounded output, delivery deadline and protected cleanup."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence

import anyio
from pydantic import JsonValue

from harnyx_commons.tools.dto import ToolInvocationRequest, tool_payload_from_args_kwargs
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.http_serialization import serialize_tool_execute_response
from harnyx_commons.tools.runtime_invoker import effective_tool_timeout_seconds
from harnyx_commons.tools.token_semaphore import ToolConcurrencyLimiter
from harnyx_commons.tools.types import ToolName
from harnyx_miner_sdk.tools.llm_chat_models import LlmChatRequest
from harnyx_miner_sdk.tools.llm_chat_stream_models import (
    LlmChatChunk,
    LlmChatCompletedFrame,
    LlmChatErrorFrame,
    LlmChatFrame,
    LlmChatOutputFrame,
    LlmChatOutputSink,
)


def requests_llm_chat_stream(tool: ToolName, args: Sequence[JsonValue], kwargs: Mapping[str, JsonValue]) -> bool:
    if tool != "llm_chat":
        return False
    payload = tool_payload_from_args_kwargs(args, kwargs)
    if "stream" not in payload:
        return False
    return LlmChatRequest.model_validate(payload).stream


class LlmChatExecutionStream:
    """The HTTP adapter supervises its entire response with deliver(), including sends."""

    def __init__(
        self,
        execute: Callable[[LlmChatOutputSink], Awaitable[JsonValue]],
        *,
        total: float,
        error_frame: Callable[[Exception], LlmChatErrorFrame],
    ) -> None:
        self._started = asyncio.get_running_loop().time()
        self._deadline = self._started + total
        self._deadline_changed = asyncio.Event()
        self._queue: asyncio.Queue[LlmChatFrame] = asyncio.Queue(maxsize=8)
        self._execute = execute
        self._error_frame = error_frame
        self._producer: asyncio.Task[None] | None = None
        self._delivery_closed = False

    def set_total(self, total: float) -> None:
        """Apply the execution owner's clamp without restarting the admission clock."""
        self._deadline = min(self._deadline, self._started + total)
        self._deadline_changed.set()

    async def _output(self, chunk: LlmChatChunk) -> None:
        if self._delivery_closed:
            raise asyncio.CancelledError
        await self._queue.put(LlmChatOutputFrame(chunk=chunk))

    async def _produce(self) -> None:
        try:
            result = await self._execute(self._output)
        except Exception as exc:
            if not self._delivery_closed:
                await self._queue.put(self._error_frame(exc))
        else:
            if not self._delivery_closed:
                await self._queue.put(LlmChatCompletedFrame(result=result))

    async def body(self) -> AsyncIterator[str]:
        while True:
            frame = await self._queue.get()
            yield f"event: {frame.type}\ndata: {frame.model_dump_json(exclude_none=True)}\n\n"
            if not isinstance(frame, LlmChatOutputFrame):
                return

    async def deliver(self, operation: Callable[[], Awaitable[None]]) -> None:
        self._producer = asyncio.create_task(self._produce())
        delivery = asyncio.ensure_future(operation())
        changed = asyncio.create_task(self._deadline_changed.wait())
        try:
            while True:
                remaining = self._deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("llm_chat total timeout during delivery")
                done, _ = await asyncio.wait(
                    {delivery, changed}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
                if delivery in done:
                    delivery.result()
                    return
                if changed in done:
                    self._deadline_changed.clear()
                    changed = asyncio.create_task(self._deadline_changed.wait())
        finally:
            self._delivery_closed = True
            # Cancel execution once. Its existing cancellation path owns accounting;
            # subsequent response/disconnect cancellation must not interrupt it.
            for task in (delivery, changed, self._producer):
                if not task.done() and not task.cancelling():
                    task.cancel()
            with anyio.CancelScope(shield=True):
                cleanup = asyncio.gather(delivery, changed, self._producer, return_exceptions=True)
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()


def open_tool_stream(
    executor: ToolExecutor,
    limiter: ToolConcurrencyLimiter,
    invocation: ToolInvocationRequest,
    *,
    error_frame: Callable[[Exception], LlmChatErrorFrame],
) -> LlmChatExecutionStream:
    executor.validate_request(invocation)

    async def execute(on_output: LlmChatOutputSink) -> JsonValue:
        await limiter.acquire_async(invocation)
        try:
            result = await executor.execute(invocation, on_output=on_output)
            return serialize_tool_execute_response(result).model_dump(mode="json")
        finally:
            limiter.release(invocation)

    return LlmChatExecutionStream(
        execute,
        total=effective_tool_timeout_seconds(invocation.tool, args=invocation.args, kwargs=invocation.kwargs),
        error_frame=error_frame,
    )
