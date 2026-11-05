"""Streaming lifetimes on the existing private worker connection."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from harnyx_miner_sdk.streaming import LlmChatFrameStream
from harnyx_miner_sdk.tools.llm_chat_models import (
    DEFAULT_LLM_CHAT_TIMEOUT_SECONDS,
    LlmChatRequest,
    LlmChatTimeout,
)
from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatErrorFrame, LlmChatFrame, LlmChatOutputFrame
from harnyx_miner_sdk.tools.proxy import ToolInvocationError
from harnyx_sandbox.sandbox.worker_protocol import (
    StreamReply,
    ToolCall,
    ToolStreamClose,
    ToolStreamRead,
    WorkerMessageViolationError,
)


async def finish_cleanup(task: asyncio.Task[None]) -> None:
    """Join owned cleanup even if its caller is cancelled repeatedly."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
    task.result()
    if cancellation is not None:
        raise cancellation


class WorkerToolStream:
    """One worker's read credit and terminal; never block the shared receiver."""

    def __init__(
        self,
        call_id: int,
        send_control: Callable[[ToolStreamRead | ToolStreamClose], Awaitable[None]],
        release: Callable[[], None],
    ) -> None:
        self.call_id = call_id
        self._send_control = send_control
        self._release = release
        loop = asyncio.get_running_loop()
        self.opened: asyncio.Future[None] = loop.create_future()
        self._closed: asyncio.Future[None] = loop.create_future()
        self._read: asyncio.Future[LlmChatFrame] | None = None
        self._terminal: LlmChatFrame | None = None
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closing(self) -> bool:
        return self._close_task is not None

    def __aiter__(self) -> WorkerToolStream:
        return self

    async def __anext__(self) -> LlmChatFrame:
        if self._terminal is not None:
            frame, self._terminal = self._terminal, None
            return frame
        if self.closing:
            raise StopAsyncIteration
        if self._read is not None:
            raise RuntimeError("llm_chat stream supports a single consumer")
        self._read = asyncio.get_running_loop().create_future()
        try:
            await self._send_control(ToolStreamRead(call_id=self.call_id))
            return await self._read
        finally:
            self._read = None

    def receive(self, reply: StreamReply) -> None:
        if reply.kind == "open":
            if self.opened.done():
                raise WorkerMessageViolationError("duplicate stream opening")
            self.opened.set_result(None)
        elif reply.kind == "closed":
            if not self.closing or self._closed.done():
                raise WorkerMessageViolationError("unexpected stream close")
            self._closed.set_result(None)
            if self._read is not None and not self._read.done():
                self._read.set_exception(ToolInvocationError("llm_chat stream closed during consumption"))
        else:
            frame = reply.frame
            if frame is None:
                raise WorkerMessageViolationError("missing stream frame")
            terminal = not isinstance(frame, LlmChatOutputFrame)
            if not self.opened.done():
                if not isinstance(frame, LlmChatErrorFrame):
                    raise WorkerMessageViolationError("stream output before opening")
                self.opened.set_exception(ToolInvocationError(frame.detail))
            if self._read is not None and not self._read.done():
                self._read.set_result(frame)
            elif terminal:
                self._terminal = frame
            elif not self.closing:
                raise WorkerMessageViolationError("stream output without read credit")
            if terminal:
                # Acknowledge even an idle deadline, so the host needs no terminal tombstones.
                self._start_close()

    def disconnect(self, exc: Exception) -> None:
        for future in (self.opened, self._read):
            if future is not None and not future.done():
                future.set_exception(exc)
        self._terminal = LlmChatErrorFrame(detail=str(exc))
        if not self._closed.done():
            self._closed.set_exception(exc)
        self._start_close()

    def _start_close(self) -> asyncio.Task[None]:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
            # Idle terminals close without an explicit caller; retain the error for aclose().
            self._close_task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return self._close_task

    async def _close(self) -> None:
        try:
            if not self._closed.done():
                await self._send_control(ToolStreamClose(call_id=self.call_id))
            await asyncio.shield(self._closed)
        finally:
            self._release()

    async def aclose(self) -> None:
        await finish_cleanup(self._start_close())


class SupervisorToolStream:
    """Own an upstream stream until terminal/close and private delivery acknowledgment."""

    def __init__(
        self,
        call: ToolCall,
        open_stream: Callable[[], Awaitable[LlmChatFrameStream]],
        reply: Callable[[StreamReply], Awaitable[None]],
    ) -> None:
        self.call = call
        self._open_stream = open_stream
        self._reply = reply
        self._read = asyncio.Event()
        self._close = asyncio.Event()
        self._reading = False
        self._terminal = False
        self._started = False
        self.task: asyncio.Task[None] | None = None

    def read(self) -> None:
        if self._terminal or self._close.is_set():
            # A read already sent by the worker can cross an idle deadline reply.
            return
        if self._reading:
            raise WorkerMessageViolationError("duplicate stream read credit")
        self._reading = True
        self._read.set()

    def close(self) -> None:
        if self._close.is_set():
            raise WorkerMessageViolationError("duplicate stream close")
        self._close.set()
        if self._started and not self._terminal and self.task is not None:
            self.task.cancel()

    async def run(self) -> None:
        self._started = True
        started = asyncio.get_running_loop().time()
        deadline = started + DEFAULT_LLM_CHAT_TIMEOUT_SECONDS
        upstream: LlmChatFrameStream | None = None
        terminal: LlmChatFrame | None = None
        try:
            if not self._close.is_set():
                request = LlmChatRequest.model_validate(self.call.kwargs)
                timeout = request.timeout
                total = timeout.total if isinstance(timeout, LlmChatTimeout) else timeout
                deadline = started + (total if total is not None else DEFAULT_LLM_CHAT_TIMEOUT_SECONDS)
                async with asyncio.timeout_at(deadline):
                    upstream = await self._open_stream()
                    await self._reply(StreamReply(call_id=self.call.call_id, kind="open"))
                    while True:
                        await self._read.wait()
                        self._read.clear()
                        frame = await anext(upstream)
                        # Clear before sending: the next worker credit may arrive during drain.
                        self._reading = False
                        if not isinstance(frame, LlmChatOutputFrame):
                            terminal = frame
                            break
                        await self._reply(StreamReply(call_id=self.call.call_id, kind="frame", frame=frame))
                    # Terminal delivery is inside the same absolute deadline.
                    await self._reply(StreamReply(call_id=self.call.call_id, kind="frame", frame=terminal))
        except asyncio.CancelledError:
            if not self._close.is_set():
                raise
        except Exception as exc:
            self._terminal = True
            terminal = LlmChatErrorFrame(detail=str(exc) or type(exc).__name__)
            # After expiry, only an immediately writable connection can carry the error.
            # A blocked connection must fail instead of extending the call's deadline.
            async with asyncio.timeout_at(deadline):
                await self._reply(StreamReply(call_id=self.call.call_id, kind="frame", frame=terminal))
        finally:
            self._terminal = True
            if upstream is not None:
                await finish_cleanup(asyncio.create_task(upstream.aclose()))
        # Terminal acknowledgment keeps protocol state bounded by existing call slots.
        await self._close.wait()
        await self._reply(StreamReply(call_id=self.call.call_id, kind="closed"))
