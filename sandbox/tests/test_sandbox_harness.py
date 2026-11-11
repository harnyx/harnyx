from __future__ import annotations

import asyncio
import json
import os
import struct
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from harnyx_sandbox.sandbox import harness as harness_module
from harnyx_sandbox.sandbox.harness import (
    EntrypointRequest,
    SandboxHarness,
    WorkerOutputPipe,
    WorkerOutputReader,
    _EntrypointContext,
)
from harnyx_sandbox.sandbox.worker_protocol import WorkerResultProtocolError, decode_worker_message, read_frame

from harnyx_miner_sdk.sandbox_protocol import AdmissionRequest

pytestmark = pytest.mark.anyio("asyncio")


SOURCE = "pass\n"


@pytest.fixture
async def harness(tmp_path):
    path = tmp_path / "agent.py"
    path.write_text(SOURCE)
    runtime = SandboxHarness(artifact_path=str(path))
    yield runtime
    await runtime.close()


def body(payload=None, seconds=5.0):
    return EntrypointRequest(
        payload=payload or {}, context=_EntrypointContext.model_validate({"time_budget": {"limit_seconds": seconds}})
    )


async def test_missing_process_handle_support_fails_before_miner_execution(harness, monkeypatch):
    def unavailable(_pid):
        raise OSError("process handles unavailable")

    monkeypatch.setattr(harness_module, "open_process_handle", unavailable)
    async with asyncio.timeout(5):
        with pytest.raises(HTTPException) as failure:
            await harness.invoke("probe", body())
    assert failure.value.status_code == 500


async def test_query_requires_cost_context_before_start(harness):
    with pytest.raises(HTTPException) as error:
        await harness.invoke("query", body())
    assert error.value.status_code == 422
    assert harness._compiler is None


@pytest.mark.parametrize("boundary", ("capacity", "compiler"))
async def test_admission_cancellation_wins_over_concurrent_wait_completion(harness, monkeypatch, boundary):
    """A completed inner wait must not swallow cancellation and allocate an orphan."""
    ready = asyncio.get_running_loop().create_future()
    if boundary == "capacity":
        ready.set_result(None)
    compiler = Mock(ready=ready, generation="test", failure=None, close_task=None, close=AsyncMock())
    monkeypatch.setattr(harness, "_compiler", compiler)
    slots = harness._slots = asyncio.Semaphore(0 if boundary == "capacity" else 1)
    reserving = asyncio.create_task(harness.reserve(AdmissionRequest(limit_seconds=5.0), {}))
    try:
        async with asyncio.timeout(1.0):
            while not (slots._waiters if boundary == "capacity" else harness._startup_waiters):
                await asyncio.sleep(0)
        # Complete and cancel in the same event-loop turn, with no scheduling gap.
        if boundary == "capacity":
            slots.release()
        else:
            ready.set_result(None)
        reserving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reserving
        assert not ready.cancelled()
        assert not harness._reservations
        assert harness._startup_waiters == 0
        await asyncio.wait_for(slots.acquire(), 0.1)
        assert slots.locked()  # Exactly one slot was recovered, not released twice.
        slots.release()
    finally:
        reserving.cancel()
        await asyncio.gather(reserving, return_exceptions=True)


async def test_cancelling_compiler_wait_preserves_other_waiters(harness, monkeypatch):
    """Cancelling one reservation must not cancel the shared compiler readiness future."""
    ready = asyncio.get_running_loop().create_future()
    compiler = Mock(ready=ready, generation="test", failure=None, close_task=None, close=AsyncMock())
    monkeypatch.setattr(harness, "_compiler", compiler)
    harness._slots = asyncio.Semaphore(2)
    first = asyncio.create_task(harness.reserve(AdmissionRequest(limit_seconds=5.0), {}))
    second = asyncio.create_task(harness.reserve(AdmissionRequest(limit_seconds=5.0), {}))
    try:
        async with asyncio.timeout(1.0):
            while harness._startup_waiters != 2:
                await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not ready.cancelled()
        compiler.close.assert_not_awaited()
        ready.set_result(None)
        admission = await second
        harness.release(admission)
        assert not harness._slots.locked()
        assert not harness._reservations
    finally:
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.parametrize("boundary", ("capacity", "compiler"))
async def test_admission_timeout_keeps_classification_and_recovers_capacity(harness, monkeypatch, boundary):
    """Wait expiry remains a settled admission timeout and cannot leak or duplicate a slot."""
    ready = asyncio.get_running_loop().create_future()
    if boundary == "capacity":
        ready.set_result(None)
    compiler = Mock(ready=ready, generation="test", failure=None, close_task=None, close=AsyncMock())
    monkeypatch.setattr(harness, "_compiler", compiler)
    harness._slots = asyncio.Semaphore(0 if boundary == "capacity" else 1)
    with pytest.raises(HTTPException) as failed:
        await harness.reserve(AdmissionRequest(limit_seconds=0.01), {})
    assert failed.value.status_code == 504
    assert failed.value.detail == {"error": "admission deadline exceeded", "exception": "TimeoutError"}
    assert not harness._reservations
    assert not ready.cancelled()
    assert harness._startup_waiters == 0
    if boundary == "capacity":
        assert harness._slots.locked()
        compiler.close.assert_not_awaited()
        harness._slots.release()
    else:
        compiler.close.assert_awaited_once()
        assert harness._compiler is None
    await asyncio.wait_for(harness._slots.acquire(), 0.1)
    assert harness._slots.locked()
    harness._slots.release()


async def test_frame_rejects_oversize_before_body():
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack(">Q", harness_module.MAX_WORKER_RESULT_BYTES + 1))
    with pytest.raises(WorkerResultProtocolError):
        await read_frame(reader)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"kind":"result","kind":"tool"}',
        b'{"kind":"result","session_id":"sibling"}',
        b'{"kind":"result","result":NaN}',
        b'{"kind":"result","result":1e999}',
        b'{"kind":"result","result":[-1e999]}',
        br'{"kind":"result","result":{"text":"\ud800"}}',
        br'{"kind":"result","result":{"\udfff":"value"}}',
        br'{"kind":"result","error":{"code":"UnhandledException","exception":"ValueError","message":"\ud800"}}',
        b'{"kind":"tool","call_id":0,"method":"x","args":[],"kwargs":{},"token":"secret"}',
    ],
)
def test_hostile_frames_cannot_select_identity(payload):
    with pytest.raises((ValueError, WorkerResultProtocolError)):
        decode_worker_message(payload)


@pytest.mark.anyio("asyncio")
async def test_worker_output_without_newline_is_forwarded_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(harness_module, "MAX_WORKER_OUTPUT_MESSAGE_CHARACTERS", 16)
    monkeypatch.setattr(harness_module, "WORKER_RESULT_READ_CHUNK_BYTES", 17)
    pipe = WorkerOutputPipe.open()
    payload = {
        "entrypoint_name": "miner_output",
        "headers": {"x-session-id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
    }
    output_text = "🙂" * 41
    output = output_text.encode()

    def write_output() -> None:
        remaining = memoryview(output)
        while remaining:
            remaining = remaining[os.write(pipe.write_fd, remaining) :]
        pipe.close_write()

    await asyncio.gather(
        WorkerOutputReader(pipe=pipe, payload=payload, stream="stdout").wait(),
        asyncio.to_thread(write_output),
    )

    records = [
        json.loads(line.removeprefix("HARNYX_SANDBOX_INVOCATION_OUTPUT "))
        for line in capfd.readouterr().out.splitlines()
        if line.startswith("HARNYX_SANDBOX_INVOCATION_OUTPUT ")
    ]
    assert "".join(record["message"] for record in records) == output_text
    assert all(len(record["message"]) <= 16 for record in records)


@pytest.mark.anyio("asyncio")
async def test_worker_output_reader_yields_while_pipe_remains_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = WorkerOutputPipe.open()
    payload = {
        "entrypoint_name": "miner_output",
        "headers": {"x-session-id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
    }
    reader = WorkerOutputReader(pipe=pipe, payload=payload, stream="stdout")
    unrelated_advanced = asyncio.Event()
    available_records = 10_000
    emitted: list[str] = []

    def emit(message: str) -> None:
        emitted.append(message)
        if len(emitted) == 1:
            asyncio.get_running_loop().call_soon(unrelated_advanced.set)

    def write_output() -> None:
        output = memoryview(("x\n" * available_records).encode())
        while output:
            output = output[os.write(pipe.write_fd, output) :]
        pipe.close_write()

    try:
        monkeypatch.setattr(reader, "_emit", emit)
        read_task = asyncio.create_task(reader.wait())
        write_task = asyncio.create_task(asyncio.to_thread(write_output))
        await asyncio.wait_for(unrelated_advanced.wait(), timeout=1.0)

        assert len(emitted) < available_records
        await asyncio.gather(read_task, write_task)
        assert emitted == ["x"] * available_records
    finally:
        reader.close()
        pipe.close_write()


def test_worker_output_reader_keeps_one_continuation_during_repeated_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = WorkerOutputPipe.open()
    payload = {
        "entrypoint_name": "miner_output",
        "headers": {"x-session-id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
    }
    reader = WorkerOutputReader(pipe=pipe, payload=payload, stream="stdout")
    scheduled: list[Callable[[], None]] = []
    emitted: list[str] = []

    try:
        reader._loop = SimpleNamespace(call_soon=scheduled.append)  # type: ignore[assignment]
        reader._buffer = "x\n" * 10_000
        monkeypatch.setattr(reader, "_emit", emitted.append)

        reader._output_ready()
        reader._output_ready()

        assert len(scheduled) == 1
        scheduled.pop()()
        assert len(scheduled) <= 1
        assert emitted
    finally:
        reader.close()
        pipe.close_write()


async def test_oom_counter_latches_initial_positive_and_rejects_later_reset(tmp_path):
    """Future failure: a startup OOM or counter reset must not silently reopen admission."""
    from harnyx_sandbox.sandbox.memory import EvaluationMemoryState

    counter = tmp_path / "memory.events.local"
    counter.write_text("oom 1\noom_kill 1\n")
    memory = EvaluationMemoryState(counter)
    assert memory.refresh().draining
    counter.write_text("oom 0\noom_kill 0\n")
    with pytest.raises(RuntimeError, match="backwards"):
        memory.refresh()
    assert memory.draining


async def test_evaluation_admission_slots_are_not_released_by_active_release(harness, tmp_path):
    """Future failure: release/expiry must not make an eleventh worker fit beside ten live workers."""
    from harnyx_sandbox.sandbox.memory import EvaluationMemoryState

    from harnyx_miner_sdk.sandbox_protocol import AdmissionRequest

    counter = tmp_path / "memory.events.local"
    counter.write_text("oom_kill 0\n")
    harness._memory = EvaluationMemoryState(counter)
    harness._slots = asyncio.Semaphore(10)
    reservations = [await harness.reserve(AdmissionRequest(limit_seconds=5.0), {}) for _ in range(10)]
    pending = asyncio.create_task(harness.reserve(AdmissionRequest(limit_seconds=5.0), {}))
    await asyncio.sleep(0.02)
    assert not pending.done()
    harness._reservations[reservations[0].reservation_id].activated = True
    harness.release(reservations[0])
    await asyncio.sleep(0.02)
    assert not pending.done()
    harness.release(reservations[1])
    following = await asyncio.wait_for(pending, 1)
    assert len(harness._reservations) == 10
    harness.release(following)
    counter.write_text("oom_kill 1\n")
    with pytest.raises(HTTPException) as rejected:
        await harness.reserve(AdmissionRequest(limit_seconds=5.0), {})
    assert rejected.value.detail["code"] == "SandboxOomDraining"
    assert harness._compiler.generation == reservations[0].generation


async def test_unreadable_counter_does_not_become_oom_evidence(harness, tmp_path):
    """Future failure: observation loss must be infrastructure uncertainty, not miner OOM."""
    from harnyx_sandbox.sandbox.memory import EvaluationMemoryState

    counter = tmp_path / "memory.events.local"
    counter.write_text("oom_kill 0\n")
    harness._memory = EvaluationMemoryState(counter)
    counter.unlink()
    status = harness.resource_status()
    assert status.counter_status == "unavailable"
    assert not status.draining


@pytest.mark.parametrize("mount_root,membership", [("/", "/"), ("/tenant", "/tenant/query")])
def test_counter_resolution_uses_own_v2_membership(monkeypatch, tmp_path, mount_root, membership):
    """The trusted reader must resolve the container cgroup rather than a visible ancestor."""
    from pathlib import Path

    from harnyx_sandbox.sandbox.memory import EvaluationMemoryState

    mount = tmp_path / "cgroup mount"
    counter = mount / ("query" if membership.endswith("query") else "") / "memory.events.local"
    counter.parent.mkdir(parents=True)
    counter.write_text("oom_kill 0\n")
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return f"0::{membership}\n"
        if str(path) == "/proc/self/mountinfo":
            escaped = str(mount).replace(" ", r"\040")
            return f"1 0 0:1 {mount_root} {escaped} ro - cgroup2 cgroup rw\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert EvaluationMemoryState().counter == counter


@pytest.mark.parametrize("control", ["stream_read", "stream_close"])
@pytest.mark.parametrize("extra", [{"reservation_id": "other-query"}, {"call_id": -1}])
async def test_stream_control_cannot_select_other_query_or_invalid_call(control, extra):
    from harnyx_sandbox.sandbox.worker_protocol import decode_worker_message
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        decode_worker_message(json.dumps({"kind": control, "call_id": 1, **extra}).encode())


@pytest.mark.parametrize("late_control", ["read", "close"])
async def test_idle_stream_deadline_crossing_control_closes_without_tombstones(late_control):
    """An idle deadline can cross a worker request; neither side may await an obsolete reply."""
    from harnyx_sandbox.sandbox.tool_stream import SupervisorToolStream, WorkerToolStream
    from harnyx_sandbox.sandbox.worker_protocol import ToolCall, ToolStreamRead

    from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatErrorFrame

    terminal_ready, release_control, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
    upstream = Mock(aclose=AsyncMock())
    delivered = []

    async def reply(message):
        if message.kind == "frame":
            terminal_ready.set()
        delivered.append(message.kind)
        worker.receive(message)

    async def control(message):
        await release_control.wait()
        if isinstance(message, ToolStreamRead):
            host.read()
        else:
            host.close()

    worker = WorkerToolStream(1, control, released.set)
    host = SupervisorToolStream(ToolCall(call_id=1, method="llm_chat", args=[], kwargs={
        "stream": True, "provider": "chutes", "model": "test", "messages": [{"role": "user", "content": "hi"}],
        "timeout": .01,
    }), AsyncMock(return_value=upstream), reply)
    host.task = asyncio.create_task(host.run())
    try:
        async with asyncio.timeout(1):
            await worker.opened
            operation = asyncio.create_task(anext(worker) if late_control == "read" else worker.aclose())
            await terminal_ready.wait()
            release_control.set()
            result = await operation
            if late_control == "read":
                assert isinstance(result, LlmChatErrorFrame)
            await worker.aclose()
            await host.task
        assert released.is_set()
        upstream.aclose.assert_awaited_once()
        assert delivered[-1] == "closed"
    finally:
        host.task.cancel()
        await asyncio.gather(host.task, return_exceptions=True)


async def test_supervisor_stream_rejects_multiple_outstanding_reads():
    from harnyx_sandbox.sandbox.tool_stream import SupervisorToolStream
    from harnyx_sandbox.sandbox.worker_protocol import ToolCall, WorkerMessageViolationError

    host = SupervisorToolStream(ToolCall(call_id=1, method="llm_chat", args=[], kwargs={}), AsyncMock(), AsyncMock())
    host.read()
    with pytest.raises(WorkerMessageViolationError, match="duplicate"):
        host.read()


@pytest.mark.parametrize("upstream_failure", [False, True])
async def test_blocked_terminal_delivery_does_not_extend_stream_total_timeout(upstream_failure):
    from harnyx_sandbox.sandbox.tool_stream import SupervisorToolStream
    from harnyx_sandbox.sandbox.worker_protocol import ToolCall

    from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatCompletedFrame

    closed = asyncio.Event()

    class Frames:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if upstream_failure:
                raise RuntimeError("upstream failed")
            return LlmChatCompletedFrame(result={})

        async def aclose(self):
            closed.set()

    async def reply(message):
        if message.kind == "open":
            host.read()
        else:
            await asyncio.Future()

    host = SupervisorToolStream(ToolCall(call_id=1, method="llm_chat", args=[], kwargs={
        "stream": True, "provider": "chutes", "model": "test", "messages": [{"role": "user", "content": "hi"}],
        "timeout": .01,
    }), AsyncMock(return_value=Frames()), reply)
    async with asyncio.timeout(1):
        with pytest.raises(TimeoutError):
            await host.run()
    assert closed.is_set()


async def test_stream_lost_close_ack_cannot_publish_successful_result():
    from harnyx_sandbox.sandbox.tool_stream import WorkerToolStream
    from harnyx_sandbox.sandbox.worker_protocol import StreamReply

    from harnyx_miner_sdk.streaming import LlmChatStream
    from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatCompletedFrame
    from harnyx_miner_sdk.tools.proxy import ToolInvocationError

    worker = WorkerToolStream(1, AsyncMock(), Mock())
    worker.receive(StreamReply(call_id=1, kind="open"))
    stream = LlmChatStream(worker, lambda value: value)
    read = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    worker.receive(StreamReply(call_id=1, kind="frame", frame=LlmChatCompletedFrame(result={})))
    worker.disconnect(ToolInvocationError("connection lost before close acknowledgment"))
    with pytest.raises(ToolInvocationError, match="connection lost"):
        await read
    with pytest.raises(RuntimeError, match="successful stream exhaustion"):
        _ = stream.result


@pytest.mark.parametrize("ending", ["reply", "error", "disconnect"])
async def test_cancelled_worker_call_releases_capacity_only_after_host_completion(ending):
    """Abandoned calls retain capacity, then free it on replies or connection loss."""
    from harnyx_sandbox.sandbox.compiler import ConnectionInvoker
    from harnyx_sandbox.sandbox.worker_protocol import ToolReply, encode

    reader = asyncio.StreamReader()
    written = asyncio.Event()
    writer = Mock(write=lambda _: written.set(), drain=AsyncMock())
    invoker = ConnectionInvoker(reader, writer)
    invoker.slots = asyncio.Semaphore(1)
    receiver = asyncio.create_task(invoker.receive())
    call = asyncio.create_task(invoker.invoke("tooling_info"))
    try:
        await written.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert invoker.slots.locked()
        if ending == "disconnect":
            reader.feed_eof()
        else:
            reader.feed_data(encode(ToolReply(call_id=0, result="ok", error="failed" if ending == "error" else None)))
        await asyncio.wait_for(invoker.slots.acquire(), 1)
        assert not invoker.calls
    finally:
        reader.feed_eof()
        await receiver


@pytest.mark.parametrize("failure", ["serialization", "cancel_before_send"])
@pytest.mark.parametrize("stream", [False, True])
async def test_unsent_worker_call_does_not_leak_capacity(failure, stream):
    """Local send failures must leave capacity available without a host reply."""
    from harnyx_sandbox.sandbox.compiler import ConnectionInvoker

    writer = Mock(drain=AsyncMock())
    invoker = ConnectionInvoker(asyncio.StreamReader(), writer)
    invoker.slots = asyncio.Semaphore(1)

    async def invoke():
        kwargs = {"temperature": float("nan")} if failure == "serialization" else {}
        if stream:
            return await invoker.invoke_llm_chat_stream(kwargs=kwargs)
        return await invoker.invoke("tooling_info", kwargs=kwargs)

    if failure == "serialization":
        with pytest.raises(ValueError):
            await asyncio.wait_for(invoke(), 1)
    else:
        await invoker.write_lock.acquire()
        call = asyncio.create_task(invoke())
        await asyncio.sleep(0)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        invoker.write_lock.release()
    await asyncio.wait_for(invoker.slots.acquire(), 1)
    assert not invoker.calls
    assert not invoker.streams
    writer.write.assert_not_called()
