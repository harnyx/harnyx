from __future__ import annotations

import asyncio
import json
import os
import struct
from collections.abc import Callable
from types import SimpleNamespace

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
from harnyx_sandbox.sandbox.worker_protocol import WORKER_MESSAGE, WorkerResultProtocolError, load_json, read_frame

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
        b'{"kind":"tool","call_id":0,"method":"x","args":[],"kwargs":{},"token":"secret"}',
    ],
)
def test_hostile_frames_cannot_select_identity(payload):
    with pytest.raises(ValueError if b"session_id" in payload or b"token" in payload else WorkerResultProtocolError):
        WORKER_MESSAGE.validate_python(load_json(payload))


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
