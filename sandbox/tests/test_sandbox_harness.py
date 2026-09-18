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
