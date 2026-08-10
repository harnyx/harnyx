from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

import harnyx_sandbox.app as sandbox_app
import harnyx_sandbox.sandbox.harness as harness_module
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from harnyx_sandbox.sandbox.harness import (
    SandboxHarness,
    SandboxPreloadFailure,
    WorkerOutputPipe,
    WorkerOutputReader,
    WorkerResultPipe,
    WorkerResultProtocolError,
    WorkerResultReader,
    _send_worker_result,
)

from harnyx_miner_sdk.api import test_tool as invoke_test_tool
from harnyx_miner_sdk.decorators import clear_entrypoints, entrypoint, entrypoint_exists
from harnyx_miner_sdk.query import Query, Response
from harnyx_miner_sdk.safe_exec import safe_exec


def _detail_code(response) -> str:
    return response.json()["detail"]["code"]


class _FakeWorkerProcess:
    def __init__(self) -> None:
        self.sentinel_read_fd, self._sentinel_write_fd = os.pipe()
        self.sentinel = self.sentinel_read_fd
        self.terminated = False
        self.killed = False
        self.join_calls = 0
        self._sentinel_closed = False

    def is_alive(self) -> bool:
        return not self.terminated and not self.killed

    def terminate(self) -> None:
        self.terminated = True
        self.finish()

    def join(self, timeout: float) -> None:
        del timeout
        self.join_calls += 1

    def kill(self) -> None:
        self.killed = True
        self.finish()

    def finish(self) -> None:
        if self._sentinel_closed:
            return
        self._sentinel_closed = True
        os.close(self._sentinel_write_fd)

    def close(self) -> None:
        self.finish()
        with contextlib.suppress(OSError):
            os.close(self.sentinel_read_fd)


def _make_worker_payload() -> dict[str, object]:
    return {
        "entrypoint_name": "miner_test",
        "headers": {},
    }


@pytest.mark.anyio("asyncio")
async def test_worker_result_wait_does_not_use_executor_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()

    def fail_run_in_executor(*_args: object, **_kwargs: object) -> None:
        pytest.fail("worker result wait must not use run_in_executor")

    async def decode_without_executor(func, *args):
        return func(*args)

    monkeypatch.setattr(asyncio, "to_thread", decode_without_executor)
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", fail_run_in_executor)

    async def send_result() -> None:
        await asyncio.sleep(0)
        _send_worker_result(pipe.write_fd, ("ok", {"status": "ready"}))
        os.close(pipe.write_fd)
        process.finish()

    send_task = asyncio.create_task(send_result())
    try:
        result = await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        await send_task
        process.close()

    assert result == ("ok", {"status": "ready"})


@pytest.mark.anyio("asyncio")
async def test_worker_result_timeout_cleans_wait_state_and_terminates_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    harness = SandboxHarness()
    monkeypatch.setattr(harness_module, "ENTRYPOINT_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(HTTPException) as exc_info:
        await harness._await_worker_result(pipe, _make_worker_payload(), process)

    os.close(pipe.write_fd)
    process.close()
    assert exc_info.value.status_code == 504
    assert process.terminated is True


@pytest.mark.anyio("asyncio")
async def test_worker_result_cancellation_waits_for_owned_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = SandboxHarness()
    termination_started = threading.Event()
    allow_termination = threading.Event()

    class TimedOutResultReader:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def wait(self, *, timeout: float) -> tuple[str, object]:
            del timeout
            raise TimeoutError

    def blocking_termination(_process: object) -> None:
        termination_started.set()
        assert allow_termination.wait(timeout=1.0)

    monkeypatch.setattr(harness_module, "WorkerResultReader", TimedOutResultReader)
    monkeypatch.setattr(harness, "_terminate_process", blocking_termination)

    result_task = asyncio.create_task(
        harness._await_worker_result(object(), _make_worker_payload(), object())  # type: ignore[arg-type]
    )
    assert await asyncio.to_thread(termination_started.wait, 1.0)
    result_task.cancel()
    await asyncio.sleep(0)

    assert not result_task.done()

    allow_termination.set()
    with pytest.raises(asyncio.CancelledError):
        await result_task


@pytest.mark.anyio("asyncio")
async def test_worker_result_partial_frame_does_not_block_event_loop() -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    os.write(pipe.write_fd, struct.pack(">Q", 100) + b"partial")

    wait_task = asyncio.create_task(WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not wait_task.done()

    wait_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await wait_task
    os.close(pipe.write_fd)
    process.close()


@pytest.mark.anyio("asyncio")
async def test_worker_exit_after_full_frame_preserves_worker_result() -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    _send_worker_result(pipe.write_fd, ("ok", {"value": 1}))
    os.close(pipe.write_fd)
    process.finish()

    try:
        result = await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()

    assert result == ("ok", {"value": 1})


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


@pytest.mark.anyio("asyncio")
async def test_worker_process_join_allows_output_readers_to_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = SandboxHarness()
    output_readers_finished = threading.Event()
    output_reader_count = 0

    class BlockingJoinProcess:
        def join(self, timeout: float) -> None:
            del timeout
            assert output_readers_finished.wait(timeout=1.0)

        def is_alive(self) -> bool:
            return False

        def kill(self) -> None:
            pytest.fail("an exited process must not be killed")

    class AdvancingOutputReader:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def wait(self) -> None:
            nonlocal output_reader_count
            output_reader_count += 1
            if output_reader_count == 2:
                output_readers_finished.set()

    process = BlockingJoinProcess()

    async def completed_result(*_args: object) -> tuple[str, object]:
        return "ok", {"status": "ready"}

    monkeypatch.setattr(harness, "_spawn_worker", lambda _payload: (process, object(), object(), object()))
    monkeypatch.setattr(harness, "_await_worker_result", completed_result)
    monkeypatch.setattr(harness_module, "WorkerOutputReader", AdvancingOutputReader)

    result = await harness._invoke_with_worker(_make_worker_payload())

    assert result == {"status": "ready"}
    assert output_reader_count == 2


@pytest.mark.anyio("asyncio")
async def test_worker_process_cancellation_waits_for_join_and_output_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = SandboxHarness()
    join_started = threading.Event()
    allow_join = threading.Event()
    allow_output_readers = asyncio.Event()
    output_reader_count = 0

    class BlockingJoinProcess:
        def join(self, timeout: float) -> None:
            del timeout
            join_started.set()
            assert allow_join.wait(timeout=1.0)

        def is_alive(self) -> bool:
            return False

        def kill(self) -> None:
            pytest.fail("an exited process must not be killed")

    class WaitingOutputReader:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def wait(self) -> None:
            nonlocal output_reader_count
            await allow_output_readers.wait()
            output_reader_count += 1

    process = BlockingJoinProcess()

    async def completed_result(*_args: object) -> tuple[str, object]:
        return "ok", {"status": "ready"}

    monkeypatch.setattr(harness, "_spawn_worker", lambda _payload: (process, object(), object(), object()))
    monkeypatch.setattr(harness, "_await_worker_result", completed_result)
    monkeypatch.setattr(harness_module, "WorkerOutputReader", WaitingOutputReader)

    invocation_task = asyncio.create_task(harness._invoke_with_worker(_make_worker_payload()))
    assert await asyncio.to_thread(join_started.wait, 1.0)
    invocation_task.cancel()
    await asyncio.sleep(0)

    assert not invocation_task.done()

    allow_output_readers.set()
    allow_join.set()
    with pytest.raises(asyncio.CancelledError):
        await invocation_task
    assert output_reader_count == 2


@pytest.mark.parametrize("failure_step", ["stdout", "stderr", "process", "start"])
def test_spawn_worker_closes_every_acquired_pipe_after_partial_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_step: str,
) -> None:
    harness = SandboxHarness()
    acquired: list[SimpleNamespace] = []
    output_open_count = 0

    def open_pipe(kind: str) -> SimpleNamespace:
        nonlocal output_open_count
        if kind == "output":
            output_open_count += 1
            step = "stdout" if output_open_count == 1 else "stderr"
            if failure_step == step:
                raise RuntimeError(f"{step} setup failed")
        pipe = SimpleNamespace(
            read_fd=10 + len(acquired) * 2,
            write_fd=11 + len(acquired) * 2,
            close_calls=0,
            close=lambda: None,
        )

        def close() -> None:
            pipe.close_calls += 1

        pipe.close = close
        acquired.append(pipe)
        return pipe

    class Process:
        def __init__(self, **_kwargs: object) -> None:
            if failure_step == "process":
                raise RuntimeError("process construction failed")

        def start(self) -> None:
            if failure_step == "start":
                raise RuntimeError("process start failed")

    monkeypatch.setattr(harness_module.WorkerResultPipe, "open", lambda: open_pipe("result"))
    monkeypatch.setattr(harness_module.WorkerOutputPipe, "open", lambda: open_pipe("output"))
    harness._mp = SimpleNamespace(Process=Process)

    with pytest.raises(RuntimeError, match="failed"):
        harness._spawn_worker(
            {
                "entrypoint_name": "miner_test",
                "request_payload": {},
                "context": {},
                "tool_config": {},
                "headers": {},
                "preload": None,
            }
        )

    assert acquired
    assert all(pipe.close_calls == 1 for pipe in acquired)


@pytest.mark.anyio("asyncio")
async def test_parent_oversized_frame_is_parent_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    harness = SandboxHarness()
    monkeypatch.setattr(harness_module, "MAX_WORKER_RESULT_BYTES", 16)
    os.write(pipe.write_fd, struct.pack(">Q", 17))
    os.close(pipe.write_fd)
    process.finish()

    with pytest.raises(HTTPException) as exc_info:
        await harness._await_worker_result(pipe, _make_worker_payload(), process)

    process.close()
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["exception"] == "WorkerResultProtocolError"


@pytest.mark.anyio("asyncio")
async def test_worker_result_too_large_is_structured_worker_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    monkeypatch.setattr(harness_module, "MAX_WORKER_RESULT_BYTES", 1024)

    _send_worker_result(pipe.write_fd, ("ok", "x" * 2000))
    os.close(pipe.write_fd)
    process.finish()

    try:
        envelope = await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()

    assert envelope[0] == "error"
    assert envelope[1]["code"] == "ResultTooLarge"


@pytest.mark.anyio("asyncio")
async def test_parent_reader_accepts_max_size_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    payload = b'{"status":"ok","result":{"value":"at-limit"}}'
    monkeypatch.setattr(harness_module, "MAX_WORKER_RESULT_BYTES", len(payload))
    os.write(pipe.write_fd, struct.pack(">Q", len(payload)) + payload)
    os.close(pipe.write_fd)
    process.finish()

    try:
        result = await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()

    assert result == ("ok", {"value": "at-limit"})


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(("tuple",), id="tuple"),
        pytest.param({1: "integer-key", "1": "string-key"}, id="non-string-colliding-key"),
        pytest.param(float("nan"), id="non-finite-number"),
    ],
)
async def test_worker_rejects_non_json_success_values(value: object) -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    _send_worker_result(pipe.write_fd, ("ok", value))
    os.close(pipe.write_fd)
    process.finish()

    try:
        envelope = await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()

    assert envelope[0] == "error"
    assert envelope[1]["code"] == "UnhandledException"


@pytest.mark.anyio("asyncio")
async def test_worker_rejects_builtin_json_subclasses() -> None:
    class MinerString(str):
        pass

    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    _send_worker_result(pipe.write_fd, ("ok", {"value": MinerString("subclass")}))
    os.close(pipe.write_fd)
    process.finish()

    try:
        envelope = await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()

    assert envelope[0] == "error"
    assert envelope[1]["code"] == "UnhandledException"


@pytest.mark.anyio("asyncio")
async def test_worker_rejects_custom_result_without_using_reduce() -> None:
    reduce_called = False

    class MaliciousResult:
        def __reduce__(self) -> object:
            nonlocal reduce_called
            reduce_called = True
            return str, ("executed",)

    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    _send_worker_result(pipe.write_fd, ("ok", MaliciousResult()))
    os.close(pipe.write_fd)
    process.finish()

    try:
        envelope = await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()

    assert reduce_called is False
    assert envelope[0] == "error"
    assert envelope[1]["code"] == "UnhandledException"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b'{"status":"ok","status":"ok","result":null}', id="top-level"),
        pytest.param(b'{"status":"ok","result":{"value":1,"value":2}}', id="nested-result"),
        pytest.param(
            b'{"status":"error","error":{"code":"x","exception":"x","message":"x","message":"y"}}',
            id="nested-error",
        ),
    ],
)
async def test_parent_rejects_duplicate_json_object_names(payload: bytes) -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    os.write(pipe.write_fd, struct.pack(">Q", len(payload)) + payload)
    os.close(pipe.write_fd)
    process.finish()

    try:
        with pytest.raises(WorkerResultProtocolError, match="invalid result frame"):
            await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"\xff", id="invalid-utf8"),
        pytest.param(b"{", id="malformed-json"),
        pytest.param(b'{"status":"ok","result":NaN}', id="non-finite"),
        pytest.param(b'{"status":"ok"}', id="missing-result"),
        pytest.param(b'{"status":"ok","result":null,"extra":1}', id="extra-field"),
        pytest.param(b'{"status":1,"result":null}', id="wrong-status-type"),
        pytest.param(
            b'{"status":"error","error":{"code":"x","exception":"x"}}',
            id="missing-error-field",
        ),
        pytest.param(
            b'{"status":"error","error":{"code":"x","exception":"x","message":1}}',
            id="wrong-error-field-type",
        ),
    ],
)
async def test_parent_rejects_invalid_json_result_envelopes(payload: bytes) -> None:
    pipe = WorkerResultPipe.open()
    process = _FakeWorkerProcess()
    os.write(pipe.write_fd, struct.pack(">Q", len(payload)) + payload)
    os.close(pipe.write_fd)
    process.finish()

    try:
        with pytest.raises(WorkerResultProtocolError, match="invalid result frame"):
            await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()


def test_harness_round_trips_miner_sdk_response_as_json_object() -> None:
    clear_entrypoints()

    @entrypoint("query")
    async def response_entrypoint(query: Query) -> Response:
        return Response(text=query.text)

    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")

    response = TestClient(app).post(
        "/entry/query",
        json={"payload": {"text": "answer"}, "context": {}},
        headers={"x-platform-token": "token", "x-session-id": "session-response"},
    )
    clear_entrypoints()

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "result": {"text": "answer", "citations": None},
    }


def test_harness_invokes_entrypoint_and_closes_tools() -> None:
    close_flag = mp.Value("i", 0)
    factory_calls = mp.Value("i", 0)
    invoke_calls = mp.Value("i", 0)

    class FakeToolProxy:
        async def invoke(
            self,
            name: str,
            *,
            args: tuple[object, ...] | None = None,
            kwargs: dict[str, object] | None = None,
        ) -> dict[str, object]:
            del name
            with invoke_calls.get_lock():
                invoke_calls.value += 1
            message: str = ""
            if args:
                message = str(args[0])
            if kwargs and "message" in kwargs:
                message = str(kwargs["message"])
            return {
                "receipt_id": "tool-1",
                "response": {"status": "ok", "echo": message},
                "results": [],
                "result_policy": "log_only",
                "budget": {
                    "session_budget_usd": 1.0,
                    "session_hard_limit_usd": 1.0,
                    "session_used_budget_usd": 0.0,
                    "session_remaining_budget_usd": 1.0,
                },
            }

        async def aclose(self) -> None:
            with close_flag.get_lock():
                close_flag.value = 1

    def tool_factory(
        config: Mapping[str, object] | None,
        headers: Mapping[str, str],
    ) -> FakeToolProxy:
        del config, headers
        with factory_calls.get_lock():
            factory_calls.value += 1
        return FakeToolProxy()

    @entrypoint("miner_echo")
    async def echo_entrypoint(request: dict[str, object]) -> dict[str, object]:
        tool_result = await invoke_test_tool(str(request.get("message", "")))
        return {
            "message": request.get("message"),
            "echo": tool_result.response.echo,
        }

    harness = SandboxHarness(tool_factory=tool_factory)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/miner_echo",
        json={
            "payload": {"message": "hello"},
            "context": {"run_id": "abc"},
        },
        headers={"x-platform-token": "token", "x-session-id": "session-1"},
    )

    assert response.status_code == 200
    assert response.json()["result"] == {
        "message": "hello",
        "echo": "hello",
    }

    assert factory_calls.value == 1
    assert invoke_calls.value == 1
    assert close_flag.value == 1


def test_harness_attributes_received_worker_stdout_and_stderr_to_invocation(
    capfd: pytest.CaptureFixture[str],
) -> None:
    @entrypoint("miner_output")
    async def output_entrypoint(request: dict[str, object]) -> dict[str, object]:
        print(f"stdout:{request['message']}")
        print(f"stderr:{request['message']}", file=sys.stderr)
        return {"ok": True}

    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/miner_output",
        json={"payload": {"message": "owned"}, "context": {}},
        headers={"x-platform-token": "token", "x-session-id": "session-output"},
    )

    assert response.status_code == 200
    captured = capfd.readouterr().out.splitlines()
    records = [
        line.removeprefix("HARNYX_SANDBOX_INVOCATION_OUTPUT ")
        for line in captured
        if line.startswith("HARNYX_SANDBOX_INVOCATION_OUTPUT ")
    ]
    assert any(
        '"session_id":"session-output"' in record
        and '"entrypoint":"miner_output"' in record
        and '"stream":"stdout"' in record
        and '"message":"stdout:owned"' in record
        for record in records
    ), captured
    assert any(
        '"session_id":"session-output"' in record
        and '"entrypoint":"miner_output"' in record
        and '"stream":"stderr"' in record
        and '"message":"stderr:owned"' in record
        for record in records
    )


def test_harness_executes_safe_exec_inside_worker() -> None:
    @entrypoint("miner_safe_exec")
    async def safe_exec_entrypoint(request: dict[str, object]) -> dict[str, object]:
        values = request.get("values")
        average = safe_exec(
            "import statistics\nresult = statistics.mean(values)",
            {"values": values},
        )
        return {"average": average}

    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/miner_safe_exec",
        json={"payload": {"values": [2, 4, 6]}, "context": {}},
        headers={"x-platform-token": "token", "x-session-id": "session-1"},
    )

    assert response.status_code == 200
    assert response.json()["result"] == {"average": 4}


def test_harness_builds_tool_proxy_before_preload() -> None:
    close_flag = mp.Value("i", 0)
    order = mp.Value("i", 0)
    factory_order = mp.Value("i", 0)
    preload_order = mp.Value("i", 0)

    class FakeToolProxy:
        async def invoke(
            self,
            name: str,
            *,
            args: tuple[object, ...] | None = None,
            kwargs: dict[str, object] | None = None,
        ) -> dict[str, object]:
            del name, args, kwargs
            return {
                "receipt_id": "tool-1",
                "response": {"status": "ok", "echo": ""},
                "results": [],
                "result_policy": "log_only",
                "budget": {
                    "session_budget_usd": 1.0,
                    "session_hard_limit_usd": 1.0,
                    "session_used_budget_usd": 0.0,
                    "session_remaining_budget_usd": 1.0,
                },
            }

        async def aclose(self) -> None:
            with close_flag.get_lock():
                close_flag.value = 1

    def tool_factory(
        config: Mapping[str, object] | None,
        headers: Mapping[str, str],
    ) -> FakeToolProxy:
        del config, headers
        with order.get_lock():
            order.value += 1
            factory_order.value = order.value
        return FakeToolProxy()

    def preload() -> None:
        with order.get_lock():
            order.value += 1
            preload_order.value = order.value

    @entrypoint("miner_factory_then_preload")
    async def ordered_entrypoint(request: dict[str, object]) -> dict[str, object]:
        return {"message": request.get("message")}

    harness = SandboxHarness(tool_factory=tool_factory, preload=preload)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/miner_factory_then_preload",
        json={"payload": {"message": "ok"}, "context": {}},
        headers={"x-platform-token": "token", "x-session-id": "session-1"},
    )

    assert response.status_code == 200
    assert response.json()["result"] == {"message": "ok"}
    assert factory_order.value == 1
    assert preload_order.value == 2
    assert close_flag.value == 1


def test_harness_accepts_neutral_session_header() -> None:
    @entrypoint("neutral_session_echo")
    async def neutral_entrypoint(request: dict[str, object]) -> dict[str, object]:
        return {"message": request.get("message")}

    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/neutral_session_echo",
        json={"payload": {"message": "hello"}, "context": {}},
        headers={"x-platform-token": "token", "x-session-id": "session-1"},
    )

    assert response.status_code == 200
    assert response.json()["result"] == {"message": "hello"}


def test_unknown_entrypoint_without_preload_returns_500() -> None:
    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post("/entry/missing", json={})
    assert response.status_code == 500
    assert _detail_code(response) == "EntrypointUnavailable"


def test_unknown_entrypoint_returns_404_with_preload() -> None:
    def noop_preload() -> None:  # executed in worker before lookup
        return None

    harness = SandboxHarness(preload=noop_preload)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post("/entry/missing", json={})
    assert response.status_code == 404
    assert _detail_code(response) == "MissingEntrypoint"


def test_worker_reports_preload_failure_with_phase_specific_code() -> None:
    clear_entrypoints()

    def preload() -> None:
        raise TypeError("query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query")

    harness = SandboxHarness(preload=preload)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post("/entry/missing", json={})

    assert response.status_code == 500
    assert _detail_code(response) == "PreloadFailed"
    assert response.json()["detail"]["exception"] == "TypeError"


def test_worker_reports_preload_infrastructure_failure_with_explicit_code() -> None:
    clear_entrypoints()

    def preload() -> SandboxPreloadFailure:
        return SandboxPreloadFailure(
            code="PreloadInfrastructureFailed",
            error="AGENT_PATH is required",
            exception="ValueError",
        )

    harness = SandboxHarness(preload=preload)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post("/entry/missing", json={})

    assert response.status_code == 500
    assert _detail_code(response) == "PreloadInfrastructureFailed"
    assert response.json()["detail"]["exception"] == "ValueError"


def test_worker_does_not_trust_miner_exception_named_like_infrastructure_error() -> None:
    clear_entrypoints()

    class SandboxPreloadInfrastructureError(RuntimeError):
        pass

    def preload() -> None:
        raise SandboxPreloadInfrastructureError("miner-controlled preload failure")

    harness = SandboxHarness(preload=preload)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post("/entry/missing", json={})

    assert response.status_code == 500
    assert _detail_code(response) == "PreloadFailed"
    assert response.json()["detail"]["exception"] == "SandboxPreloadInfrastructureError"


def test_worker_reports_query_runtime_type_error_as_unhandled_exception() -> None:
    clear_entrypoints()

    @entrypoint("miner_runtime_type_error")
    async def runtime_type_error(_request: dict[str, object]) -> dict[str, object]:
        raise TypeError("query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query")

    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post("/entry/miner_runtime_type_error", json={"payload": {}, "context": {}})

    assert response.status_code == 500
    assert _detail_code(response) == "UnhandledException"
    assert response.json()["detail"]["exception"] == "TypeError"


def test_load_agent_from_env_requires_agent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_app, "_agent_loaded", False)
    monkeypatch.delenv("AGENT_PATH", raising=False)
    monkeypatch.delenv("AGENT_MODULE", raising=False)

    assert sandbox_app._load_agent_from_env() == SandboxPreloadFailure(
        code="PreloadInfrastructureFailed",
        error="AGENT_PATH is required",
        exception="ValueError",
    )


def test_load_agent_from_env_requires_present_agent_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sandbox_app, "_agent_loaded", False)
    missing_path = tmp_path / "missing-agent.py"
    monkeypatch.setenv("AGENT_PATH", str(missing_path))
    monkeypatch.delenv("AGENT_MODULE", raising=False)

    assert sandbox_app._load_agent_from_env() == SandboxPreloadFailure(
        code="PreloadInfrastructureFailed",
        error="agent path is not present inside sandbox",
        exception="FileNotFoundError",
    )


def test_load_agent_from_env_wraps_loader_os_error_as_preload_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent_path = tmp_path / "agent.py"
    agent_path.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(sandbox_app, "_agent_loaded", False)
    monkeypatch.setenv("AGENT_PATH", str(agent_path))
    monkeypatch.delenv("AGENT_MODULE", raising=False)
    monkeypatch.setattr(
        sandbox_app.runpy,
        "run_path",
        lambda _path: (_ for _ in ()).throw(PermissionError(13, "denied", str(agent_path))),
    )

    assert sandbox_app._load_agent_from_env() == SandboxPreloadFailure(
        code="PreloadInfrastructureFailed",
        error="failed to read mounted agent path",
        exception="PermissionError",
    )


def test_load_agent_from_env_ignores_miner_monkeypatch_of_preload_globals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    miner_owned_path = tmp_path / "miner-owned.txt"
    agent_path = tmp_path / "agent.py"
    path_type = type(Path.cwd())
    original_relative_to = path_type.relative_to
    sandbox_app_dict = vars(sandbox_app)
    missing = object()
    original_loader_owned_helper = sandbox_app_dict.get("_is_loader_mounted_path_os_error", missing)
    original_preload_failure_helper = sandbox_app_dict.get("_preload_infrastructure_failure", missing)
    agent_path.write_text(
        "\n".join(
            [
                "import pathlib",
                "import harnyx_sandbox.app as sandbox_app",
                "pathlib.PosixPath.relative_to = lambda self, *_args, **_kwargs: self",
                "sandbox_app._is_loader_mounted_path_os_error = lambda *_args, **_kwargs: True",
                (
                    "sandbox_app._preload_infrastructure_failure = "
                    "lambda *_args, **_kwargs: 'forced-infrastructure-failure'"
                ),
                f"raise PermissionError(13, 'denied', {str(miner_owned_path)!r})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sandbox_app, "_agent_loaded", False)
    monkeypatch.setenv("AGENT_PATH", str(agent_path))
    monkeypatch.delenv("AGENT_MODULE", raising=False)

    try:
        with pytest.raises(PermissionError, match="denied"):
            sandbox_app._load_agent_from_env()
    finally:
        path_type.relative_to = original_relative_to
        if original_loader_owned_helper is missing:
            sandbox_app_dict.pop("_is_loader_mounted_path_os_error", None)
        else:
            sandbox_app._is_loader_mounted_path_os_error = original_loader_owned_helper
        if original_preload_failure_helper is missing:
            sandbox_app_dict.pop("_preload_infrastructure_failure", None)
        else:
            sandbox_app._preload_infrastructure_failure = original_preload_failure_helper


def test_load_agent_from_env_keeps_miner_runtime_os_error_miner_owned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.txt"
    agent_path = tmp_path / "agent.py"
    agent_path.write_text(
        f"open({str(missing_path)!r}, encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sandbox_app, "_agent_loaded", False)
    monkeypatch.setenv("AGENT_PATH", str(agent_path))
    monkeypatch.delenv("AGENT_MODULE", raising=False)

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        sandbox_app._load_agent_from_env()


def test_harness_terminates_long_running_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "sleeper.txt"

    @entrypoint("miner_sleeper_timeout")
    async def sleeper_entrypoint(request: dict[str, object]) -> dict[str, object]:
        del request
        await asyncio.sleep(1)
        marker.write_text("done", encoding="utf-8")
        return {"ok": True}

    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    monkeypatch.setattr(harness_module, "ENTRYPOINT_TIMEOUT_SECONDS", 0.1)

    response = client.post(
        "/entry/miner_sleeper_timeout",
        json={"payload": {}, "context": {}},
    )

    assert response.status_code == 504
    time.sleep(0.3)
    assert not marker.exists()


def test_preload_registers_entrypoint_inside_worker() -> None:
    def preload() -> None:
        if entrypoint_exists("miner_lazy_preload"):
            return

        @entrypoint("miner_lazy_preload")
        async def lazy_entrypoint(request: dict[str, object]) -> dict[str, object]:
            return {
                "message": request.get("message"),
            }

    harness = SandboxHarness(preload=preload)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/miner_lazy_preload",
        json={
            "payload": {"message": "hi"},
            "context": {"tenant": "abc"},
        },
    )

    assert response.status_code == 200
    assert response.json()["result"] == {
        "message": "hi",
    }


def test_entrypoint_key_error_returns_500() -> None:
    @entrypoint("miner_key_error")
    async def key_error_entrypoint(request: dict[str, object]) -> dict[str, object]:
        del request
        raise KeyError("boom")

    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/miner_key_error",
        json={"payload": {}, "context": {}},
    )

    assert response.status_code == 500
