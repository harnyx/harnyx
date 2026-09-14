from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import struct
import threading
from collections.abc import Callable
from pathlib import Path
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

from harnyx_miner_sdk.sandbox_protocol import AdmissionRequest

pytestmark = pytest.mark.anyio("asyncio")

SOURCE = """
import asyncio
import os
import fcntl
import socket
from harnyx_miner_sdk.decorators import entrypoint
counter = 0
@entrypoint("probe")
async def probe(request: dict[str, object]) -> dict[str, object]:
    global counter
    counter += 1
    mode = request.get("mode")
    if mode == "descriptor":
        with socket.socket() as descriptor:
            descriptor.setblocking(False)
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            fcntl.fcntl(descriptor, fcntl.F_SETFL, flags)
            try:
                fcntl.fcntl(descriptor, fcntl.F_SETOWN, os.getpid())
            except PermissionError:
                return {"notification_owner_denied": True, "nonblocking": not descriptor.getblocking()}
            raise AssertionError("signal notification ownership was permitted")
    if mode == "block":
        while True:
            pass
    if mode == "sleep":
        await asyncio.sleep(0.1)
    if mode == "cancel":
        raise asyncio.CancelledError()
    if mode == "crash":
        os._exit(7)
    if mode == "error":
        raise KeyError("miner runtime error")
    return {"pid": os.getpid(), "counter": counter, "file": __file__, "name": __name__}
"""


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


async def test_compiles_once_and_initializes_each_query_independently(harness):
    first = await harness.invoke("probe", body())
    # Replacing mounted source cannot affect an existing compiled generation.
    Path(harness.artifact_path).write_text("this is no longer valid Python!")
    results = await asyncio.gather(*(harness.invoke("probe", body({"mode": "sleep"})) for _ in range(10)))
    assert len({first["pid"], *(result["pid"] for result in results)}) == 11
    assert all(result["counter"] == 1 for result in results)
    assert first["file"] == harness.artifact_path
    assert first["name"] == "<run_path>"


async def test_descriptor_notifications_denied_while_async_io_remains_available(harness):
    assert await harness.invoke("probe", body({"mode": "descriptor"})) == {
        "notification_owner_denied": True,
        "nonblocking": True,
    }
    assert (await harness.invoke("probe", body()))["counter"] == 1


async def test_missing_process_handle_support_fails_before_miner_execution(harness, monkeypatch):
    def unavailable(_pid):
        raise OSError("process handles unavailable")

    monkeypatch.setattr(harness_module, "open_process_handle", unavailable)
    async with asyncio.timeout(5):
        with pytest.raises(HTTPException) as failure:
            await harness.invoke("probe", body())
    assert failure.value.status_code == 500


async def test_timeout_does_not_stop_sibling_or_future_query(harness):
    await harness.invoke("probe", body())
    failed, healthy = await asyncio.gather(
        harness.invoke("probe", body({"mode": "block"}, 0.1)),
        harness.invoke("probe", body({"mode": "sleep"})),
        return_exceptions=True,
    )
    assert isinstance(failed, HTTPException) and failed.status_code == 504
    assert isinstance(healthy, dict)
    assert (await harness.invoke("probe", body()))["counter"] == 1
    assert not harness._compiler.children


@pytest.mark.parametrize("mode", ["cancel", "crash", "error"])
async def test_query_failure_settles_and_does_not_poison_next_query(harness, mode):
    with pytest.raises(HTTPException) as error:
        await harness.invoke("probe", body({"mode": mode}))
    assert error.value.status_code == 500
    assert (await harness.invoke("probe", body()))["counter"] == 1


async def test_compiler_death_reaps_workers_and_next_admission_recovers(harness):
    await harness.invoke("probe", body())
    active = asyncio.create_task(harness.invoke("probe", body({"mode": "block"})))
    while not harness._compiler.children:
        await asyncio.sleep(0.01)
    compiler = harness._compiler
    os.kill(compiler.process.pid, signal.SIGKILL)
    with pytest.raises(HTTPException):
        await asyncio.wait_for(active, 2)
    result = await harness.invoke("probe", body())
    assert result["pid"] > 0
    assert harness._compiler is not compiler


async def test_unused_admission_expires(harness):
    await harness.invoke("probe", body())
    admission = await harness.reserve(AdmissionRequest(limit_seconds=0.05), {})
    await asyncio.sleep(0.06)
    assert admission.reservation_id not in harness._reservations


async def test_duplicate_and_expired_admissions_are_rejected(harness):
    admission = await harness.reserve(AdmissionRequest(limit_seconds=5.0), {})
    request = body({"mode": "sleep"})
    request.admission = admission
    active = asyncio.create_task(harness.invoke("probe", request))
    await asyncio.sleep(0.01)
    with pytest.raises(HTTPException) as error:
        await harness.invoke("probe", request)
    assert error.value.status_code == 409
    await active
    with pytest.raises(HTTPException) as error:
        await harness.invoke("probe", request)
    assert error.value.status_code == 409


@pytest.mark.parametrize(
    "source,code,status",
    [
        ("syntax error!", "PreloadFailed", 500),
        ("raise RuntimeError('initialization error')", "PreloadFailed", 500),
        ("x = 1", "MissingEntrypoint", 404),
        ("raise FileNotFoundError('miner-owned missing file')", "PreloadFailed", 500),
    ],
)
async def test_failure_phase_is_preserved(tmp_path, source, code, status):
    path = tmp_path / "agent.py"
    path.write_text(source)
    runtime = SandboxHarness(artifact_path=str(path))
    try:
        with pytest.raises(HTTPException) as error:
            await runtime.invoke("probe", body())
        assert error.value.status_code == status
        assert error.value.detail["code"] == code
    finally:
        await runtime.close()


async def test_missing_mounted_source_is_infrastructure_failure(tmp_path):
    runtime = SandboxHarness(artifact_path=str(tmp_path / "missing.py"))
    try:
        with pytest.raises(HTTPException) as error:
            await runtime.reserve(AdmissionRequest(limit_seconds=5.0), {})
        assert error.value.detail["code"] == "PreloadInfrastructureFailed"
        assert not runtime._reservations
    finally:
        await runtime.close()


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


async def test_one_cancelled_startup_waiter_does_not_cancel_another(harness):
    first = asyncio.create_task(harness.reserve(AdmissionRequest(limit_seconds=5.0), {}))
    second = asyncio.create_task(harness.reserve(AdmissionRequest(limit_seconds=5.0), {}))
    await asyncio.sleep(0.01)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    admission = await second
    request = body()
    request.admission = admission
    assert (await harness.invoke("probe", request))["pid"] > 0


async def test_missing_source_recovers_on_following_admission(tmp_path):
    path = tmp_path / "agent.py"
    runtime = SandboxHarness(artifact_path=str(path))
    try:
        with pytest.raises(HTTPException):
            await runtime.invoke("probe", body())
        path.write_text(SOURCE)
        assert (await runtime.invoke("probe", body()))["pid"] > 0
    finally:
        await runtime.close()


async def test_tools_remain_bound_to_their_own_query_and_settle_before_return(tmp_path):
    source = """
import asyncio
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk._internal.tool_invoker import _current_tool_invoker
@entrypoint("tool")
async def tool(request: dict[str, object]) -> object:
    invoker = _current_tool_invoker()
    if request.get("detach"):
        asyncio.create_task(invoker.invoke("detached"))
        await asyncio.sleep(.03)
        return "finished"
    return await asyncio.gather(invoker.invoke("first"), invoker.invoke("second"))
"""
    path = tmp_path / "agent.py"
    path.write_text(source)
    calls = []
    closed = []

    class Proxy:
        def __init__(self, token):
            self.token = token

        async def invoke(self, method, *, args=None, kwargs=None):
            await asyncio.sleep(0.1)
            calls.append((self.token, method))
            return self.token

        async def aclose(self):
            closed.append(self.token)

    runtime = SandboxHarness(artifact_path=str(path), tool_factory=lambda config, headers: Proxy(headers["token"]))
    try:
        results = await asyncio.gather(
            *(runtime.invoke("tool", body(), headers={"token": token}) for token in ("dummy-a", "dummy-b"))
        )
        assert results == [["dummy-a", "dummy-a"], ["dummy-b", "dummy-b"]]
        assert await runtime.invoke("tool", body({"detach": True}), headers={"token": "dummy-c"}) == "finished"
        assert ("dummy-c", "detached") in calls
        assert "dummy-c" in closed
    finally:
        await runtime.close()
    assert ("dummy-c", "detached") in calls
    assert sorted(closed) == ["dummy-a", "dummy-b", "dummy-c"]


async def test_cancellation_before_fork_dispatch_does_not_wait_for_nonexistent_child(harness):
    await harness.invoke("probe", body())
    compiler = harness._compiler
    async with compiler.send_lock:
        invocation = asyncio.create_task(harness.invoke("probe", body()))
        while not compiler.children:
            await asyncio.sleep(0.001)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(invocation, 1)
    assert not compiler.children
    assert (await harness.invoke("probe", body()))["pid"] > 0


async def test_tool_factory_failure_does_not_leak_descriptors(harness):
    await harness.invoke("probe", body())
    before = len(os.listdir("/proc/self/fd"))

    def fail(config, headers):
        raise ValueError("unsupported configuration")

    harness._tool_factory = fail
    for _ in range(3):
        with pytest.raises(HTTPException):
            await harness.invoke("probe", body())
    assert len(os.listdir("/proc/self/fd")) == before


async def test_compiler_descriptor_allocation_failure_does_not_poison_following_admission(harness, monkeypatch):
    real_pair = harness_module.socket.socketpair
    first = True

    def fail_once(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise OSError("descriptor allocation failed")
        return real_pair(*args, **kwargs)

    monkeypatch.setattr(harness_module.socket, "socketpair", fail_once)
    with pytest.raises(HTTPException) as error:
        await harness.invoke("probe", body())
    assert error.value.detail["code"] == "PreloadInfrastructureFailed"
    assert (await harness.invoke("probe", body()))["pid"] > 0


async def test_compilation_does_not_inherit_harness_future_flags(tmp_path):
    path = tmp_path / "agent.py"
    path.write_text("""
from harnyx_miner_sdk.decorators import entrypoint

def typed(value: int) -> int:
    return value

@entrypoint("probe")
async def probe(request: dict[str, object]) -> bool:
    return typed.__annotations__["value"] is int
""")
    runtime = SandboxHarness(artifact_path=str(path))
    try:
        assert await runtime.invoke("probe", body()) is True
    finally:
        await runtime.close()


async def test_new_admission_waits_for_retiring_compiler_before_replacement(harness, monkeypatch):
    original_run = harness_module._Compiler._run
    original_close = harness_module._Compiler._close
    closing = asyncio.Event()
    release = asyncio.Event()
    old = None

    async def delayed_start(compiler):
        nonlocal old
        if old is None:
            old = compiler
            await asyncio.Event().wait()
        else:
            await original_run(compiler)

    async def delayed_close(compiler):
        if compiler is old:
            closing.set()
            await release.wait()
        await original_close(compiler)

    monkeypatch.setattr(harness_module._Compiler, "_run", delayed_start)
    monkeypatch.setattr(harness_module._Compiler, "_close", delayed_close)
    first = asyncio.create_task(harness.reserve(AdmissionRequest(limit_seconds=5.0), {}))
    while old is None:
        await asyncio.sleep(0.001)
    first.cancel()
    await closing.wait()
    following = asyncio.create_task(harness.reserve(AdmissionRequest(limit_seconds=5.0), {}))
    await asyncio.sleep(0.01)
    assert not following.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    admission = await asyncio.wait_for(following, 5)
    assert admission.generation != old.generation
    request = body()
    request.admission = admission
    assert (await harness.invoke("probe", request))["pid"] > 0


async def test_slow_worker_decode_keeps_siblings_and_deadlines_responsive(harness, monkeypatch):
    await harness.invoke("probe", body())
    loop = asyncio.get_running_loop()
    decoding = asyncio.Event()
    release = threading.Event()
    decode = harness_module.decode_worker_message
    first = True

    def slow_decode(frame):
        nonlocal first
        if first:
            first = False
            loop.call_soon_threadsafe(decoding.set)
            release.wait(5)
        return decode(frame)

    monkeypatch.setattr(harness_module, "decode_worker_message", slow_decode)
    invocation = asyncio.create_task(harness.invoke("probe", body(seconds=1.0)))
    try:
        await asyncio.wait_for(decoding.wait(), timeout=2)
        sibling = await asyncio.wait_for(harness.invoke("probe", body()), timeout=0.5)
        assert sibling["counter"] == 1
        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(invocation, timeout=2)
        assert error.value.status_code == 504
        assert not release.is_set()
    finally:
        release.set()
        await asyncio.gather(invocation, return_exceptions=True)
    assert (await harness.invoke("probe", body()))["counter"] == 1


async def test_pending_tool_is_cancelled_and_closed_before_timeout_returns(tmp_path):
    source = """
import asyncio
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk._internal.tool_invoker import _current_tool_invoker
@entrypoint("probe")
async def probe(request: dict[str, object]) -> str:
    asyncio.create_task(_current_tool_invoker().invoke("pending"))
    await asyncio.sleep(.05)
    return "result before tool completion"
"""
    path = tmp_path / "agent.py"
    path.write_text(source)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    closed = asyncio.Event()

    class Proxy:
        async def invoke(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def aclose(self):
            assert cancelled.is_set()
            closed.set()

    runtime = SandboxHarness(artifact_path=str(path), tool_factory=lambda config, headers: Proxy())
    try:
        async with asyncio.timeout(5):
            with pytest.raises(HTTPException) as error:
                await runtime.invoke("probe", body(seconds=2.0))
        assert error.value.status_code == 504
        assert started.is_set() and cancelled.is_set() and closed.is_set()
        assert not runtime._compiler.children
    finally:
        await runtime.close()


async def test_timeout_terminates_worker_when_compiler_cannot_process_commands(harness):
    await harness.invoke("probe", body())
    compiler = harness._compiler
    invocation = asyncio.create_task(harness.invoke("probe", body({"mode": "block"}, seconds=1.0)))
    async with asyncio.timeout(3):
        while not compiler.children or not all(child.spawned.done() for child in compiler.children.values()):
            await asyncio.sleep(0.01)
    os.kill(compiler.process.pid, signal.SIGSTOP)
    try:
        async with asyncio.timeout(6):
            with pytest.raises(HTTPException) as error:
                await invocation
        assert error.value.status_code == 504
        assert compiler.process.poll() is not None
        assert not compiler.children
        assert (await harness.invoke("probe", body()))["counter"] == 1
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(compiler.process.pid, signal.SIGCONT)
        await asyncio.gather(invocation, return_exceptions=True)


async def test_cancellation_during_activation_retires_gated_worker(harness, monkeypatch):
    await harness.invoke("probe", body())
    compiler = harness._compiler
    original_command = compiler.command
    activation = asyncio.Event()

    async def delay_activation(value, fds=()):
        if value["kind"] == "activate":
            activation.set()
            await asyncio.Future()
        await original_command(value, fds)

    monkeypatch.setattr(compiler, "command", delay_activation)
    invocation = asyncio.create_task(harness.invoke("probe", body()))
    async with asyncio.timeout(6):
        await activation.wait()
        handles = [child.pidfd for child in compiler.children.values()]
        assert handles and all(handle >= 0 for handle in handles)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
    assert not compiler.children
    for handle in handles:
        with pytest.raises(OSError):
            os.fstat(handle)
    assert (await harness.invoke("probe", body()))["counter"] == 1


async def test_cleanup_deadline_reports_uncertainty(monkeypatch, harness):
    await harness.invoke("probe", body())
    compiler = harness._compiler
    original_cleanup = compiler._close_processes

    async def stalled_cleanup():
        await asyncio.Future()

    monkeypatch.setattr(compiler, "_close_processes", stalled_cleanup)
    monkeypatch.setattr(harness_module, "WORKER_KILL_GRACE_SECONDS", 0.01)
    try:
        async with asyncio.timeout(1):
            with pytest.raises(harness_module.SandboxCleanupUnconfirmedError):
                await compiler._close()
    finally:
        # The simulated uncertainty never substitutes for physical cleanup.
        monkeypatch.setattr(compiler, "_close_processes", original_cleanup)
        await original_cleanup()


async def test_module_initialization_can_run_its_own_event_loop(tmp_path):
    path = tmp_path / "agent.py"
    path.write_text("""
import asyncio
from harnyx_miner_sdk.decorators import entrypoint
async def setup():
    await asyncio.sleep(0)
    return 42
initialized = asyncio.run(setup())
@entrypoint("probe")
async def probe(request: dict[str, object]) -> object:
    return initialized
""")
    runtime = SandboxHarness(artifact_path=str(path))
    try:
        assert await runtime.invoke("probe", body()) == 42
    finally:
        await runtime.close()


@pytest.mark.parametrize("text", ["partial output", "unicode 🙂", "invalid \ud800"])
async def test_worker_print_delivers_unterminated_output(tmp_path, capfd, text):
    path = tmp_path / "agent.py"
    path.write_text(
        "import sys\nfrom harnyx_miner_sdk.decorators import entrypoint\n"
        "@entrypoint('probe')\nasync def probe(request: dict[str, object]) -> object:\n"
        f"    print({text!r}, end='')\n"
        f"    print({text!r}, end='', file=sys.stderr)\n"
        "    return 'ok'\n"
    )
    runtime = SandboxHarness(artifact_path=str(path))
    try:
        assert await runtime.invoke(
            "probe", body(), headers={"x-session-id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}
        ) == "ok"
    finally:
        await runtime.close()
    records = [
        json.loads(line.removeprefix("HARNYX_SANDBOX_INVOCATION_OUTPUT "))
        for line in capfd.readouterr().out.splitlines()
        if line.startswith("HARNYX_SANDBOX_INVOCATION_OUTPUT ")
    ]
    expected = text.encode("utf-8", errors="replace").decode("utf-8")
    for stream in ("stdout", "stderr"):
        assert "".join(record["message"] for record in records if record["stream"] == stream) == expected


async def test_blocking_module_initialization_remains_subject_to_query_deadline(tmp_path):
    path = tmp_path / "agent.py"
    path.write_text("while True:\n    pass\n")
    runtime = SandboxHarness(artifact_path=str(path))
    try:
        admission = await runtime.reserve(AdmissionRequest(limit_seconds=5.0), {})
        runtime.release(admission)
        async with asyncio.timeout(5):
            with pytest.raises(HTTPException) as error:
                await runtime.invoke("probe", body(seconds=0.3))
        assert error.value.status_code == 504
        assert error.value.detail["error"] == "entrypoint deadline exceeded"
        assert not runtime._compiler.children
    finally:
        await runtime.close()
