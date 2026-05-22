from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import pickle
import struct
import time
from collections.abc import Mapping
from pathlib import Path

import harnyx_sandbox.app as sandbox_app
import harnyx_sandbox.sandbox.harness as harness_module
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from harnyx_sandbox.sandbox.harness import (
    SandboxHarness,
    SandboxPreloadFailure,
    WorkerResultPipe,
    WorkerResultReader,
    _send_worker_result,
)

from harnyx_miner_sdk.api import test_tool as invoke_test_tool
from harnyx_miner_sdk.decorators import clear_entrypoints, entrypoint, entrypoint_exists


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

    _send_worker_result(pipe.write_fd, ("ok", b"x" * 2000))
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
    payload = pickle.dumps(("ok", {"value": "at-limit"}), protocol=pickle.HIGHEST_PROTOCOL)
    monkeypatch.setattr(harness_module, "MAX_WORKER_RESULT_BYTES", len(payload))
    os.write(pipe.write_fd, struct.pack(">Q", len(payload)) + payload)
    os.close(pipe.write_fd)
    process.finish()

    try:
        result = await WorkerResultReader(process=process, pipe=pipe).wait(timeout=1.0)
    finally:
        process.close()

    assert result == ("ok", {"value": "at-limit"})


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
