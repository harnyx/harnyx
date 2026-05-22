"""Utilities for binding agent entrypoints to a FastAPI sandbox."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import inspect
import logging
import multiprocessing
import os
import pickle
import struct
import traceback
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import pyseccomp as seccomp
from fastapi import APIRouter, HTTPException, Request

from harnyx_miner_sdk._internal.tool_invoker import bind_tool_invoker
from harnyx_miner_sdk.decorators import (
    EntrypointRegistry,
    get_entrypoint,
    get_entrypoint_registry,
)
from harnyx_miner_sdk.sandbox_headers import read_session_id_header
from harnyx_sandbox.context.snapshot import ContextSnapshot
from harnyx_sandbox.sandbox.timeout import ENTRYPOINT_TIMEOUT_SECONDS

ToolConfig = Mapping[str, Any] | None
ToolHeaders = Mapping[str, str]
ToolFactory = Callable[[ToolConfig, ToolHeaders], Any]


@dataclass
class EntrypointRequest:
    payload: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    tool_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class SandboxPreloadFailure:
    code: str
    error: str
    exception: str


class MpContext(Protocol):
    def Process(  # noqa: N802 - mirror multiprocessing
        self,
        *,
        target: Callable[..., Any] | None = None,
        args: tuple[Any, ...] = ...,
    ) -> multiprocessing.Process: ...


logger = logging.getLogger("harnyx_sandbox.sandbox")
WORKER_KILL_GRACE_SECONDS = 1.0
WORKER_RESULT_HEADER_BYTES = 8
WORKER_RESULT_READ_CHUNK_BYTES = 64 * 1024
MAX_WORKER_RESULT_BYTES = 64 * 1024 * 1024


class WorkerResultProtocolError(RuntimeError):
    """Raised when the parent observes invalid worker result framing."""


@dataclass(frozen=True)
class WorkerResultPipe:
    read_fd: int
    write_fd: int

    @classmethod
    def open(cls) -> WorkerResultPipe:
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(read_fd, False)
        except BaseException:
            _close_fd(read_fd)
            _close_fd(write_fd)
            raise
        return cls(read_fd=read_fd, write_fd=write_fd)

    def close_read(self) -> None:
        _close_fd(self.read_fd)

    def close_write(self) -> None:
        _close_fd(self.write_fd)

    def close(self) -> None:
        self.close_read()
        self.close_write()


@dataclass(frozen=True)
class WorkerResultFrame:
    payload: bytes | None = None
    oversized: bool = False

    @property
    def complete(self) -> bool:
        return self.payload is not None

def _default_mp_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-Unix platforms
        return multiprocessing.get_context()


class WorkerResultReader:
    """Reads a worker result pipe without occupying one executor thread per worker."""

    def __init__(self, *, process: multiprocessing.Process, pipe: WorkerResultPipe) -> None:
        self._process = process
        self._pipe = pipe
        self._buffer = bytearray()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._future: asyncio.Future[tuple[str, Any]] | None = None
        self._decode_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def wait(self, *, timeout: float) -> tuple[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, Any]] = loop.create_future()
        self._loop = loop
        self._future = future
        try:
            self._add_reader(self._pipe.read_fd, self._result_fd_ready)
            self._add_reader(self._process.sentinel, self._process_exited)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            self._remove_reader(self._pipe.read_fd)
            self._remove_reader(self._process.sentinel)
        if self._decode_task is not None and not self._decode_task.done():
            self._decode_task.cancel()
        self._pipe.close_read()

    def _add_reader(self, fd: int, callback: Callable[[], None]) -> None:
        if self._loop is None:  # pragma: no cover - defensive guard
            raise WorkerResultProtocolError("worker result reader was not started")
        self._loop.add_reader(fd, callback)

    def _remove_reader(self, fd: int) -> None:
        if self._loop is None:
            return
        with contextlib.suppress(Exception):
            self._loop.remove_reader(fd)

    def _result_fd_ready(self) -> None:
        self._drain_available_result()

    def _process_exited(self) -> None:
        self._drain_available_result()
        if self._decode_task is not None:
            return
        self._set_exception(WorkerResultProtocolError("entrypoint worker exited before returning result"))

    def _drain_available_result(self) -> None:
        while not self._closed and self._decode_task is None:
            try:
                chunk = os.read(self._pipe.read_fd, WORKER_RESULT_READ_CHUNK_BYTES)
            except BlockingIOError:
                return
            except OSError as exc:
                self._set_exception(WorkerResultProtocolError(f"failed to read worker result pipe: {exc}"))
                return
            if chunk == b"":
                self._set_exception(WorkerResultProtocolError("worker closed result pipe before complete result"))
                return
            self._buffer.extend(chunk)
            frame = _try_extract_worker_result_frame(self._buffer)
            if frame.oversized:
                self._set_exception(WorkerResultProtocolError("worker result frame exceeded maximum size"))
                return
            if frame.complete:
                self._start_decode(frame.payload)
                return

    def _start_decode(self, payload: bytes | None) -> None:
        if payload is None or self._loop is None:
            return
        self._remove_reader(self._pipe.read_fd)
        self._remove_reader(self._process.sentinel)
        self._decode_task = self._loop.create_task(self._decode_complete_frame(payload))

    async def _decode_complete_frame(self, payload: bytes) -> None:
        try:
            envelope = await asyncio.to_thread(pickle.loads, payload)
        except Exception:
            self._set_exception(WorkerResultProtocolError("worker returned invalid result frame"))
            return
        if not _is_worker_result_envelope(envelope):
            self._set_exception(WorkerResultProtocolError("worker returned invalid result envelope"))
            return
        self._set_result(envelope)

    def _set_result(self, result: tuple[str, Any]) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(result)

    def _set_exception(self, exc: Exception) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_exception(exc)


class SandboxHarness:
    """Coordinates entrypoint invocation for sandboxed agents."""

    def __init__(
        self,
        *,
        registry: EntrypointRegistry | None = None,
        tool_factory: ToolFactory | None = None,
        preload: Callable[[], SandboxPreloadFailure | None] | None = None,
    ) -> None:
        self._registry = registry or get_entrypoint_registry()
        self._tool_factory = tool_factory
        self._preload = preload
        self._mp: MpContext = cast(MpContext, _default_mp_context())

    async def invoke(
        self,
        entrypoint_name: str,
        body: EntrypointRequest,
        *,
        headers: ToolHeaders | None = None,
    ) -> Any:
        request_payload = body.payload
        tool_config = body.tool_config
        context_snapshot = ContextSnapshot(body.context or {})

        call_kwargs = {
            "entrypoint_name": entrypoint_name,
            "request_payload": request_payload,
            "context": context_snapshot.to_dict(),
            "tool_config": tool_config,
            "headers": dict(headers or {}),
            "preload": self._preload,
        }

        return await self._invoke_with_worker(call_kwargs)

    def create_router(self) -> APIRouter:
        """Return a FastAPI router exposing entrypoint invocation endpoints."""
        router = APIRouter()

        @router.post(
            "/{entrypoint_name}",
            tags=["entrypoints"],
            description="Invoke a registered entrypoint by name in a sandboxed worker process.",
        )
        async def dispatch(
            entrypoint_name: str,
            body: EntrypointRequest,
            request: Request,
        ) -> dict[str, Any]:
            headers = request.headers
            try:
                result = await self.invoke(entrypoint_name, body, headers=headers)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except HTTPException:
                raise
            except Exception as exc:
                session_id = read_session_id_header(headers)
                logger.exception(
                    "sandbox entrypoint failed",
                    extra={
                        "entrypoint": entrypoint_name,
                        "session_id": session_id,
                    },
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": str(exc),
                        "exception": exc.__class__.__name__,
                    },
                ) from exc
            return {"ok": True, "result": result}

        return router

    @staticmethod
    def _build_call_kwargs(
        func: Callable[..., Any],
        request_payload: Any,
        context_snapshot: ContextSnapshot,
        tool_proxy: Any,
    ) -> dict[str, Any]:
        del func, context_snapshot, tool_proxy
        return {"request": request_payload}

    async def _invoke_with_worker(self, payload: Mapping[str, Any]) -> Any:
        process, result_pipe = self._spawn_worker(payload)
        try:
            result_kind, result_data = await self._await_worker_result(result_pipe, payload, process)
            return self._unwrap_worker_result(result_kind, result_data)
        finally:
            self._join_process(process)

    def _spawn_worker(self, payload: Mapping[str, Any]) -> tuple[multiprocessing.Process, WorkerResultPipe]:
        result_pipe = WorkerResultPipe.open()
        process = self._mp.Process(
            target=_entrypoint_worker,
            args=(
                payload["entrypoint_name"],
                payload["request_payload"],
                payload["context"],
                payload["tool_config"],
                payload["headers"],
                self._tool_factory,
                payload["preload"],
                result_pipe.read_fd,
                result_pipe.write_fd,
            ),
        )
        try:
            process.start()
        except BaseException:
            result_pipe.close()
            raise
        result_pipe.close_write()
        return process, result_pipe

    def _unwrap_worker_result(self, kind: str, data: Any) -> Any:
        if kind == "ok":
            return data

        detail = data if isinstance(data, Mapping) else {"error": "entrypoint failed"}
        code = detail.get("code") if isinstance(detail, Mapping) else None
        if code == "MissingEntrypoint":
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    async def _await_worker_result(
        self,
        result_pipe: WorkerResultPipe,
        payload: Mapping[str, Any],
        process: multiprocessing.Process,
    ) -> tuple[str, Any]:
        reader = WorkerResultReader(process=process, pipe=result_pipe)
        try:
            return await reader.wait(timeout=ENTRYPOINT_TIMEOUT_SECONDS)
        except TimeoutError as exc:  # pragma: no cover - integration timing
            return self._handle_timeout(process, payload, exc)
        except Exception as exc:  # pragma: no cover - unexpected worker failure
            return self._handle_worker_failure(process, exc)

    def _terminate_process(self, process: multiprocessing.Process) -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(WORKER_KILL_GRACE_SECONDS)
        if process.is_alive():  # pragma: no cover - guardrail
            process.kill()

    def _handle_timeout(
        self,
        process: multiprocessing.Process,
        payload: Mapping[str, Any],
        exc: TimeoutError,
    ) -> tuple[str, Any]:
        self._terminate_process(process)
        session_id = read_session_id_header(payload["headers"])
        logger.exception(
            "sandbox entrypoint timed out",
            extra={
                "entrypoint": payload["entrypoint_name"],
                "session_id": session_id,
                "timeout_seconds": ENTRYPOINT_TIMEOUT_SECONDS,
            },
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": f"entrypoint exceeded {ENTRYPOINT_TIMEOUT_SECONDS}s",
                "exception": "TimeoutError",
            },
        ) from exc

    def _handle_worker_failure(
        self,
        process: multiprocessing.Process,
        exc: Exception,
    ) -> tuple[str, Any]:
        self._terminate_process(process)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "entrypoint worker failed",
                "exception": exc.__class__.__name__,
            },
        ) from exc

    def _join_process(self, process: multiprocessing.Process) -> None:
        process.join(WORKER_KILL_GRACE_SECONDS)
        if process.is_alive():  # pragma: no cover - guardrail
            process.kill()


def _entrypoint_worker(
    entrypoint_name: str,
    request_payload: Mapping[str, Any],
    context_data: Mapping[str, Any],
    tool_config: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    tool_factory: ToolFactory | None,
    preload: Callable[[], SandboxPreloadFailure | None] | None,
    read_fd: int,
    result_fd: int,
) -> None:
    tool_proxy = None
    preload_completed = False
    try:
        _close_fd(read_fd)
        if tool_factory is not None:
            # Build the proxy before seccomp so hostname resolution/client setup
            # cannot trigger blocked task-creation syscalls inside the worker.
            tool_proxy = tool_factory(tool_config, headers)
        _block_new_tasks_in_this_process()
        if preload is not None:
            try:
                preload_failure = preload()
            except BaseException as exc:
                _send_worker_error(result_fd, "PreloadFailed", exc)
                return
            if preload_failure is not None:
                _send_preload_failure(result_fd, preload_failure)
                return
            preload_completed = True
        try:
            func = get_entrypoint(entrypoint_name)
        except KeyError as exc:
            detail_code = "MissingEntrypoint" if preload_completed else "EntrypointUnavailable"
            _send_worker_error(result_fd, detail_code, exc)
            return
        context_snapshot = ContextSnapshot(context_data or {})
        call_kwargs = SandboxHarness._build_call_kwargs(
            func,
            request_payload,
            context_snapshot,
            tool_proxy,
        )
        if tool_proxy is not None:
            with bind_tool_invoker(tool_proxy):
                result = _execute_entrypoint(func, call_kwargs)
        else:
            result = _execute_entrypoint(func, call_kwargs)
        _send_worker_result(result_fd, ("ok", result))
    except BaseException as exc:  # pragma: no cover - propagated to parent
        _send_worker_error(result_fd, "UnhandledException", exc)
    finally:
        if tool_proxy is not None:
            with contextlib.suppress(Exception):
                asyncio.run(tool_proxy.aclose())
        _close_fd(result_fd)


def _block_new_tasks_in_this_process() -> None:
    """Install a seccomp filter that denies task-creation syscalls."""

    filter_ = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    for name in ("clone", "clone3", "fork", "vfork", "execve", "execveat"):
        filter_.add_rule(seccomp.ERRNO(errno.EPERM), name)
    filter_.load()
    logger.debug("worker seccomp filter installed", extra={"pid": os.getpid()})


def _try_extract_worker_result_frame(buffer: bytearray) -> WorkerResultFrame:
    if len(buffer) < WORKER_RESULT_HEADER_BYTES:
        return WorkerResultFrame()
    payload_size = struct.unpack(">Q", buffer[:WORKER_RESULT_HEADER_BYTES])[0]
    if payload_size > MAX_WORKER_RESULT_BYTES:
        return WorkerResultFrame(oversized=True)
    frame_size = WORKER_RESULT_HEADER_BYTES + payload_size
    if len(buffer) < frame_size:
        return WorkerResultFrame()
    return WorkerResultFrame(payload=bytes(buffer[WORKER_RESULT_HEADER_BYTES:frame_size]))


def _is_worker_result_envelope(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)


def _send_worker_result(result_fd: int, result: tuple[str, Any]) -> None:
    payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_WORKER_RESULT_BYTES:
        payload = pickle.dumps(
            (
                "error",
                {
                    "code": "ResultTooLarge",
                    "error": "worker result exceeded maximum size",
                    "exception": "ResultTooLarge",
                    "traceback": None,
                },
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    _write_all(result_fd, struct.pack(">Q", len(payload)) + payload)


def _send_worker_error(result_fd: int, code: str, exc: BaseException) -> None:
    _send_worker_result(
        result_fd,
        (
            "error",
            {
                "code": code,
                "error": str(exc),
                "exception": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            },
        ),
    )


def _send_preload_failure(result_fd: int, failure: SandboxPreloadFailure) -> None:
    _send_worker_result(
        result_fd,
        (
            "error",
            {
                "code": failure.code,
                "error": failure.error,
                "exception": failure.exception,
                "traceback": None,
            },
        ),
    )


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _close_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.close(fd)


def _execute_entrypoint(func: Callable[..., Any], call_kwargs: Mapping[str, Any]) -> Any:
    if not inspect.iscoroutinefunction(func):
        raise RuntimeError("sandbox entrypoints must be async def")
    coroutine = cast(Coroutine[Any, Any, Any], func(**call_kwargs))
    return asyncio.run(coroutine)


__all__ = [
    "EntrypointRequest",
    "MAX_WORKER_RESULT_BYTES",
    "SandboxHarness",
    "SandboxPreloadFailure",
    "ToolConfig",
    "ToolFactory",
    "ToolHeaders",
    "WorkerResultPipe",
    "WorkerResultProtocolError",
    "WorkerResultReader",
]
