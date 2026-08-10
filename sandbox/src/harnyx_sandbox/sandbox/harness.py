"""Utilities for binding agent entrypoints to a FastAPI sandbox."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import errno
import inspect
import json
import logging
import math
import multiprocessing
import os
import struct
import sys
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import pyseccomp as seccomp
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

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
MAX_WORKER_OUTPUT_MESSAGE_CHARACTERS = 64 * 1024
MAX_WORKER_OUTPUT_RECORDS_PER_CALLBACK = 64


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
class WorkerOutputPipe:
    read_fd: int
    write_fd: int

    @classmethod
    def open(cls) -> WorkerOutputPipe:
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
            envelope = await asyncio.to_thread(_decode_worker_result, payload)
        except Exception:
            self._set_exception(WorkerResultProtocolError("worker returned invalid result frame"))
            return
        self._set_result(envelope)

    def _set_result(self, result: tuple[str, Any]) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(result)

    def _set_exception(self, exc: Exception) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_exception(exc)


class WorkerOutputReader:
    """Forwards one worker output stream without occupying an executor thread."""

    def __init__(
        self,
        *,
        pipe: WorkerOutputPipe,
        payload: Mapping[str, Any],
        stream: str,
    ) -> None:
        self._pipe = pipe
        self._session_id = str(read_session_id_header(payload["headers"]))
        self._entrypoint = str(payload["entrypoint_name"])
        self._stream = stream
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._future: asyncio.Future[None] | None = None
        self._closed = False
        self._eof = False
        self._continuation_scheduled = False

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._future = loop.create_future()
        try:
            loop.add_reader(self._pipe.read_fd, self._output_ready)
            await self._future
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            with contextlib.suppress(Exception):
                self._loop.remove_reader(self._pipe.read_fd)
        self._pipe.close_read()

    def _output_ready(self) -> None:
        if self._closed:
            return
        remaining_records = self._emit_available(MAX_WORKER_OUTPUT_RECORDS_PER_CALLBACK)
        if remaining_records and not self._eof:
            try:
                chunk = os.read(self._pipe.read_fd, WORKER_RESULT_READ_CHUNK_BYTES)
            except BlockingIOError:
                chunk = None
            except OSError:
                chunk = b""
            if chunk == b"":
                self._eof = True
                if self._loop is not None:
                    self._loop.remove_reader(self._pipe.read_fd)
                self._buffer += self._decoder.decode(b"", final=True)
            elif chunk is not None:
                self._buffer += self._decoder.decode(chunk)
            remaining_records = self._emit_available(remaining_records)

        if self._has_available_record():
            self._schedule_continuation()
        elif self._eof and self._future is not None and not self._future.done():
            self._future.set_result(None)

    def _emit_available(self, record_budget: int) -> int:
        while record_budget and (message := self._take_next_message()) is not None:
            self._emit(message)
            record_budget -= 1
        return record_budget

    def _take_next_message(self) -> str | None:
        newline_index = self._buffer.find("\n")
        if 0 <= newline_index <= MAX_WORKER_OUTPUT_MESSAGE_CHARACTERS:
            line = self._buffer[:newline_index]
            self._buffer = self._buffer[newline_index + 1 :]
            return line.rstrip("\r")
        if len(self._buffer) >= MAX_WORKER_OUTPUT_MESSAGE_CHARACTERS:
            line = self._buffer[:MAX_WORKER_OUTPUT_MESSAGE_CHARACTERS]
            self._buffer = self._buffer[MAX_WORKER_OUTPUT_MESSAGE_CHARACTERS:]
            return line
        if self._eof and self._buffer:
            line = self._buffer
            self._buffer = ""
            return line.rstrip("\r")
        return None

    def _has_available_record(self) -> bool:
        return (
            "\n" in self._buffer
            or len(self._buffer) >= MAX_WORKER_OUTPUT_MESSAGE_CHARACTERS
            or (self._eof and bool(self._buffer))
        )

    def _schedule_continuation(self) -> None:
        if self._continuation_scheduled or self._loop is None:
            return
        self._continuation_scheduled = True
        self._loop.call_soon(self._continue_output)

    def _continue_output(self) -> None:
        self._continuation_scheduled = False
        self._output_ready()

    def _emit(self, line: str) -> None:
        record = {
            "session_id": self._session_id,
            "entrypoint": self._entrypoint,
            "stream": self._stream,
            "message": line,
        }
        print(
            "HARNYX_SANDBOX_INVOCATION_OUTPUT " + json.dumps(record, separators=(",", ":")),
            flush=True,
        )


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
        process, result_pipe, stdout_pipe, stderr_pipe = self._spawn_worker(payload)
        output_tasks = (
            asyncio.create_task(WorkerOutputReader(pipe=stdout_pipe, payload=payload, stream="stdout").wait()),
            asyncio.create_task(WorkerOutputReader(pipe=stderr_pipe, payload=payload, stream="stderr").wait()),
        )
        try:
            result_kind, result_data = await self._await_worker_result(result_pipe, payload, process)
            return self._unwrap_worker_result(result_kind, result_data)
        finally:
            cleanup_task = asyncio.create_task(self._finish_worker(process, output_tasks))
            await self._await_owned_cleanup(cleanup_task)

    async def _finish_worker(
        self,
        process: multiprocessing.Process,
        output_tasks: tuple[asyncio.Task[None], asyncio.Task[None]],
    ) -> None:
        join_task = asyncio.create_task(asyncio.to_thread(self._join_process, process))
        await asyncio.gather(*output_tasks, return_exceptions=True)
        await join_task

    @staticmethod
    async def _await_owned_cleanup(task: asyncio.Task[None]) -> None:
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if task.cancelled():
                    break
                cancellation = exc
            except BaseException:
                break
        try:
            task.result()
        except BaseException as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
        if cancellation is not None:
            raise cancellation

    def _spawn_worker(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[multiprocessing.Process, WorkerResultPipe, WorkerOutputPipe, WorkerOutputPipe]:
        acquired_pipes: list[WorkerResultPipe | WorkerOutputPipe] = []
        try:
            result_pipe = WorkerResultPipe.open()
            acquired_pipes.append(result_pipe)
            stdout_pipe = WorkerOutputPipe.open()
            acquired_pipes.append(stdout_pipe)
            stderr_pipe = WorkerOutputPipe.open()
            acquired_pipes.append(stderr_pipe)
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
                    stdout_pipe.read_fd,
                    stdout_pipe.write_fd,
                    stderr_pipe.read_fd,
                    stderr_pipe.write_fd,
                ),
            )
            process.start()
        except BaseException:
            for pipe in acquired_pipes:
                pipe.close()
            raise
        result_pipe.close_write()
        stdout_pipe.close_write()
        stderr_pipe.close_write()
        return process, result_pipe, stdout_pipe, stderr_pipe

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
            terminate_task = asyncio.create_task(asyncio.to_thread(self._terminate_process, process))
            await self._await_owned_cleanup(terminate_task)
            return self._handle_timeout(payload, exc)
        except Exception as exc:  # pragma: no cover - unexpected worker failure
            terminate_task = asyncio.create_task(asyncio.to_thread(self._terminate_process, process))
            await self._await_owned_cleanup(terminate_task)
            return self._handle_worker_failure(exc)

    def _terminate_process(self, process: multiprocessing.Process) -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(WORKER_KILL_GRACE_SECONDS)
        if process.is_alive():  # pragma: no cover - guardrail
            process.kill()

    def _handle_timeout(
        self,
        payload: Mapping[str, Any],
        exc: TimeoutError,
    ) -> tuple[str, Any]:
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
        exc: Exception,
    ) -> tuple[str, Any]:
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
    stdout_read_fd: int,
    stdout_write_fd: int,
    stderr_read_fd: int,
    stderr_write_fd: int,
) -> None:
    tool_proxy = None
    preload_completed = False
    try:
        _close_fd(read_fd)
        _close_fd(stdout_read_fd)
        _close_fd(stderr_read_fd)
        os.dup2(stdout_write_fd, 1)
        os.dup2(stderr_write_fd, 2)
        _close_fd(stdout_write_fd)
        _close_fd(stderr_write_fd)
        sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
        sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
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


def _send_worker_result(result_fd: int, result: tuple[str, Any]) -> None:
    kind, data = result
    if kind == "ok":
        try:
            normalized = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
            _validate_exact_json_value(normalized)
            envelope: dict[str, object] = {"status": "ok", "result": normalized}
            payload = _encode_worker_result(envelope)
        except BaseException as exc:
            payload = _encode_worker_error(
                code="UnhandledException",
                exception=exc.__class__.__name__,
                message=str(exc),
            )
    elif kind == "error" and type(data) is dict:
        payload = _encode_worker_error(
            code=_required_error_string(data, "code"),
            exception=_required_error_string(data, "exception"),
            message=_required_error_string(data, "error"),
        )
    else:
        payload = _encode_worker_error(
            code="UnhandledException",
            exception="WorkerResultProtocolError",
            message="worker produced an invalid result envelope",
        )
    if len(payload) > MAX_WORKER_RESULT_BYTES:
        payload = _encode_worker_error(
            code="ResultTooLarge",
            exception="ResultTooLarge",
            message="worker result exceeded maximum size",
        )
    _write_all(result_fd, struct.pack(">Q", len(payload)) + payload)


def _encode_worker_result(envelope: object) -> bytes:
    return json.dumps(
        envelope,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _encode_worker_error(*, code: str, exception: str, message: str) -> bytes:
    return _encode_worker_result(
        {
            "status": "error",
            "error": {
                "code": code,
                "exception": exception,
                "message": message,
            },
        }
    )


def _required_error_string(data: dict[object, object], key: str) -> str:
    value = data.get(key)
    if type(value) is not str:
        raise WorkerResultProtocolError(f"worker error {key} must be a string")
    return value


def _validate_exact_json_value(value: object) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, str, int}:
        return
    if value_type is float:
        if not math.isfinite(cast(float, value)):
            raise TypeError("worker result contains a non-finite number")
        return
    if value_type is list:
        for item in cast(list[object], value):
            _validate_exact_json_value(item)
        return
    if value_type is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError("worker result object keys must be strings")
            _validate_exact_json_value(item)
        return
    raise TypeError(f"worker result contains unsupported {value_type.__name__}")


def _decode_worker_result(payload: bytes) -> tuple[str, Any]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"invalid JSON constant: {value}")

    def reject_duplicate_names(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object name: {key}")
            result[key] = value
        return result

    envelope = json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_names,
    )
    if type(envelope) is not dict:
        raise WorkerResultProtocolError("worker result envelope must be an object")
    status = envelope.get("status")
    if status == "ok" and set(envelope) == {"status", "result"}:
        result = envelope["result"]
        _validate_exact_json_value(result)
        return "ok", result
    if status == "error" and set(envelope) == {"status", "error"}:
        error = envelope["error"]
        if type(error) is not dict or set(error) != {"code", "exception", "message"}:
            raise WorkerResultProtocolError("worker error must match the result protocol")
        code = _required_error_string(error, "code")
        exception = _required_error_string(error, "exception")
        message = _required_error_string(error, "message")
        return "error", {"code": code, "exception": exception, "error": message}
    raise WorkerResultProtocolError("worker returned invalid result envelope")


def _send_worker_error(result_fd: int, code: str, exc: BaseException) -> None:
    _send_worker_result(
        result_fd,
        (
            "error",
            {
                "code": code,
                "error": str(exc),
                "exception": exc.__class__.__name__,
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
