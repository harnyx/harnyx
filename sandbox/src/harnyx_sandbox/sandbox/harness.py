"""Admission and trusted supervision for compile-once, isolated query workers."""

from __future__ import annotations

import array
import asyncio
import codecs
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.sandbox_headers import read_session_id_header
from harnyx_miner_sdk.sandbox_protocol import AdmissionRequest, SandboxAdmission
from harnyx_miner_sdk.tools.http_models import ToolBudgetDTO
from harnyx_miner_sdk.tools.time_budget import ExecutionTimeBudgetDTO
from harnyx_sandbox.sandbox.compiler import open_process_handle, protect_process, signal_process_handle
from harnyx_sandbox.sandbox.worker_protocol import (
    MAX_FRAME_BYTES,
    Invocation,
    QueryResult,
    ToolCall,
    ToolReply,
    WorkerError,
    WorkerResultProtocolError,
    decode_worker_message,
    read_frame,
    send,
)

ToolConfig = Mapping[str, Any] | None
ToolHeaders = Mapping[str, str]
ToolFactory = Callable[[ToolConfig, ToolHeaders], Any]
logger = logging.getLogger("harnyx_sandbox.sandbox")
WORKER_KILL_GRACE_SECONDS = 1.0
WORKER_RESULT_READ_CHUNK_BYTES = 64 * 1024
MAX_WORKER_RESULT_BYTES = MAX_FRAME_BYTES
MAX_WORKER_OUTPUT_MESSAGE_CHARACTERS = 64 * 1024
MAX_WORKER_OUTPUT_RECORDS_PER_CALLBACK = 64


class _EntrypointContext(BaseModel):
    """Validated sandbox-boundary context for every entrypoint invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    time_budget: ExecutionTimeBudgetDTO
    cost_budget: ToolBudgetDTO | None = None


@dataclass
class EntrypointRequest:
    context: _EntrypointContext
    payload: dict[str, Any] = field(default_factory=dict)
    tool_config: dict[str, Any] | None = None
    admission: SandboxAdmission | None = None


@dataclass
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
        self.read_fd = -1

    def close_write(self) -> None:
        _close_fd(self.write_fd)
        self.write_fd = -1

    def close(self) -> None:
        self.close_read()
        self.close_write()


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
        prefix = "HARNYX_SANDBOX_INVOCATION_OUTPUT "
        if not self._session_id:
            record.pop("session_id")
            prefix = "HARNYX_SANDBOX_ARTIFACT_OUTPUT "
        print(
            prefix + json.dumps(record, separators=(",", ":")),
            flush=True,
        )


@dataclass
class _Child:
    spawned: asyncio.Future[int]
    exited: asyncio.Future[int]
    requested: bool = False
    pidfd: int = -1


class SandboxCleanupUnconfirmedError(RuntimeError):
    """The host must retire a container whose process cleanup could not finish."""


class _Compiler:
    """Own a fresh compiler process and its forked children, never their credentials."""

    def __init__(self, artifact_path: str) -> None:
        self.generation = uuid4().hex
        self.path = artifact_path
        self.process: subprocess.Popen[bytes] | None = None
        self.control: socket.socket | None = None
        self.children: dict[str, _Child] = {}
        self.ready: asyncio.Future[WorkerError | None] = asyncio.get_running_loop().create_future()
        self.failure: Exception | None = None
        self.send_lock = asyncio.Lock()
        self.outputs: list[asyncio.Task[None]] = []
        self.task = asyncio.create_task(self._run())
        self.close_task: asyncio.Task[None] | None = None

    async def command(self, value: dict[str, object], fds: tuple[int, ...] = ()) -> None:
        control = self.control
        if control is None or self.failure is not None:
            raise RuntimeError("compiler unavailable")
        data = json.dumps(value).encode()
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))] if fds else []
        async with self.send_lock:
            while True:
                try:
                    control.sendmsg([data], ancillary)
                    return
                except BlockingIOError:
                    loop = asyncio.get_running_loop()
                    writable = loop.create_future()
                    loop.add_writer(control.fileno(), _wake, writable)
                    try:
                        await writable
                    finally:
                        loop.remove_writer(control.fileno())

    async def fork(self, key: str, fds: tuple[int, ...]) -> _Child:
        loop = asyncio.get_running_loop()
        child = _Child(loop.create_future(), loop.create_future())
        self.children[key] = child
        await self.command({"kind": "fork", "id": key}, fds)
        child.requested = True
        await asyncio.shield(child.spawned)
        return child

    async def stop_child(self, key: str) -> None:
        child = self.children.get(key)
        if child is None:
            return
        if not child.requested:
            self.children.pop(key, None)
            return
        try:
            if child.pidfd < 0 or self.failure is not None:
                await self.close()
                return
            loop = asyncio.get_running_loop()
            exited = loop.create_future()
            loop.add_reader(child.pidfd, _wake, exited)
            try:
                with contextlib.suppress(ProcessLookupError):
                    signal_process_handle(child.pidfd, signal.SIGTERM)
                try:
                    await asyncio.wait_for(asyncio.shield(exited), WORKER_KILL_GRACE_SECONDS)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        signal_process_handle(child.pidfd, signal.SIGKILL)
                    await asyncio.wait_for(asyncio.shield(exited), WORKER_KILL_GRACE_SECONDS)
            finally:
                loop.remove_reader(child.pidfd)
            try:
                await asyncio.wait_for(asyncio.shield(child.exited), WORKER_KILL_GRACE_SECONDS)
            except TimeoutError:
                await self.close()
        except TimeoutError:
            await self.close()
        finally:
            # Cancellation can abandon the shielded registration waiter before
            # generation retirement completes it with an exception.
            if child.spawned.done() and not child.spawned.cancelled():
                child.spawned.exception()
            _close_fd(child.pidfd)
            child.pidfd = -1
            self.children.pop(key, None)

    async def _receive(self, control: socket.socket) -> tuple[bytes, list[int]]:
        loop = asyncio.get_running_loop()
        while True:
            try:
                data, ancillary, flags, _address = control.recvmsg(65536, socket.CMSG_SPACE(array.array("i").itemsize))
            except BlockingIOError:
                readable = loop.create_future()
                loop.add_reader(control.fileno(), _wake, readable)
                try:
                    await readable
                finally:
                    loop.remove_reader(control.fileno())
                continue
            fds = []
            for level, kind, payload in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    received = array.array("i")
                    received.frombytes(payload)
                    fds.extend(received)
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                for fd in fds:
                    os.close(fd)
                raise RuntimeError("truncated compiler response")
            return data, fds

    async def _run(self) -> None:
        child = None
        pipes: list[WorkerOutputPipe] = []
        try:
            capability_handle = open_process_handle(os.getpid())
            try:
                signal_process_handle(capability_handle, 0)
            finally:
                os.close(capability_handle)
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.control = parent
            stdout = WorkerOutputPipe.open()
            pipes.append(stdout)
            stderr = WorkerOutputPipe.open()
            pipes.append(stderr)
            protect_process(subreaper=True)
            self.process = subprocess.Popen(  # noqa: S603 — fixed interpreter/module, no shell
                [sys.executable, "-I", "-m", "harnyx_sandbox.sandbox.compiler", str(child.fileno()), self.path],
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                stdin=subprocess.DEVNULL,
                stdout=stdout.write_fd,
                stderr=stderr.write_fd,
                pass_fds=(child.fileno(),),
                start_new_session=True,
            )
            child.close()
            stdout.close_write()
            stderr.close_write()
            parent.setblocking(False)
            for pipe, stream in ((stdout, "stdout"), (stderr, "stderr")):
                self.outputs.append(
                    asyncio.create_task(
                        WorkerOutputReader(
                            pipe=pipe, payload={"headers": {}, "entrypoint_name": "compile"}, stream=stream
                        ).wait()
                    )
                )
            while True:
                data, received_fds = await self._receive(parent)
                if not data:
                    raise RuntimeError("compiler connection closed")
                try:
                    reply = json.loads(data)  # Compiler is trusted and never executes miner code.
                    if reply["kind"] == "ready":
                        logger.info(
                            "artifact compiled",
                            extra={"compile_seconds": reply["compile_seconds"], "generation": self.generation},
                        )
                        self.ready.set_result(None)
                    elif reply["kind"] == "failed":
                        if reply["code"] == "PreloadInfrastructureFailed":
                            self.failure = RuntimeError(reply["error"])
                        self.ready.set_result(
                            WorkerError(code=reply["code"], exception=reply["exception"], message=reply["error"])
                        )
                        # Keep the classified compile failure available for invocation reporting.
                        return
                    else:
                        owned = self.children.get(reply["id"])
                        if owned is not None:
                            if reply["kind"] == "spawned":
                                if len(received_fds) != 1 or owned.pidfd >= 0:
                                    raise RuntimeError("invalid worker process handle")
                                owned.pidfd = received_fds.pop()
                                await self.command({"kind": "activate", "id": reply["id"]})
                            future = owned.spawned if reply["kind"] == "spawned" else owned.exited
                            if not future.done():
                                future.set_result(reply["pid"] if reply["kind"] == "spawned" else reply["status"])
                finally:
                    for fd in received_fds:
                        os.close(fd)
        except Exception as exc:
            self.failure = exc
            if not self.ready.done():
                self.ready.set_result(
                    WorkerError(code="PreloadInfrastructureFailed", exception=type(exc).__name__, message=str(exc))
                )
            for owned in self.children.values():
                if not owned.spawned.done():
                    owned.spawned.set_exception(exc)
            if self.close_task is None:
                self.close_task = asyncio.create_task(self._close())
        finally:
            if child is not None:
                child.close()
            for pipe in pipes:
                pipe.close_write()
                if not self.outputs:
                    pipe.close_read()

    async def close(self) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close())
        await SandboxHarness._await_owned_cleanup(self.close_task)

    async def _close(self) -> None:
        try:
            async with asyncio.timeout(4 * WORKER_KILL_GRACE_SECONDS):
                await self._close_processes()
        except TimeoutError as exc:
            raise SandboxCleanupUnconfirmedError("sandbox process cleanup remains unconfirmed") from exc

    async def _close_processes(self) -> None:
        if not self.ready.done():
            self.ready.set_result(
                WorkerError(code="PreloadInfrastructureFailed", exception="RuntimeError", message="compiler stopped")
            )
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task
        if self.control is not None:
            self.control.close()
        process = self.process
        if process is not None:
            # Workers cannot change process groups; subreaper adopts them if compiler dies.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + WORKER_KILL_GRACE_SECONDS
            while process.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            while process.poll() is None:
                await asyncio.sleep(0.01)
            while True:
                try:
                    pid, _status = os.waitpid(-process.pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    await asyncio.sleep(0.01)
        for owned in self.children.values():
            if not owned.spawned.done():
                owned.spawned.set_exception(RuntimeError("compiler generation retired before worker activation"))
            if not owned.exited.done():
                owned.exited.set_result(-1)
        for task in self.outputs:
            with contextlib.suppress(Exception):
                await task


def _wake(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


@dataclass
class _Reservation:
    admission: SandboxAdmission
    expiry: asyncio.TimerHandle
    activated: bool = False


class SandboxHarness:
    """Authorize each query by its private connection, outside miner memory."""

    def __init__(self, *, artifact_path: str | None = None, tool_factory: ToolFactory | None = None) -> None:
        self.artifact_path = artifact_path
        self._tool_factory = tool_factory
        self._compiler: _Compiler | None = None
        self._reservations: dict[str, _Reservation] = {}
        self._closing = False
        self._invocations: set[asyncio.Task[Any]] = set()
        self._startup_waiters = 0

    async def reserve(self, request: AdmissionRequest, headers: ToolHeaders) -> SandboxAdmission:
        del headers  # Admission does not carry credentials into the compiler.
        if self._closing:
            raise HTTPException(503, detail={"error": "sandbox closing"})
        began = time.monotonic_ns()
        compiler = self._compiler
        if compiler is not None and (compiler.failure is not None or compiler.close_task is not None):
            await compiler.close()
            if self._compiler is compiler:
                self._compiler = None
            compiler = self._compiler
        if self._closing:
            raise HTTPException(503, detail={"error": "sandbox closing"})
        if compiler is None:
            path = self.artifact_path if self.artifact_path is not None else os.getenv("AGENT_PATH", "")
            compiler = self._compiler = _Compiler(path)
        self._startup_waiters += 1
        try:
            failure = await asyncio.wait_for(asyncio.shield(compiler.ready), request.limit_seconds)
            if failure is not None and failure.code == "PreloadInfrastructureFailed":
                self._unwrap_result(QueryResult(error=failure))
        except TimeoutError as exc:
            raise HTTPException(
                504, detail={"error": "admission deadline exceeded", "exception": "TimeoutError"}
            ) from exc
        finally:
            self._startup_waiters -= 1
            if not compiler.ready.done() and self._startup_waiters == 0:
                await compiler.close()
                if self._compiler is compiler:
                    self._compiler = None
        start = began if request.include_wait_in_budget else time.monotonic_ns()
        if self._closing:
            raise HTTPException(503, detail={"error": "sandbox closing"})
        admission = SandboxAdmission(
            reservation_id=uuid4().hex,
            generation=compiler.generation,
            deadline_monotonic_ns=start + int(request.limit_seconds * 1_000_000_000),
        )
        expiry = asyncio.get_running_loop().call_later(
            max(0, (admission.deadline_monotonic_ns - time.monotonic_ns()) / 1e9), self.release, admission
        )
        self._reservations[admission.reservation_id] = _Reservation(admission, expiry)
        return admission

    def release(self, admission: SandboxAdmission) -> None:
        item = self._reservations.get(admission.reservation_id)
        if item is not None and item.admission == admission:
            item.expiry.cancel()
            self._reservations.pop(admission.reservation_id)

    async def close(self) -> None:
        self._closing = True
        for item in tuple(self._reservations.values()):
            self.release(item.admission)
        for task in tuple(self._invocations):
            task.cancel()
        if self._compiler is not None:
            await self._compiler.close()
        await asyncio.gather(*self._invocations, return_exceptions=True)

    async def invoke(self, entrypoint_name: str, body: EntrypointRequest, *, headers: ToolHeaders | None = None) -> Any:
        context_data = body.context.model_dump(mode="json", exclude_none=True)
        if entrypoint_name == "query":
            try:
                ContextSnapshot.model_validate(context_data)
            except ValidationError as exc:
                raise HTTPException(422, detail=exc.errors(include_input=False)) from exc
        admission = body.admission or await self.reserve(
            AdmissionRequest(limit_seconds=body.context.time_budget.limit_seconds), {}
        )
        compiler = self._compiler
        item = self._reservations.get(admission.reservation_id)
        if (
            compiler is None
            or compiler.generation != admission.generation
            or item is None
            or item.admission != admission
        ):
            raise HTTPException(409, detail={"error": "expired admission"})
        if item.activated:
            raise HTTPException(409, detail={"error": "admission already consumed"})
        item.activated = True
        item.expiry.cancel()
        task = asyncio.current_task()
        if task is not None:
            self._invocations.add(task)
        try:
            failure = compiler.ready.result()
            if failure is not None:
                return self._unwrap_result(QueryResult(error=failure))
            remaining = (admission.deadline_monotonic_ns - time.monotonic_ns()) / 1e9
            async with asyncio.timeout(max(0, remaining)):
                return await self._invoke_child(
                    compiler,
                    admission,
                    Invocation(entrypoint=entrypoint_name, payload=body.payload, context=context_data),
                    body.tool_config,
                    headers or {},
                )
        except TimeoutError as exc:
            raise HTTPException(
                504, detail={"error": "entrypoint deadline exceeded", "exception": "TimeoutError"}
            ) from exc
        except (HTTPException, SandboxCleanupUnconfirmedError):
            raise
        except Exception as exc:
            raise HTTPException(
                500, detail={"error": "entrypoint worker failed", "exception": type(exc).__name__}
            ) from exc
        finally:
            self.release(admission)
            if task is not None:
                self._invocations.discard(task)

    async def _invoke_child(
        self,
        compiler: _Compiler,
        admission: SandboxAdmission,
        request: Invocation,
        config: ToolConfig,
        headers: ToolHeaders,
    ) -> Any:
        with contextlib.ExitStack() as resources:
            parent, child = socket.socketpair()
            resources.callback(parent.close)
            resources.callback(child.close)
            stdout = WorkerOutputPipe.open()
            resources.callback(stdout.close)
            stderr = WorkerOutputPipe.open()
            resources.callback(stderr.close)
            writer = None
            outputs: list[asyncio.Task[None]] = []
            calls: set[asyncio.Task[None]] = set()
            proxy = None
            active = True
            try:
                for pipe, stream in ((stdout, "stdout"), (stderr, "stderr")):
                    outputs.append(
                        asyncio.create_task(
                            WorkerOutputReader(
                                pipe=pipe,
                                payload={"headers": headers, "entrypoint_name": request.entrypoint},
                                stream=stream,
                            ).wait()
                        )
                    )
                proxy = self._tool_factory(config, headers) if self._tool_factory else None
                await compiler.fork(admission.reservation_id, (child.fileno(), stdout.write_fd, stderr.write_fd))
                child.close()
                stdout.close_write()
                stderr.close_write()
                parent.setblocking(False)
                reader, writer = await asyncio.open_connection(sock=parent)
                await send(writer, request)
                seen: set[int] = set()
                write_lock = asyncio.Lock()

                async def execute(call: ToolCall) -> None:
                    try:
                        if proxy is None:
                            raise RuntimeError("no tool invoker bound in this context")
                        result = await proxy.invoke(call.method, args=call.args, kwargs=call.kwargs)
                        reply = ToolReply(call_id=call.call_id, result=result)
                    except Exception as exc:
                        reply = ToolReply(call_id=call.call_id, error=str(exc))
                    if active:
                        async with write_lock:
                            with contextlib.suppress(ConnectionError, WorkerResultProtocolError):
                                await send(writer, reply)

                while True:
                    frame = await read_frame(reader)
                    message = await asyncio.to_thread(decode_worker_message, frame)
                    if isinstance(message, QueryResult):
                        if message.error is not None and message.error.code not in {
                            "PreloadFailed",
                            "MissingEntrypoint",
                            "UnhandledException",
                        }:
                            raise WorkerResultProtocolError("worker cannot report infrastructure failures")
                        active = False
                        await asyncio.gather(*calls, return_exceptions=True)
                        return self._unwrap_result(message)
                    if message.call_id in seen:
                        raise WorkerResultProtocolError("duplicate tool call id")
                    seen.add(message.call_id)
                    call_task = asyncio.create_task(execute(message))
                    calls.add(call_task)
                    call_task.add_done_callback(calls.discard)
            finally:
                active = False
                if writer is not None:
                    writer.close()
                else:
                    parent.close()
                child.close()
                stdout.close_write()
                stderr.close_write()

                for call_task in calls:
                    call_task.cancel()

                async def cleanup() -> None:
                    try:
                        await compiler.stop_child(admission.reservation_id)
                        await asyncio.gather(*outputs, *calls, return_exceptions=True)
                    finally:
                        if proxy is not None:
                            await proxy.aclose()

                await self._await_owned_cleanup(asyncio.create_task(cleanup()))

    @staticmethod
    def _unwrap_result(result: QueryResult) -> Any:
        if result.error is None:
            return result.result
        error = result.error
        raise HTTPException(
            404 if error.code == "MissingEntrypoint" else 500,
            detail={"code": error.code, "exception": error.exception, "error": error.message},
        )

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
            except (HTTPException, SandboxCleanupUnconfirmedError):
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


def _close_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.close(fd)
