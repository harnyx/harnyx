"""Clean compiler and fork parent. Never execute miner code in this process."""

from __future__ import annotations

import array
import asyncio
import contextlib
import ctypes
import errno
import fcntl
import io
import json
import os
import select
import signal
import socket
import sys
import termios
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyseccomp as seccomp
from pydantic import BaseModel

from harnyx_miner_sdk._internal.tool_invoker import bind_tool_invoker
from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.decorators import get_entrypoint
from harnyx_miner_sdk.tools.proxy import ToolInvocationError
from harnyx_sandbox.sandbox.worker_protocol import (
    Invocation,
    QueryResult,
    ToolCall,
    ToolReply,
    WorkerError,
    load_json,
    read_frame,
    send,
)


def open_process_handle(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "cannot open worker process handle")
    os.set_inheritable(fd, False)
    return fd


def signal_process_handle(fd: int, number: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.pidfd_send_signal(fd, number, None, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot signal worker process handle")


def protect_process(*, subreaper: bool = False) -> None:
    """Prevent same-UID processes reading this process's memory and descriptors."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise OSError(ctypes.get_errno(), "cannot disable process dumping")
    if subreaper and libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot enable orphan reaping")


def restrict_worker() -> None:
    protect_process()
    # Every operation is denied unless this runtime explicitly needs it.
    rules = seccomp.SyscallFilter(defaction=seccomp.ERRNO(errno.EPERM))
    for name in (
        "read",
        "write",
        "readv",
        "writev",
        "pread64",
        "pwrite64",
        "close",
        "close_range",
        "fstat",
        "newfstatat",
        "stat",
        "lstat",
        "statx",
        "lseek",
        "getdents64",
        "open",
        "openat",
        "access",
        "faccessat",
        "faccessat2",
        "readlink",
        "readlinkat",
        "getcwd",
        "chdir",
        "fchdir",
        "dup",
        "dup2",
        "dup3",
        "mmap",
        "mprotect",
        "munmap",
        "mremap",
        "madvise",
        "brk",
        "msync",
        "futex",
        "set_robust_list",
        "rseq",
        "arch_prctl",
        "rt_sigaction",
        "rt_sigprocmask",
        "rt_sigreturn",
        "sigaltstack",
        "getpid",
        "getppid",
        "gettid",
        "getuid",
        "geteuid",
        "getgid",
        "getegid",
        "getgroups",
        "clock_gettime",
        "clock_getres",
        "clock_nanosleep",
        "nanosleep",
        "gettimeofday",
        "time",
        "getrandom",
        "uname",
        "sysinfo",
        "getrusage",
        "times",
        "getrlimit",
        "setrlimit",
        "sched_yield",
        "sched_getaffinity",
        "sched_getscheduler",
        "sched_getparam",
        "poll",
        "ppoll",
        "select",
        "pselect6",
        "epoll_create",
        "epoll_create1",
        "epoll_ctl",
        "epoll_wait",
        "epoll_pwait",
        "eventfd",
        "eventfd2",
        "pipe",
        "pipe2",
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "getsockname",
        "getpeername",
        "getsockopt",
        "setsockopt",
        "sendto",
        "recvfrom",
        "shutdown",
        "mkdir",
        "mkdirat",
        "unlink",
        "unlinkat",
        "rmdir",
        "rename",
        "renameat",
        "renameat2",
        "ftruncate",
        "truncate",
        "fsync",
        "fdatasync",
        "flock",
        "fchmod",
        "chmod",
        "umask",
        "utimensat",
        "statfs",
        "fstatfs",
        "exit",
        "exit_group",
        "restart_syscall",
    ):
        rules.add_rule(seccomp.ALLOW, name)
    # Descriptor ownership and async notification commands can signal peers
    # without calling kill. Permit only ordinary descriptor operations.
    for command in (
        fcntl.F_DUPFD,
        fcntl.F_DUPFD_CLOEXEC,
        fcntl.F_GETFD,
        fcntl.F_SETFD,
        fcntl.F_GETFL,
        fcntl.F_GETLK,
        fcntl.F_SETLK,
        fcntl.F_SETLKW,
    ):
        rules.add_rule(seccomp.ALLOW, "fcntl", seccomp.Arg(1, seccomp.EQ, command))
    rules.add_rule(
        seccomp.ALLOW,
        "fcntl",
        seccomp.Arg(1, seccomp.EQ, fcntl.F_SETFL),
        seccomp.Arg(2, seccomp.MASKED_EQ, os.O_ASYNC, 0),
    )
    for command in (termios.TCGETS, termios.TIOCGWINSZ, termios.FIONREAD, termios.FIONBIO):
        rules.add_rule(seccomp.ALLOW, "ioctl", seccomp.Arg(1, seccomp.EQ, command))
    pid = os.getpid()
    for name in ("kill", "tkill", "rt_sigqueueinfo"):
        rules.add_rule(seccomp.ALLOW, name, seccomp.Arg(0, seccomp.EQ, pid))
    for name in ("tgkill", "rt_tgsigqueueinfo"):
        rules.add_rule(seccomp.ALLOW, name, seccomp.Arg(0, seccomp.EQ, pid), seccomp.Arg(1, seccomp.EQ, pid))
    for name in ("prlimit64", "sched_setaffinity", "sched_setscheduler", "sched_setparam", "sched_setattr"):
        for target in (0, pid):
            rules.add_rule(seccomp.ALLOW, name, seccomp.Arg(0, seccomp.EQ, target))
    for target in (0, pid):
        rules.add_rule(seccomp.ALLOW, "setpriority", seccomp.Arg(0, seccomp.EQ, 0), seccomp.Arg(1, seccomp.EQ, target))
    rules.load()


class ConnectionInvoker:
    """SDK invoker with authority supplied solely by its host-side connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.calls: dict[int, asyncio.Future[Any]] = {}
        self.next_call = 0
        self.write_lock = asyncio.Lock()

    async def invoke(
        self, method: str, *, args: Sequence[Any] | None = None, kwargs: Mapping[str, Any] | None = None
    ) -> Any:
        call_id = self.next_call
        self.next_call += 1
        future = asyncio.get_running_loop().create_future()
        self.calls[call_id] = future
        try:
            async with self.write_lock:
                await send(
                    self.writer,
                    ToolCall(call_id=call_id, method=method, args=list(args or ()), kwargs=dict(kwargs or {})),
                )
            return await future
        finally:
            self.calls.pop(call_id, None)

    async def receive(self) -> None:
        try:
            while True:
                reply = ToolReply.model_validate(load_json(await read_frame(self.reader)))
                future = self.calls.get(reply.call_id)
                if future is not None and not future.done():
                    if reply.error is not None:
                        future.set_exception(ToolInvocationError(reply.error))
                    else:
                        future.set_result(reply.result)
        except Exception as exc:
            for future in self.calls.values():
                if not future.done():
                    future.set_exception(exc)


def initialize_module(code: types.CodeType, path: str) -> WorkerError | None:
    """Initialize in the query child, before a harness event loop is running."""
    try:
        # Match runpy.run_path's script namespace without compiling a second time.
        module = types.ModuleType("<run_path>")
        module.__dict__.update(__file__=path, __cached__=None, __loader__=None, __package__="", __spec__=None)
        sys.modules["<run_path>"] = module
        sys.argv[0] = path
        exec(code, module.__dict__)  # noqa: S102 — only in the isolated query child
    except BaseException as exc:
        return WorkerError(code="PreloadFailed", exception=type(exc).__name__, message=str(exc)[:65536])
    return None


async def run_query(channel: socket.socket, preload_failure: WorkerError | None) -> None:
    channel.setblocking(False)
    reader, writer = await asyncio.open_connection(sock=channel)
    receiver = None
    phase = "PreloadFailed"
    try:
        request = Invocation.model_validate(load_json(await read_frame(reader)))
        if preload_failure is not None:
            await send(writer, QueryResult(error=preload_failure))
            return
        phase = "MissingEntrypoint"
        func = get_entrypoint(request.entrypoint)
        phase = "UnhandledException"
        invoker = ConnectionInvoker(reader, writer)
        receiver = asyncio.create_task(invoker.receive())
        with bind_tool_invoker(invoker):
            if request.entrypoint == "query":
                result = await func(request=request.payload, context=ContextSnapshot.model_validate(request.context))
            else:
                result = await func(request=request.payload)
        if isinstance(result, BaseModel):
            result = result.model_dump(mode="json")
        await send(writer, QueryResult(result=result))
    except BaseException as exc:
        with contextlib.suppress(Exception):
            await send(
                writer,
                QueryResult(error=WorkerError(code=phase, exception=type(exc).__name__, message=str(exc)[:65536])),
            )
    finally:
        if receiver is not None:
            receiver.cancel()
        writer.close()


def child_main(fds: list[int], code: types.CodeType, path: str) -> None:
    channel_fd, stdout_fd, stderr_fd = fds
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    for value in os.listdir("/proc/self/fd"):
        fd = int(value)
        if fd not in (channel_fd, 1, 2):
            with contextlib.suppress(OSError):
                os.close(fd)
    # Both layers must be unbuffered: os._exit deliberately skips Python cleanup.
    sys.stdout = io.TextIOWrapper(
        os.fdopen(1, "wb", buffering=0, closefd=False), encoding="utf-8", errors="replace", write_through=True
    )
    sys.stderr = io.TextIOWrapper(
        os.fdopen(2, "wb", buffering=0, closefd=False), encoding="utf-8", errors="replace", write_through=True
    )
    try:
        restrict_worker()
        preload_failure = initialize_module(code, path)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_query(socket.socket(fileno=channel_fd), preload_failure))
    finally:
        # No asyncio.run shutdown: miner tasks may suppress cancellation forever.
        os._exit(0)


def main() -> None:
    control = socket.socket(fileno=int(sys.argv[1]))
    path = sys.argv[2]
    protect_process()
    started = time.monotonic()
    try:
        source = Path(path).read_bytes()
        code = compile(source, path, "exec", dont_inherit=True)
        del source
    except Exception as exc:
        failure = "PreloadInfrastructureFailed" if isinstance(exc, OSError) else "PreloadFailed"
        control.send(
            json.dumps({"kind": "failed", "code": failure, "exception": type(exc).__name__, "error": str(exc)}).encode()
        )
        return
    control.send(json.dumps({"kind": "ready", "compile_seconds": time.monotonic() - started}).encode())
    children: dict[str, int] = {}
    gates: dict[str, int] = {}
    try:
        while True:
            for key, pid in list(children.items()):
                exited, status = os.waitpid(pid, os.WNOHANG)
                if exited:
                    children.pop(key)
                    if key in gates:
                        os.close(gates.pop(key))
                    control.send(json.dumps({"kind": "exited", "id": key, "status": status}).encode())
            if not select.select([control], [], [], 0.02)[0]:
                continue
            data, ancillary, _flags, _address = control.recvmsg(4096, socket.CMSG_SPACE(3 * array.array("i").itemsize))
            if not data:
                return
            command = json.loads(data)
            if command["kind"] == "stop":
                return
            if command["kind"] == "activate":
                gate = gates.pop(command["id"], None)
                if gate is not None:
                    try:
                        os.write(gate, b"1")
                    finally:
                        os.close(gate)
                continue
            fds: list[int] = []
            for level, kind, payload in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    received = array.array("i")
                    received.frombytes(payload)
                    fds.extend(received)
            if command["kind"] != "fork" or len(fds) != 3:
                raise RuntimeError("invalid trusted compiler command")
            gate_read, gate_write = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(gate_write)
                try:
                    if os.read(gate_read, 1) != b"1":
                        os._exit(0)
                finally:
                    os.close(gate_read)
                child_main(fds, code, path)
            os.close(gate_read)
            gates[command["id"]] = gate_write
            for fd in fds:
                os.close(fd)
            children[command["id"]] = pid
            pidfd = open_process_handle(pid)
            try:
                control.sendmsg(
                    [json.dumps({"kind": "spawned", "id": command["id"], "pid": pid}).encode()],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [pidfd]))],
                )
            finally:
                os.close(pidfd)

    finally:
        for gate in gates.values():
            os.close(gate)
        for pid in children.values():
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        for pid in children.values():
            os.waitpid(pid, 0)
        control.close()


if __name__ == "__main__":
    main()
