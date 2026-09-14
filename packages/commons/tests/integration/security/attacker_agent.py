"""Adversarial sandbox agent used by security tests."""

from __future__ import annotations

import asyncio
import ctypes
import errno
import json
import os
import resource
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pyseccomp

from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


class _MaliciousResult:
    def __init__(self, marker_url: str) -> None:
        self._marker_url = marker_url

    def __reduce__(self):
        urllib.request.urlopen(self._marker_url, timeout=2).close()  # noqa: S310 - fixed test-server URL
        return str, ("executed",)


@entrypoint("query")
async def query(request: Query, _context: ContextSnapshot) -> Response:
    return Response(text=request.text)


@entrypoint("probe")
async def probe(request: Mapping[str, Any]) -> Any:
    mode = request.get("mode")
    if mode == "peer_resources":
        return _probe_peer_resources()
    if mode == "signals":
        return _probe_signals()
    if mode == "peer_access":
        return _probe_peer_access()
    if mode == "forge_frame":
        from harnyx_miner_sdk._internal.tool_invoker import _current_tool_invoker

        invoker = _current_tool_invoker()
        frame = json.dumps(request["frame"]).encode()
        invoker.writer.write(struct.pack(">Q", len(frame)) + frame)
        await invoker.writer.drain()
        await asyncio.sleep(10)
    if mode == "fs":
        return {
            "ok_tmp": _try_write_temp(),
            "err_root": _try_write(str(ROOT_BLOCKED_TARGET)),
        }
    if mode == "pids":
        return {"spawned": _spawn_until_failure(_pids_probe_limit())}
    if mode == "sleep":
        await asyncio.sleep(int(request.get("secs", 999)))
        return {"done": True}
    if mode == "result_ipc_overlap":
        barrier_id = str(request["barrier_id"])
        role = str(request["role"])
        barrier_url = str(request["barrier_url"])
        await _reach_result_ipc_barrier(barrier_url, barrier_id, role)
        if role == "malicious":
            marker_url = _result_ipc_url(barrier_url, barrier_id, "reduce")
            return _MaliciousResult(marker_url)
        return {"role": "healthy"}
    return {"error": f"unknown mode {mode!r}"}


def _probe_peer_access() -> dict[str, object]:
    peers = [int(name) for name in os.listdir("/proc") if name.isdecimal() and int(name) != os.getpid()]
    readable = []
    for pid in peers:
        for suffix in ("mem", "environ", "maps", "fd/1"):
            try:
                fd = os.open(f"/proc/{pid}/{suffix}", os.O_RDONLY)
            except OSError:
                continue
            else:
                os.close(fd)
                readable.append(f"{pid}/{suffix}")
    left, right = socket.socketpair()
    try:
        try:
            left.sendmsg([b"probe"])
            descriptor_transfer = True
        except PermissionError:
            descriptor_transfer = False
    finally:
        left.close()
        right.close()
    libc = ctypes.CDLL(None, use_errno=True)
    reset_dumpable = libc.prctl(4, 1, 0, 0, 0) == 0
    return {
        "peers": len(peers),
        "readable": readable,
        "descriptor_transfer": descriptor_transfer,
        "reset_dumpable": reset_dumpable,
    }


def _try_write(path: str) -> bool | str:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        return True
    except Exception as exc:  # pragma: no cover - exercised in docker tests
        return f"err:{exc.__class__.__name__}"


async def _reach_result_ipc_barrier(barrier_url: str, barrier_id: str, role: str) -> None:
    async with httpx.AsyncClient(trust_env=False, timeout=4) as client:
        response = await client.get(_result_ipc_url(barrier_url, barrier_id, role))
        response.raise_for_status()


def _result_ipc_url(barrier_url: str, barrier_id: str, role: str) -> str:
    query = urllib.parse.urlencode({"barrier_id": barrier_id, "role": role})
    return f"{barrier_url}?{query}"


def _try_write_temp() -> bool:
    try:
        target = Path(tempfile.gettempdir()) / "ok"
    except Exception:
        return False
    return _try_write(str(target)) is True


def _spawn_until_failure(limit: int) -> int | str:
    procs: list[subprocess.Popen[str]] = []
    try:
        for _ in range(limit):
            procs.append(
                subprocess.Popen(  # noqa: S603 - command is fixed for stress testing
                    [sys.executable, "-c", "import time; time.sleep(2)"],
                    text=True,
                ),
            )
        return len(procs)
    except Exception as exc:  # pragma: no cover - exercised in docker tests
        return f"err:{exc.__class__.__name__}"
    finally:
        for proc in procs:
            try:
                proc.terminate()
            except Exception:  # pragma: no cover - cleanup best effort
                proc.kill()


def _pids_probe_limit() -> int:
    raw_limit = os.getenv("SANDBOX_PIDS_PROBE_LIMIT")
    if raw_limit is None:
        return 800
    return int(raw_limit)


ROOT_BLOCKED_TARGET = Path("/root/blocked")


def _probe_signals() -> dict[str, object]:
    pid = os.getpid()
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    blocked = {}
    attempts = {
        "kill_parent": ("kill", (parent, signal.SIGSTOP)),
        "kill_group": ("kill", (0, 0)),
        "kill_all": ("kill", (-1, 0)),
        "kill_parent_group": ("kill", (-parent, 0)),
        "tkill": ("tkill", (parent, 0)),
        "tgkill": ("tgkill", (parent, parent, 0)),
        "rt_sigqueueinfo": ("rt_sigqueueinfo", (parent, 0, 0)),
        "rt_tgsigqueueinfo": ("rt_tgsigqueueinfo", (parent, parent, 0, 0)),
        "pidfd_send_signal": ("pidfd_send_signal", (-1, 0, 0, 0)),
    }
    for label, (name, args) in attempts.items():
        number = pyseccomp.resolve_syscall(pyseccomp.Arch.NATIVE, name)
        ctypes.set_errno(0)
        result = libc.syscall(number, *args)
        blocked[label] = result == -1 and ctypes.get_errno() == errno.EPERM
    os.kill(pid, 0)
    if libc.tgkill(pid, pid, 0) != 0:
        raise RuntimeError("self signal denied")
    return {"blocked": blocked, "self_signal": True}


def _probe_peer_resources() -> dict[str, bool]:
    parent = os.getppid()
    blocked = {}
    # Harmless reads and unchanged settings still exercise target permissions.
    operations = {
        "limits": lambda: resource.prlimit(parent, resource.RLIMIT_NOFILE),
        "affinity": lambda: os.sched_setaffinity(parent, os.sched_getaffinity(parent)),
        "scheduler": lambda: os.sched_setscheduler(parent, os.sched_getscheduler(parent), os.sched_getparam(parent)),
        "priority": lambda: os.setpriority(os.PRIO_PROCESS, parent, 19),
    }
    for name, operation in operations.items():
        try:
            operation()
        except PermissionError:
            blocked[name] = True
        else:
            blocked[name] = False
    own_limits = resource.prlimit(0, resource.RLIMIT_NOFILE)
    resource.prlimit(0, resource.RLIMIT_NOFILE, own_limits)
    os.sched_setaffinity(0, os.sched_getaffinity(0))
    return blocked
