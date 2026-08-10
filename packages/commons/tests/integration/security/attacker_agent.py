"""Adversarial sandbox agent used by security tests."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


class _MaliciousResult:
    def __init__(self, marker_url: str) -> None:
        self._marker_url = marker_url

    def __reduce__(self):
        urllib.request.urlopen(self._marker_url, timeout=2).close()  # noqa: S310 - fixed test-server URL
        return str, ("executed",)


@entrypoint("query")
async def query(request: Query) -> Response:
    return Response(text=request.text)


@entrypoint("probe")
async def probe(request: Mapping[str, Any]) -> Any:
    mode = request.get("mode")
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
        _reach_result_ipc_barrier(barrier_url, barrier_id, role)
        if role == "malicious":
            marker_url = _result_ipc_url(barrier_url, barrier_id, "reduce")
            return _MaliciousResult(marker_url)
        return {"role": "healthy"}
    return {"error": f"unknown mode {mode!r}"}


def _try_write(path: str) -> bool | str:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        return True
    except Exception as exc:  # pragma: no cover - exercised in docker tests
        return f"err:{exc.__class__.__name__}"


def _reach_result_ipc_barrier(barrier_url: str, barrier_id: str, role: str) -> None:
    urllib.request.urlopen(  # noqa: S310 - URL points to the test-owned host server
        _result_ipc_url(barrier_url, barrier_id, role),
        timeout=4,
    ).close()


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
