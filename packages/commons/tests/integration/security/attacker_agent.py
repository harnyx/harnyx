"""Adversarial sandbox agent used by security tests."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from caster_miner_sdk.decorators import entrypoint


@entrypoint("probe")
async def probe(request: Mapping[str, Any]) -> dict[str, Any]:
    mode = request.get("mode")
    if mode == "fs":
        return {
            "ok_tmp": _try_write(str(TMP_WRITE_TARGET)),
            "err_root": _try_write(str(ROOT_BLOCKED_TARGET)),
        }
    if mode == "pids":
        return {"spawned": _spawn_until_failure(400)}
    if mode == "sleep":
        await asyncio.sleep(int(request.get("secs", 999)))
        return {"done": True}
    return {"error": f"unknown mode {mode!r}"}


def _try_write(path: str) -> bool | str:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        return True
    except Exception as exc:  # pragma: no cover - exercised in docker tests
        return f"err:{exc.__class__.__name__}"


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
TMP_WRITE_TARGET = Path(tempfile.gettempdir()) / "ok"
ROOT_BLOCKED_TARGET = Path("/root/blocked")
