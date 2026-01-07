from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import caster_commons.sandbox.harness as harness_module
from caster_commons.sandbox.harness import SandboxHarness
from caster_commons.tools.api import test_tool as invoke_test_tool
from caster_commons.tools.decorators import entrypoint, entrypoint_exists


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
        with factory_calls.get_lock():
            factory_calls.value += 1
        return FakeToolProxy()

    @entrypoint("echo")
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
        "/entry/echo",
        json={
            "payload": {"message": "hello"},
            "context": {"run_id": "abc"},
        },
        headers={"x-caster-token": "token"},
    )

    assert response.status_code == 200
    assert response.json()["result"] == {
        "message": "hello",
        "echo": "hello",
    }

    assert factory_calls.value == 1
    assert invoke_calls.value == 1
    assert close_flag.value == 1


def test_unknown_entrypoint_returns_404() -> None:
    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post("/entry/missing", json={})
    assert response.status_code == 404


def test_unknown_entrypoint_returns_404_with_preload() -> None:
    def noop_preload() -> None:  # executed in worker before lookup
        return None

    harness = SandboxHarness(preload=noop_preload)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post("/entry/missing", json={})
    assert response.status_code == 404


def test_harness_terminates_long_running_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "sleeper.txt"

    @entrypoint("sleeper_timeout")
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
        "/entry/sleeper_timeout",
        json={"payload": {}, "context": {}},
    )

    assert response.status_code == 504
    time.sleep(0.3)
    assert not marker.exists()


def test_preload_registers_entrypoint_inside_worker() -> None:
    def preload() -> None:
        if entrypoint_exists("lazy_preload"):
            return

        @entrypoint("lazy_preload")
        async def lazy_entrypoint(request: dict[str, object]) -> dict[str, object]:
            return {
                "message": request.get("message"),
            }

    harness = SandboxHarness(preload=preload)
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/lazy_preload",
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
    @entrypoint("key_error")
    async def key_error_entrypoint(request: dict[str, object]) -> dict[str, object]:
        del request
        raise KeyError("boom")

    harness = SandboxHarness()
    app = FastAPI()
    app.include_router(harness.create_router(), prefix="/entry")
    client = TestClient(app)

    response = client.post(
        "/entry/key_error",
        json={"payload": {}, "context": {}},
    )

    assert response.status_code == 500
