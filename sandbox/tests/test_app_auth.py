from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from harnyx_sandbox.app import _AdmissionResponse, app
from harnyx_sandbox.sandbox.harness import SandboxHarness

from harnyx_miner_sdk.sandbox_protocol import AdmissionRequest, SandboxAdmission


@pytest.fixture(autouse=True)
def control_credential(monkeypatch):
    monkeypatch.setattr(app.state, "control_token", "test-control", raising=False)


def test_entry_route_requires_x_platform_token_header() -> None:
    client = TestClient(app, headers={"x-sandbox-control-token": "test-control"})

    response = client.post(
        "/entry/missing",
        json={"context": {"time_budget": {"limit_seconds": 300.0}}},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing x-platform-token header"}


def test_entry_route_openapi_security_declares_platform_token() -> None:
    security = app.openapi()["paths"]["/entry/{entrypoint_name}"]["post"]["security"]
    assert {"PlatformToken": []} in security


def test_entry_route_accepts_neutral_platform_token_header() -> None:
    client = TestClient(app, headers={"x-sandbox-control-token": "test-control"})

    response = client.post(
        "/entry/missing",
        json={"context": {"time_budget": {"limit_seconds": 300.0}}},
        headers={"x-platform-token": "token"},
    )

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "PreloadInfrastructureFailed"


@pytest.mark.parametrize("route", ["/admission", "/admission/release", "/entry/query"])
@pytest.mark.parametrize("credential", ["", "wrong"])
def test_control_routes_reject_missing_or_wrong_credential(route, credential):
    response = TestClient(app).post(
        route,
        json={},
        headers={
            "x-platform-token": "present-but-not-control",
            "x-sandbox-control-token": credential,
        },
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid sandbox control credential"


def test_startup_requires_control_credential(monkeypatch):
    monkeypatch.delenv("SANDBOX_CONTROL_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SANDBOX_CONTROL_TOKEN is required"), TestClient(app):
        pass


def test_startup_consumes_control_credential(monkeypatch):
    from harnyx_sandbox import app as app_module
    from harnyx_sandbox.sandbox.harness import SandboxHarness

    monkeypatch.setattr(app_module, "sandbox_harness", SandboxHarness())
    credential = uuid4().hex
    monkeypatch.setenv("SANDBOX_CONTROL_TOKEN", credential)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert app.state.control_token == credential
        assert "SANDBOX_CONTROL_TOKEN" not in os.environ


def test_unconfirmed_cleanup_is_not_reported_as_settled_worker_failure(monkeypatch):
    from harnyx_sandbox.app import sandbox_harness
    from harnyx_sandbox.sandbox.harness import SandboxCleanupUnconfirmedError

    async def unconfirmed(*args, **kwargs):
        raise SandboxCleanupUnconfirmedError("cleanup pending")

    monkeypatch.setattr(sandbox_harness, "invoke", unconfirmed)
    response = TestClient(app).post(
        "/entry/missing",
        json={"context": {"time_budget": {"limit_seconds": 1.0}}},
        headers={"x-platform-token": "session", "x-sandbox-control-token": "test-control"},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "SandboxCleanupUnconfirmed"}}


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("send_fails", (False, True))
async def test_admission_response_keeps_only_successfully_sent_reservations(send_fails):
    """Send failure releases admission; disconnect after a completed send must retain it."""
    admission = SandboxAdmission(
        reservation_id="reservation",
        generation="compiler",
        deadline_monotonic_ns=time.monotonic_ns() + 10**9,
    )
    harness = Mock(reserve=AsyncMock(return_value=admission))
    disconnected = asyncio.Event()

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            if send_fails:
                raise OSError("connection closed")
            disconnected.set()

    response = _AdmissionResponse(AdmissionRequest(limit_seconds=1.0), {}, harness)
    if send_fails:
        with pytest.raises(OSError, match="connection closed"):
            await response({"type": "http"}, receive, send)
        harness.release.assert_called_once_with(admission)
    else:
        await response({"type": "http"}, receive, send)
        harness.release.assert_not_called()


@pytest.mark.anyio("asyncio")
async def test_disconnect_joins_admission_cleanup_without_cancelling_it_again():
    """Disconnect must await asynchronous reservation cleanup rather than interrupt it twice."""
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleaned = asyncio.Event()

    async def reserve(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await asyncio.sleep(0.01)
            cleaned.set()

    async def receive():
        await entered.wait()
        return {"type": "http.disconnect"}

    send = AsyncMock()
    response = _AdmissionResponse(AdmissionRequest(limit_seconds=1.0), {}, Mock(reserve=reserve))
    await asyncio.wait_for(response({}, receive, send), 1.0)
    assert cleanup_started.is_set() and cleaned.is_set()
    send.assert_not_awaited()


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("cold_compiler", (False, True))
async def test_observed_disconnect_cannot_continue_admission(cold_compiler):
    """An observed disconnect must settle before a cold compiler or discard-only send can hide it."""
    harness = SandboxHarness()
    harness._slots = asyncio.Semaphore(1 if cold_compiler else 0)
    ready = asyncio.get_running_loop().create_future()
    if not cold_compiler:
        ready.set_result(None)
    cleanup_started = asyncio.Event()
    cleaned = asyncio.Event()

    async def close():
        cleanup_started.set()
        await asyncio.sleep(0)
        cleaned.set()

    compiler = harness._compiler = Mock(
        ready=ready,
        generation="test",
        failure=None,
        close_task=None,
        close=AsyncMock(side_effect=close),
    )

    async def receive():
        if not cold_compiler:
            while not harness._slots._waiters:
                await asyncio.sleep(0)
            harness._slots.release()
        return {"type": "http.disconnect"}

    # Uvicorn silently returns from send on a disconnected connection.
    send = AsyncMock()
    response = asyncio.create_task(
        _AdmissionResponse(AdmissionRequest(limit_seconds=5.0), {}, harness)(
            {"type": "http"},
            receive,
            send,
        )
    )
    try:
        done, _ = await asyncio.wait((response,), timeout=0.2)
        assert response in done, "disconnected response still holds capacity awaiting compiler readiness"
        await response
        send.assert_not_awaited()
        assert not harness._reservations
        assert not harness._slots.locked()
        assert not ready.cancelled()
        if cold_compiler:
            assert cleanup_started.is_set() and cleaned.is_set()
            compiler.close.assert_awaited_once()
    finally:
        if not ready.done():
            ready.set_result(None)
        await asyncio.gather(response, return_exceptions=True)
        await harness.close()
