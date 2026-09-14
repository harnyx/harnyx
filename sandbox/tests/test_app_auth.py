from __future__ import annotations

import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from harnyx_sandbox.app import app


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
