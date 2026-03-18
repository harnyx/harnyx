from __future__ import annotations

from fastapi.testclient import TestClient
from harnyx_sandbox.app import app


def test_entry_route_requires_x_platform_token_header() -> None:
    client = TestClient(app)

    response = client.post(
        "/entry/missing",
        json={},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing x-platform-token header"}


def test_entry_route_openapi_security_declares_platform_token() -> None:
    security = app.openapi()["paths"]["/entry/{entrypoint_name}"]["post"]["security"]
    assert {"PlatformToken": []} in security


def test_entry_route_accepts_neutral_platform_token_header() -> None:
    client = TestClient(app)

    response = client.post(
        "/entry/missing",
        json={},
        headers={"x-platform-token": "token"},
    )

    assert response.status_code == 404
