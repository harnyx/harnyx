from __future__ import annotations

from caster_sandbox.app import app
from fastapi.testclient import TestClient


def test_entry_route_requires_x_caster_token_header() -> None:
    client = TestClient(app)

    response = client.post(
        "/entry/missing",
        json={},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing x-caster-token header"}


def test_entry_route_openapi_security_declares_caster_token() -> None:
    security = app.openapi()["paths"]["/entry/{entrypoint_name}"]["post"]["security"]
    assert {"CasterToken": []} in security


def test_entry_route_accepts_neutral_platform_token_header() -> None:
    client = TestClient(app)

    response = client.post(
        "/entry/missing",
        json={},
        headers={"x-platform-token": "token"},
    )

    assert response.status_code == 404
