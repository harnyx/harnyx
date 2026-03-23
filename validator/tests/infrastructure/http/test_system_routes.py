from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from harnyx_validator.application.status import StatusProvider
from harnyx_validator.infrastructure.http.routes import add_system_routes


def _create_test_client(status_provider: StatusProvider) -> TestClient:
    app = FastAPI()
    add_system_routes(app, status_provider)
    return TestClient(app)


def test_healthz_returns_ok() -> None:
    client = _create_test_client(StatusProvider())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_starting_until_platform_registration_completes() -> None:
    client = _create_test_client(StatusProvider())

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {"status": "waiting_for_platform_registration"}


def test_readyz_returns_ok_after_platform_registration() -> None:
    status_provider = StatusProvider()
    status_provider.mark_platform_registration_succeeded()
    client = _create_test_client(status_provider)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_surfaces_registration_failures() -> None:
    status_provider = StatusProvider()
    status_provider.mark_platform_registration_failed("platform rejected validator")
    client = _create_test_client(status_provider)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "registration_failed",
        "detail": "platform rejected validator",
    }
