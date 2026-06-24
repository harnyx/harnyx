from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from harnyx_commons.bittensor import VerificationError
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.infrastructure.http.middleware import request_logging_middleware
from harnyx_validator.infrastructure.http.routes import (
    ControlRouteAuth,
    ValidatorControlDeps,
    add_control_routes,
)
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageSnapshot


def _create_test_app(
    provider: DemoControlDependencyProvider,
) -> FastAPI:
    app = FastAPI()
    app.middleware("http")(request_logging_middleware)
    add_control_routes(app, provider)
    return app


@dataclass(frozen=True)
class _StubHotkey:
    ss58_address: str = "5validator"

    def sign(self, payload: bytes) -> bytes:
        return b"sig:" + payload


class StubStatusProvider:
    def snapshot(self) -> dict[str, object]:
        return {"status": "ok", "running": False}


class StubResourceUsageProvider:
    def snapshot(self) -> ValidatorResourceUsageSnapshot:
        return ValidatorResourceUsageSnapshot(
            captured_at=datetime(2026, 3, 31, 6, 42, tzinfo=UTC),
            cpu_percent=12.5,
            cpu_capacity_cores=4.0,
            memory_used_bytes=512,
            memory_total_bytes=2048,
            memory_percent=25.0,
            disk_used_bytes=4096,
            disk_total_bytes=8192,
            disk_percent=50.0,
        )


class DemoControlDependencyProvider:
    def __init__(
        self,
        *,
        auth: ControlRouteAuth | None = None,
        validator_hotkey: _StubHotkey | None = None,
        status_provider: StatusProvider | None = None,
        resource_usage_provider: object | None = None,
    ) -> None:
        self._deps = ValidatorControlDeps(
            status_provider=StubStatusProvider() if status_provider is None else status_provider,
            auth=_allow_all_auth if auth is None else auth,
            validator_hotkey=validator_hotkey or _StubHotkey(),
            resource_usage_provider=(
                StubResourceUsageProvider() if resource_usage_provider is None else resource_usage_provider
            ),
            batch_activity=object(),
        )

    def __call__(self) -> ValidatorControlDeps:
        return self._deps


async def _allow_all_auth(_: str, __: str, ___: bytes, ____: str | None) -> str:
    return "caller"


async def _auth_unavailable(_: str, __: str, ___: bytes, ____: str | None) -> str:
    raise VerificationError(
        "auth_unavailable",
        "inbound auth verifier has not completed initial hotkey warmup",
    )


def test_status_endpoint_awaits_auth_with_request_primitives() -> None:
    auth_calls: list[tuple[str, str, bytes, str | None]] = []

    async def _record_auth(
        method: str,
        path_qs: str,
        body: bytes,
        authorization_header: str | None,
    ) -> str:
        auth_calls.append((method, path_qs, body, authorization_header))
        return "caller"

    provider = DemoControlDependencyProvider(auth=_record_auth)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(
        "/validator/status?verbose=1",
        headers={"Authorization": 'Bittensor ss58="5demo",sig="00"'},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["hotkey"] == "5validator"
    assert response.json()["resource_usage"] == {
        "captured_at": "2026-03-31T06:42:00+00:00",
        "cpu_percent": 12.5,
        "cpu_capacity_cores": 4.0,
        "memory_used_bytes": 512,
        "memory_total_bytes": 2048,
        "memory_percent": 25.0,
        "disk_used_bytes": 4096,
        "disk_total_bytes": 8192,
        "disk_percent": 50.0,
    }
    assert auth_calls == [
        (
            "GET",
            "/validator/status?verbose=1",
            b"",
            'Bittensor ss58="5demo",sig="00"',
        )
    ]


def test_status_endpoint_returns_signed_ownership_proof_when_timestamp_header_is_present() -> None:
    provider = DemoControlDependencyProvider(validator_hotkey=_StubHotkey("5proof"))
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(
        "/validator/status",
        headers={
            "Authorization": 'Bittensor ss58="5demo",sig="00"',
            "X-Harnyx-Status-Ts": "2026-03-26T04:00:00+00:00",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["hotkey"] == "5proof"
    expected = "\n".join(
        (
            "validator-status-v1",
            "path=/validator/status",
            "request_ts=2026-03-26T04:00:00+00:00",
            "hotkey=5proof",
            "status=ok",
            "running=False",
        )
    ).encode("utf-8")
    assert payload["signature_hex"] == (b"sig:" + expected).hex()


def test_control_routes_return_503_when_auth_warmup_is_unavailable() -> None:
    provider = DemoControlDependencyProvider(auth=_auth_unavailable)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get("/validator/status")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "inbound auth verifier has not completed initial hotkey warmup",
    }


def test_pushed_batch_and_restore_routes_are_not_registered() -> None:
    provider = DemoControlDependencyProvider()
    app = _create_test_app(provider)
    client = TestClient(app)
    batch_id = uuid4()

    assert client.post("/validator/miner-task-batches/batch", json={}).status_code == 404
    assert client.get(f"/validator/miner-task-batches/{batch_id}/status").status_code == 404
    assert client.get(f"/validator/miner-task-batches/{batch_id}/runs").status_code == 404
