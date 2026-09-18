"""Bound untrusted execution bodies before logging/auth and retain raw proxy paths."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request

from harnyx_commons.endpoint_execution import delegation_header
from harnyx_miner_sdk.endpoint_protocol import (
    ENDPOINT_CALLBACK_CONTEXT_HEADER,
    EndpointCallbackAcknowledgement,
    EndpointDelegation,
    EndpointDurableTerminalResult,
)
from harnyx_validator.infrastructure.http.middleware import request_logging_middleware
from harnyx_validator.infrastructure.http.routes import add_control_routes

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("callback", [False, True])
@pytest.mark.parametrize("prefix", ["", "/caf%C3%A9%2Ftenant"])
async def test_oversized_chunked_body_never_reaches_auth_or_logging_buffer(callback, prefix, monkeypatch, caplog):
    async def forbidden_body(request):
        raise AssertionError("raw request buffered before route bound")

    monkeypatch.setattr(Request, "body", forbidden_body)
    auth, worker = AsyncMock(), AsyncMock()
    app = FastAPI(root_path=unquote(prefix))
    app.middleware("http")(request_logging_middleware)
    add_control_routes(app, lambda: SimpleNamespace(auth=auth, endpoint_execution=worker))
    limit = 1_000_000 if callback else 2_000_000
    consumed = []

    async def body():
        consumed.append(1)
        yield b"private-evidence" + b" " * (limit - 16)
        consumed.append(2)
        yield b"xx"
        consumed.append(3)
        yield b"must not read"

    path = prefix + "/validator/endpoint-assignments" + (f"/{uuid4()}/callback" if callback else "")
    with caplog.at_level(logging.INFO, logger="harnyx_validator.http"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://validator.example"
        ) as client:
            response = await client.post(path, content=body())
    assert response.status_code == 413 and consumed == [1, 2]
    auth.assert_not_awaited()
    worker.execute.assert_not_awaited()
    worker.accept_callback.assert_not_awaited()
    assert "private-evidence" not in str([getattr(record, "data", {}) for record in caplog.records])


async def test_callback_passes_complete_body_and_original_encoded_proxy_path():
    identity = uuid4()
    prefix = "/caf%C3%A9%2Ftenant"
    path = f"{prefix}/validator/endpoint-assignments/{identity}/callback"
    worker = AsyncMock()
    worker.accept_callback.return_value = EndpointCallbackAcknowledgement(
        durable_terminal_result=EndpointDurableTerminalResult.PERSISTED
    )
    app = FastAPI(root_path=unquote(prefix))
    add_control_routes(app, lambda: SimpleNamespace(endpoint_execution=worker))
    delegation = EndpointDelegation(platform_hotkey="platform", body_utf8="{}", signature_hex="a" * 128)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://validator.example"
    ) as client:
        response = await client.post(
            path,
            content=b"original bytes",
            headers={
                ENDPOINT_CALLBACK_CONTEXT_HEADER: delegation_header(delegation),
                "Authorization": "original signature",
            },
        )
    assert response.status_code == 200
    values = worker.accept_callback.call_args.kwargs
    assert values["signed_path"] == path and values["raw_body"] == b"original bytes"
    assert values["authorization_header"] == "original signature" and values["delegation"] == delegation


@pytest.mark.parametrize(
    "context", [None, "", "not-an-encoded-document", "x" * 24_001], ids=["missing", "empty", "invalid", "oversized"]
)
async def test_coordinated_callback_requires_decodable_context_before_reporting(context):
    worker = AsyncMock()
    app = FastAPI()
    add_control_routes(app, lambda: SimpleNamespace(endpoint_execution=worker))
    headers = {} if context is None else {ENDPOINT_CALLBACK_CONTEXT_HEADER: context}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://validator.example"
    ) as client:
        response = await client.post(
            f"/validator/endpoint-assignments/{uuid4()}/callback", content=b"{}", headers=headers
        )
    assert response.status_code == 422
    worker.accept_callback.assert_not_awaited()
