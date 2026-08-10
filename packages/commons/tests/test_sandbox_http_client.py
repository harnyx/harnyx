from __future__ import annotations

import json
from collections.abc import Callable
from uuid import uuid4

import httpx
import pytest

import harnyx_commons.sandbox.docker as docker_module
from harnyx_commons.protocol_headers import SESSION_ID_HEADER
from harnyx_commons.sandbox.client import (
    InvalidSandboxResponseError,
    SandboxInvokeError,
    SandboxResponseProcessingError,
)
from harnyx_commons.sandbox.docker import HttpSandboxClient


def _request_json(request: httpx.Request) -> object:
    return json.loads(request.content.decode("utf-8"))


@pytest.mark.anyio("asyncio")
async def test_http_sandbox_client_retries_connect_error_with_same_session_and_connection_close() -> None:
    session_id = uuid4()
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("connect failed", request=request)
        return httpx.Response(status_code=200, json={"result": {"answer": "ok"}})

    async with httpx.AsyncClient(
        base_url="http://sandbox.local",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        result = await HttpSandboxClient("http://sandbox.local", client=http_client).invoke(
            "query",
            payload={"question": "hello"},
            context={"trace": "same"},
            token="session-token",  # noqa: S106 - fixed test-only sandbox token
            session_id=session_id,
        )

    assert result == {"answer": "ok"}
    assert len(requests) == 2
    assert {request.url.path for request in requests} == {"/entry/query"}
    assert {request.headers[SESSION_ID_HEADER] for request in requests} == {str(session_id)}
    assert {request.headers["Connection"] for request in requests} == {"close"}
    assert [_request_json(request) for request in requests] == [
        {"payload": {"question": "hello"}, "context": {"trace": "same"}},
        {"payload": {"question": "hello"}, "context": {"trace": "same"}},
    ]


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "exception_factory",
    [
        lambda request: httpx.ReadError("lost response", request=request),
        lambda request: httpx.RemoteProtocolError("remote disconnected", request=request),
        lambda request: httpx.WriteError("write failed", request=request),
    ],
    ids=["read", "remote-protocol", "write"],
)
async def test_http_sandbox_client_does_not_retry_errors_that_may_have_reached_sandbox(
    exception_factory: Callable[[httpx.Request], httpx.RequestError],
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise exception_factory(request)

    async with httpx.AsyncClient(
        base_url="http://sandbox.local",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        with pytest.raises(SandboxInvokeError) as exc_info:
            await HttpSandboxClient("http://sandbox.local", client=http_client).invoke(
                "query",
                payload={"question": "hello"},
                context={},
                token="session-token",  # noqa: S106 - fixed test-only sandbox token
                session_id=uuid4(),
            )

    assert len(requests) == 1
    assert exc_info.value.status_code == 0
    assert exc_info.value.detail_exception == type(exception_factory(requests[-1])).__name__
    assert exc_info.value.remote_state_uncertain is True


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "exception_factory",
    [
        lambda request: httpx.ReadTimeout("read timed out", request=request),
        lambda request: httpx.ConnectTimeout("connect timed out", request=request),
        lambda request: httpx.WriteTimeout("write timed out", request=request),
    ],
    ids=["read", "connect", "write"],
)
async def test_http_sandbox_client_does_not_retry_timeout_errors(
    exception_factory: Callable[[httpx.Request], httpx.TimeoutException],
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise exception_factory(request)

    async with httpx.AsyncClient(
        base_url="http://sandbox.local",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        with pytest.raises(SandboxInvokeError) as exc_info:
            await HttpSandboxClient("http://sandbox.local", client=http_client).invoke(
                "query",
                payload={"question": "hello"},
                context={},
                token="session-token",  # noqa: S106 - fixed test-only sandbox token
                session_id=uuid4(),
            )

    assert len(requests) == 1
    assert exc_info.value.status_code == 504
    assert exc_info.value.detail_exception == "TimeoutException"
    assert exc_info.value.remote_state_uncertain is True


@pytest.mark.anyio("asyncio")
async def test_http_sandbox_client_does_not_retry_sandbox_http_status_error() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=500,
            json={"detail": {"code": "UnhandledException", "exception": "ValueError", "error": "boom"}},
        )

    async with httpx.AsyncClient(
        base_url="http://sandbox.local",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        with pytest.raises(SandboxInvokeError) as exc_info:
            await HttpSandboxClient("http://sandbox.local", client=http_client).invoke(
                "query",
                payload={"question": "hello"},
                context={},
                token="session-token",  # noqa: S106 - fixed test-only sandbox token
                session_id=uuid4(),
            )

    assert len(requests) == 1
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail_code == "UnhandledException"
    assert exc_info.value.remote_state_uncertain is False


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=200, content=b"{not-json", headers={"content-type": "application/json"}),
        httpx.Response(status_code=200, json={"result": []}),
    ],
    ids=["malformed-json", "non-object-result"],
)
async def test_http_sandbox_client_reports_malformed_success_as_invalid_response(
    response: httpx.Response,
) -> None:
    async with httpx.AsyncClient(
        base_url="http://sandbox.local",
        transport=httpx.MockTransport(lambda _request: response),
    ) as http_client:
        with pytest.raises(InvalidSandboxResponseError):
            await HttpSandboxClient("http://sandbox.local", client=http_client).invoke(
                "query",
                payload={"question": "hello"},
                context={},
                token="session-token",  # noqa: S106 - fixed test-only sandbox token
                session_id=uuid4(),
            )


@pytest.mark.anyio("asyncio")
async def test_http_sandbox_client_distinguishes_unexpected_post_response_processing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(status_code=200, json={"result": {"answer": "ok"}})

    def fail_after_response(_value: object) -> dict[str, object]:
        raise RuntimeError("unexpected local parser failure")

    monkeypatch.setattr(docker_module, "_parse_sandbox_invoke_result", fail_after_response)
    async with httpx.AsyncClient(
        base_url="http://sandbox.local",
        transport=httpx.MockTransport(lambda _request: response),
    ) as http_client:
        with pytest.raises(SandboxResponseProcessingError, match="processing failed"):
            await HttpSandboxClient("http://sandbox.local", client=http_client).invoke(
                "query",
                payload={"question": "hello"},
                context={},
                token="session-token",  # noqa: S106 - fixed test-only sandbox token
                session_id=uuid4(),
            )


@pytest.mark.anyio("asyncio")
async def test_http_sandbox_client_preserves_failed_response_settlement_on_processing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(
        status_code=500,
        json={"detail": {"code": "UnhandledException", "exception": "ValueError", "error": "boom"}},
    )

    def fail_after_response(_value: object) -> object:
        raise RuntimeError("unexpected failed-response parser failure")

    monkeypatch.setattr(docker_module, "_parse_sandbox_response_detail", fail_after_response)
    async with httpx.AsyncClient(
        base_url="http://sandbox.local",
        transport=httpx.MockTransport(lambda _request: response),
    ) as http_client:
        with pytest.raises(SandboxResponseProcessingError, match="processing failed"):
            await HttpSandboxClient("http://sandbox.local", client=http_client).invoke(
                "query",
                payload={"question": "hello"},
                context={},
                token="session-token",  # noqa: S106 - fixed test-only sandbox token
                session_id=uuid4(),
            )


@pytest.mark.anyio("asyncio")
async def test_http_sandbox_client_metadata_only_failure_omits_raw_detail_and_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={
                "detail": {
                    "code": "UnhandledException",
                    "exception": "ValueError",
                    "error": "sandbox-response-sentinel",
                }
            },
        )

    caplog.set_level("ERROR", logger="harnyx_commons.sandbox.docker")
    async with httpx.AsyncClient(
        base_url="http://sandbox.local",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        with pytest.raises(SandboxInvokeError) as exc_info:
            await HttpSandboxClient("http://sandbox.local", client=http_client).invoke(
                "query",
                payload={"question": "query-sentinel"},
                context={},
                token="session-token",  # noqa: S106 - fixed test-only sandbox token
                session_id=uuid4(),
                include_failure_details=False,
            )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail_code == "UnhandledException"
    assert exc_info.value.detail_error is None
    assert exc_info.value.__cause__ is None
    serialized = repr(exc_info.value) + repr([record.__dict__ for record in caplog.records])
    assert "sandbox-response-sentinel" not in serialized


def test_http_sandbox_client_owned_client_disables_keepalive_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class CapturingAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(docker_module.httpx, "AsyncClient", CapturingAsyncClient)

    HttpSandboxClient("http://sandbox.local")

    limits = captured_kwargs["limits"]
    assert isinstance(limits, httpx.Limits)
    assert limits.max_keepalive_connections == 0
