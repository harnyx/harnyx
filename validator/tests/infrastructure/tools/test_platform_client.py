from __future__ import annotations

import json
import re
import socket
from datetime import datetime
from uuid import uuid4

import bittensor as bt
import httpx
import pytest

from harnyx_commons.bittensor import build_canonical_request
from harnyx_commons.errors import BudgetExceededError, ToolInvocationTimeoutError, ToolProviderError
from harnyx_validator.infrastructure.tools.platform_client import (
    AsyncPlatformToolProxyPlatformClient,
    HttpPlatformClient,
    PlatformClientError,
    PlatformToolProxyBudgetExceededError,
    PlatformToolProxyInterruptedError,
    PlatformToolProxyInvocationError,
    PlatformToolProxyProviderError,
    PlatformToolProxyToolTimeoutError,
)

_HEADER_PATTERN = re.compile(r'^Bittensor\s+ss58="(?P<ss58>[^"]+)",\s*sig="(?P<sig>[0-9a-f]+)"$')


def _keypair() -> bt.Keypair:
    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _weights_response() -> httpx.Response:
    payload = {
        "weights": {"42": 0.7, "7": 0.3},
        "champion_uid": 42,
    }
    return httpx.Response(status_code=200, json=payload)


def _batch_response(
    *,
    batch_id,
    task_id,
    artifact_id,
    champion_artifact_id,
    budget_usd: float,
) -> httpx.Response:
    payload = {
        "batch_id": str(batch_id),
        "cutoff_at": "2025-10-17T12:00:00Z",
        "created_at": "2025-10-17T12:00:00Z",
        "tasks": [
            {
                "task_id": str(task_id),
                "query": {"text": "smoke"},
                "reference_answer": {"text": "ok"},
                "budget_usd": budget_usd,
            },
        ],
        "artifacts": [
            {
                "uid": 7,
                "artifact_id": str(artifact_id),
                "content_hash": "abc",
                "size_bytes": 1,
                "miner_hotkey_ss58": "miner-hotkey",
                "task_retry_count": 2,
            }
        ],
        "champion_artifact_id": str(champion_artifact_id),
        "completed_at": "2025-10-17T12:05:00Z",
        "failed_at": None,
    }
    return httpx.Response(status_code=200, json=payload)


def _artifact_response(content: bytes) -> httpx.Response:
    return httpx.Response(status_code=200, content=content)


class _FlakyTransport:
    def __init__(
        self,
        *,
        first_exception: type[httpx.TransportError],
        success_response: httpx.Response,
    ) -> None:
        self._first_exception = first_exception
        self._success_response = success_response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise self._first_exception("timed out", request=request)
        return self._success_response


class _AlwaysFailTransport:
    def __init__(self, *, exceptions: list[httpx.TransportError]) -> None:
        self._exceptions = exceptions
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raise self._exceptions[len(self.requests) - 1]


def _assert_signed(request: httpx.Request, keypair: bt.Keypair) -> None:
    header = request.headers.get("Authorization")
    assert header is not None
    match = _HEADER_PATTERN.match(header)
    assert match is not None
    assert match.group("ss58") == keypair.ss58_address
    path = request.url.raw_path.decode()
    query = request.url.query
    if query and "?" not in path:
        path = f"{path}?{query}"
    body = request.content or b""
    canonical = build_canonical_request(request.method, path, body)
    signature = bytes.fromhex(match.group("sig"))
    assert keypair.verify(canonical, signature)


def test_get_champion_weights_returns_weights() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        if request.method == "GET" and request.url.path == "/v1/weights":
            return _weights_response()
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    weights = client.get_champion_weights()

    assert weights.weights == {42: 0.7, 7: 0.3}
    assert weights.champion_uid == 42


def test_get_miner_task_batch_parses_tasks_and_artifacts() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    champion_artifact_id = uuid4()
    budget_usd = 0.123

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        expected_path = f"/v1/miner-task-batches/batch/{batch_id}"
        if request.method == "GET" and request.url.path == expected_path:
            return _batch_response(
                batch_id=batch_id,
                task_id=task_id,
                artifact_id=artifact_id,
                champion_artifact_id=champion_artifact_id,
                budget_usd=budget_usd,
            )
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    batch = client.get_miner_task_batch(batch_id)

    assert batch.batch_id == batch_id
    assert batch.tasks[0].task_id == task_id
    assert batch.tasks[0].budget_usd == pytest.approx(budget_usd)
    assert batch.tasks[0].query.text == "smoke"
    assert batch.tasks[0].reference_answer.text == "ok"
    assert batch.artifacts[0].artifact_id == artifact_id


def test_get_miner_task_batch_accepts_bri_494_deployment_safe_response_shape() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    champion_artifact_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        if request.method == "GET" and request.url.path == f"/v1/miner-task-batches/batch/{batch_id}":
            return _batch_response(
                batch_id=batch_id,
                task_id=task_id,
                artifact_id=artifact_id,
                champion_artifact_id=champion_artifact_id,
                budget_usd=0.1,
            )
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    batch = client.get_miner_task_batch(batch_id)

    assert batch.artifacts[0].artifact_id == artifact_id
    assert batch.artifacts[0].miner_hotkey_ss58 == "miner-hotkey"
    assert batch.artifacts[0].task_retry_count == 2


def test_get_miner_task_batch_accepts_submitted_at_on_platform_read_artifacts() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    champion_artifact_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        if request.method == "GET" and request.url.path == f"/v1/miner-task-batches/batch/{batch_id}":
            response = _batch_response(
                batch_id=batch_id,
                task_id=task_id,
                artifact_id=artifact_id,
                champion_artifact_id=champion_artifact_id,
                budget_usd=0.1,
            )
            payload = response.json()
            payload["artifacts"][0]["submitted_at"] = "2026-05-30T12:00:00Z"
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    batch = client.get_miner_task_batch(batch_id)

    assert batch.artifacts[0].artifact_id == artifact_id


def test_get_miner_task_batch_rejects_unknown_artifact_extra_after_submitted_at_strip() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    champion_artifact_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        if request.method == "GET" and request.url.path == f"/v1/miner-task-batches/batch/{batch_id}":
            response = _batch_response(
                batch_id=batch_id,
                task_id=task_id,
                artifact_id=artifact_id,
                champion_artifact_id=champion_artifact_id,
                budget_usd=0.1,
            )
            payload = response.json()
            payload["artifacts"][0]["submitted_at"] = "2026-05-30T12:00:00Z"
            payload["artifacts"][0]["unexpected_extra"] = True
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError):
        client.get_miner_task_batch(batch_id)


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_posts_attempt_number() -> None:
    batch_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    validator_session_id = uuid4()

    async def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        assert request.method == "POST"
        assert request.url.path == "/v1/platform-tool-proxy/grants"
        assert json.loads(request.content) == {
            "batch_id": str(batch_id),
            "artifact_id": str(artifact_id),
            "task_id": str(task_id),
            "validator_session_id": str(validator_session_id),
            "attempt_number": 2,
        }
        return httpx.Response(
            status_code=200,
            json={"token": "platform-tool-proxy-token", "expires_at": "2026-05-30T12:15:00+00:00"},
        )

    keypair = _keypair()
    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    grant = await client.create_platform_tool_proxy_grant(
        batch_id=batch_id,
        artifact_id=artifact_id,
        task_id=task_id,
        validator_session_id=validator_session_id,
        attempt_number=2,
    )

    assert grant.token == "platform-tool-proxy-token"  # noqa: S105 - fixed test-only proxy token


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_retries_transient_connect_timeout_then_succeeds() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectTimeout("grant connect timed out", request=request)
        return httpx.Response(
            status_code=200,
            json={"token": "platform-tool-proxy-token", "expires_at": "2026-05-30T12:15:00+00:00"},
        )

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0,),
    )

    grant = await client.create_platform_tool_proxy_grant(
        batch_id=uuid4(),
        artifact_id=uuid4(),
        task_id=uuid4(),
        validator_session_id=uuid4(),
        attempt_number=1,
    )

    assert grant.token == "platform-tool-proxy-token"  # noqa: S105 - fixed test-only proxy token
    assert [request.url.path for request in requests] == [
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
    ]


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
async def test_platform_tool_proxy_grant_retries_transient_status_then_succeeds(status_code: int) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status_code=status_code, json={"error_code": "temporary_platform_error"})
        return httpx.Response(
            status_code=200,
            json={"token": "platform-tool-proxy-token", "expires_at": "2026-05-30T12:15:00+00:00"},
        )

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0,),
    )

    grant = await client.create_platform_tool_proxy_grant(
        batch_id=uuid4(),
        artifact_id=uuid4(),
        task_id=uuid4(),
        validator_session_id=uuid4(),
        attempt_number=1,
    )

    assert grant.token == "platform-tool-proxy-token"  # noqa: S105 - fixed test-only proxy token
    assert [request.url.path for request in requests] == [
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
    ]


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_retry_exhaustion_maps_to_grant_failed() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectTimeout("grant connect timed out", request=request)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0, 0.0),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
        )

    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "platform_tool_proxy_grant_failed"
    assert [request.url.path for request in requests] == [
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
    ]


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_transient_status_retry_exhaustion_maps_to_grant_failed() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code=503, json={"error_code": "temporary_platform_error"})

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0, 0.0),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == "platform_tool_proxy_grant_failed"
    assert [request.url.path for request in requests] == [
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
    ]


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_does_not_retry_deterministic_response() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=403,
            json={"error_code": "platform_tool_proxy_denied", "message": "grant denied"},
        )

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0, 0.0),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
        )

    assert exc_info.value.error_code == "platform_tool_proxy_denied"
    assert len(requests) == 1


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    ("error_code", "status_code"),
    [
        ("platform_tool_proxy_denied", 403),
        ("platform_tool_proxy_grant_failed", 400),
    ],
)
async def test_platform_tool_proxy_grant_preserves_proxy_error_code(
    error_code: str,
    status_code: int,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/grants":
            return httpx.Response(
                status_code=status_code,
                json={"error_code": error_code, "message": f"{error_code} message"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.error_code == error_code
    assert str(exc_info.value) == f"{error_code} message"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=500, text="not json"),
        httpx.Response(status_code=500, json={"error_code": "unknown_platform_error"}),
    ],
)
async def test_platform_tool_proxy_grant_unknown_error_response_preserves_grant_failed_metadata(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/grants":
            return response
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_code == "platform_tool_proxy_grant_failed"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=200, text="not json"),
        httpx.Response(status_code=200, json={"token": "grant"}),
    ],
)
async def test_platform_tool_proxy_grant_invalid_success_response_preserves_grant_failed_metadata(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/grants":
            return response
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
        )

    assert exc_info.value.status_code == 200
    assert exc_info.value.error_code == "platform_tool_proxy_grant_failed"


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_tool_timeout_error_code_to_timeout_exception() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=408,
                json={"error_code": "tool_timeout", "message": "search_web timed out after 1 second"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyToolTimeoutError, match="search_web timed out") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert isinstance(exc_info.value, ToolInvocationTimeoutError)
    assert exc_info.value.status_code == 408
    assert exc_info.value.error_code == "tool_timeout"
    assert requests[0].headers["x-platform-tool-proxy-token"] == "proxy-token"


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_read_timeout_to_tool_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            raise httpx.ReadTimeout("platform execution timed out", request=request)
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyToolTimeoutError, match="awaiting tool result") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert isinstance(exc_info.value, ToolInvocationTimeoutError)
    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "tool_timeout"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    ("exception_type", "message"),
    [
        (httpx.RemoteProtocolError, "server disconnected without response"),
        (httpx.ReadError, "response stream closed"),
    ],
)
async def test_platform_tool_proxy_execute_maps_response_side_interruption_to_platform_interrupted(
    exception_type: type[httpx.HTTPError],
    message: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            raise exception_type(message, request=request)
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInterruptedError, match="interrupted before a response") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "platform_interrupted"


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_platform_interrupted_error_code_to_interrupted_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=400,
                json={
                    "error_code": "platform_interrupted",
                    "message": "platform tool proxy execution interrupted before completion",
                },
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInterruptedError, match="interrupted before completion") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "platform_interrupted"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    ("exception_type", "message"),
    [
        (httpx.ConnectTimeout, "platform endpoint unavailable"),
        (httpx.WriteError, "request body write failed"),
    ],
)
async def test_platform_tool_proxy_execute_keeps_pre_response_start_failures_validator_owned(
    exception_type: type[httpx.HTTPError],
    message: str,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            raise exception_type(message, request=request)
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "platform_tool_proxy_execution_failed"
    assert len(requests) == 1


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_provider_failed_to_tool_provider_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=502,
                json={"error_code": "provider_failed", "message": "provider rejected"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyProviderError, match="provider rejected") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert isinstance(exc_info.value, ToolProviderError)
    assert exc_info.value.status_code == 502
    assert exc_info.value.error_code == "provider_failed"


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_budget_exhausted_to_budget_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=400,
                json={"error_code": "budget_exhausted", "message": "grant budget exhausted"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyBudgetExceededError, match="grant budget exhausted") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert isinstance(exc_info.value, BudgetExceededError)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "budget_exhausted"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "error_code",
    [
        "platform_tool_proxy_denied",
        "miner_credential_missing",
        "concurrency_exhausted",
        "unsupported_provider",
        "unsupported_model",
        "invalid_request",
        "platform_error",
        "platform_tool_proxy_execution_failed",
        "duplicate_call",
    ],
)
async def test_platform_tool_proxy_execute_maps_proxy_policy_errors_to_non_provider_error(
    error_code: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=400,
                json={"error_code": error_code, "message": f"{error_code} message"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == error_code
    assert str(exc_info.value) == f"{error_code} message"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=500, text="not json"),
        httpx.Response(status_code=500, json={"error_code": "unknown_platform_error"}),
    ],
)
async def test_platform_tool_proxy_execute_unknown_error_response_preserves_platform_error_metadata(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return response
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_code == "platform_error"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=200, text="not json"),
        httpx.Response(status_code=200, json={"execution": {"elapsed_ms": 1.0}}),
    ],
)
async def test_platform_tool_proxy_execute_invalid_success_response_preserves_platform_error_metadata(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return response
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 200
    assert exc_info.value.error_code == "platform_error"


def test_get_champion_weights_retries_transient_connect_timeout() -> None:
    keypair = _keypair()
    transport = _FlakyTransport(
        first_exception=httpx.ConnectTimeout,
        success_response=_weights_response(),
    )
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(transport),
    )

    weights = client.get_champion_weights()

    assert weights.weights == {42: 0.7, 7: 0.3}
    assert weights.champion_uid == 42
    assert [request.url.path for request in transport.requests] == [
        "/v1/weights",
        "/v1/weights",
    ]
    for request in transport.requests:
        _assert_signed(request, keypair)


def test_get_miner_task_batch_does_not_retry_broad_connect_error() -> None:
    batch_id = uuid4()
    keypair = _keypair()
    connect_error = httpx.ConnectError("connect failed")
    transport = _AlwaysFailTransport(exceptions=[connect_error])
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(transport),
    )

    with pytest.raises(httpx.ConnectError) as exc_info:
        client.get_miner_task_batch(batch_id)

    assert exc_info.value is connect_error
    expected_path = f"/v1/miner-task-batches/batch/{batch_id}"
    assert [request.url.path for request in transport.requests] == [expected_path]


def test_fetch_artifact_retries_transient_connect_timeout() -> None:
    batch_id = uuid4()
    artifact_id = uuid4()
    content = b"artifact"
    keypair = _keypair()
    transport = _FlakyTransport(
        first_exception=httpx.ConnectTimeout,
        success_response=_artifact_response(content),
    )
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(transport),
    )

    fetched = client.fetch_artifact(batch_id, artifact_id)

    expected_path = f"/v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}"
    assert fetched == content
    assert [request.url.path for request in transport.requests] == [
        expected_path,
        expected_path,
    ]
    for request in transport.requests:
        _assert_signed(request, keypair)


def test_get_miner_task_batch_raises_original_exception_after_retry_exhaustion() -> None:
    batch_id = uuid4()
    keypair = _keypair()
    first_exception = httpx.ConnectTimeout("first timeout")
    final_exception = httpx.ConnectTimeout("final timeout")
    transport = _AlwaysFailTransport(exceptions=[first_exception, final_exception])
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(transport),
    )

    with pytest.raises(httpx.ConnectTimeout) as exc_info:
        client.get_miner_task_batch(batch_id)

    assert exc_info.value is final_exception
    expected_path = f"/v1/miner-task-batches/batch/{batch_id}"
    assert [request.url.path for request in transport.requests] == [
        expected_path,
        expected_path,
    ]


def test_get_miner_task_batch_retries_connect_error_with_temporary_dns_cause() -> None:
    batch_id = uuid4()
    keypair = _keypair()
    connect_error = httpx.ConnectError("connect failed")
    connect_error.__cause__ = socket.gaierror(socket.EAI_AGAIN, "temporary dns")
    final_exception = httpx.ConnectError("second connect failed")
    final_exception.__cause__ = socket.gaierror(socket.EAI_AGAIN, "temporary dns")
    transport = _AlwaysFailTransport(
        exceptions=[
            connect_error,
            final_exception,
        ]
    )
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(transport),
    )

    with pytest.raises(httpx.ConnectError) as exc_info:
        client.get_miner_task_batch(batch_id)

    assert exc_info.value is final_exception
    expected_path = f"/v1/miner-task-batches/batch/{batch_id}"
    assert [request.url.path for request in transport.requests] == [
        expected_path,
        expected_path,
    ]


def test_get_champion_weights_does_not_retry_non_connect_transport_failure() -> None:
    requests: list[httpx.Request] = []
    read_timeout = httpx.ReadTimeout("read timed out")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        _assert_signed(request, keypair)
        raise read_timeout

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(httpx.ReadTimeout) as exc_info:
        client.get_champion_weights()

    assert exc_info.value is read_timeout
    assert [request.url.path for request in requests] == ["/v1/weights"]


def test_fetch_artifact_does_not_retry_http_status_failure() -> None:
    batch_id = uuid4()
    artifact_id = uuid4()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        _assert_signed(request, keypair)
        return httpx.Response(status_code=500)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformClientError):
        client.fetch_artifact(batch_id, artifact_id)

    assert [request.url.path for request in requests] == [
        f"/v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}",
    ]


def test_get_restore_runs_page_accepts_null_next_cursor() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    keypair = _keypair()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == f"/v1/miner-task-batches/batch/{batch_id}":
            _assert_signed(request, keypair)
            return _batch_response(
                batch_id=batch_id,
                task_id=task_id,
                artifact_id=artifact_id,
                champion_artifact_id=artifact_id,
                budget_usd=1.0,
            )
        if request.url.path == f"/v1/miner-task-batches/{batch_id}/restore/runs":
            _assert_signed(request, keypair)
            return httpx.Response(
                status_code=200,
                json={
                    "batch_id": str(batch_id),
                    "snapshot_received_at": "2026-05-21T06:00:00+00:00",
                    "cursor": 0,
                    "limit": 50,
                    "next_cursor": None,
                    "items": [],
                },
            )
        return httpx.Response(status_code=404)

    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )
    batch = client.get_miner_task_batch(batch_id)

    page = client.get_restore_runs_page(
        batch=batch,
        snapshot_received_at=datetime.fromisoformat("2026-05-21T06:00:00+00:00"),
        cursor=0,
        limit=50,
    )

    assert page.next_cursor is None
    assert page.items == ()
    restore_request = requests[-1]
    assert restore_request.extensions["timeout"]["read"] == pytest.approx(300.0)
    assert [request.url.path for request in requests] == [
        f"/v1/miner-task-batches/batch/{batch_id}",
        f"/v1/miner-task-batches/{batch_id}/restore/runs",
    ]


def test_get_restore_metadata_uses_long_restore_timeout() -> None:
    batch_id = uuid4()
    keypair = _keypair()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == f"/v1/miner-task-batches/{batch_id}/restore":
            _assert_signed(request, keypair)
            return httpx.Response(
                status_code=200,
                json={
                    "batch_id": str(batch_id),
                    "snapshot_received_at": "2026-05-21T06:00:00+00:00",
                    "total_restore_runs": 0,
                    "page_limit": 50,
                    "last_progress_detail_sequence": 9,
                    "provider_model_evidence": [],
                },
            )
        return httpx.Response(status_code=404)

    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    metadata = client.get_restore_metadata(batch_id)

    assert metadata.page_limit == 50
    assert metadata.last_progress_detail_sequence == 9
    assert requests[0].extensions["timeout"]["read"] == pytest.approx(300.0)
