"""Protect signed assignment transport and uncertainty classification."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import bittensor as bt
import httpx
import pytest

from harnyx_commons.bittensor import build_canonical_request, verify_signed_request
from harnyx_commons.endpoint_execution import EndpointAuthority, delegation_signing_bytes
from harnyx_miner.endpoint_test_server import create_endpoint_test_app
from harnyx_miner_sdk.endpoint_protocol import EndpointAssignment, EndpointDelegation, EndpointMinerStatus, query_digest
from harnyx_miner_sdk.query import Query
from harnyx_validator.application.endpoint_execution import EndpointStatusUnavailableError
from harnyx_validator.infrastructure.endpoint_client import SignedMinerEndpointClient

pytestmark = pytest.mark.anyio("asyncio")


@pytest.mark.parametrize("operation", ["send", "status"])
@pytest.mark.parametrize("budget", [0.05, 60.0])
async def test_streaming_response_obeys_total_call_budget(operation, budget):
    """Progressing chunks cannot keep an acknowledgement or status request open indefinitely."""

    class Trickle(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.01)
                yield b" "

        async def aclose(self):
            self.closed = True

    stream = Trickle()
    platform = bt.Keypair.create_from_uri("//Alice")
    miner = bt.Keypair.create_from_uri("//Bob")
    assignment = _assignment(miner.ss58_address, platform)
    client = SignedMinerEndpointClient(
        validator_hotkey=platform,
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))),
    )
    started = asyncio.get_running_loop().time()
    try:
        async with asyncio.timeout(12):
            if operation == "send":
                assert (
                    await client.send_assignment(
                        endpoint_url="https://miner.example",
                        expected_hotkey=miner.ss58_address,
                        assignment=assignment,
                        stop_at=started + budget,
                    )
                ).attempted_at is not None
            else:
                with pytest.raises(EndpointStatusUnavailableError):
                    await client.get_status(
                        endpoint_url="https://miner.example",
                        expected_hotkey=miner.ss58_address,
                        assignment_id=assignment.assignment_id,
                        deadline_at=assignment.expires_at,
                        stop_at=started + budget,
                    )
        elapsed = asyncio.get_running_loop().time() - started
        assert min(budget, 10) - 0.02 <= elapsed < min(budget, 10) + 1
        assert stream.closed
    finally:
        await client.aclose()


def _assignment(miner_hotkey: str, platform=None) -> EndpointAssignment:
    platform = platform or bt.Keypair.create_from_uri("//Alice")
    query = Query(text="Find evidence")
    identity = uuid4()
    start = datetime.now(UTC)
    authority = EndpointAuthority(
        assignment_id=identity,
        validator_hotkey=platform.ss58_address,
        miner_hotkey=miner_hotkey,
        query_digest=query_digest(query),
        nonce="a" * 64,
        endpoint_url="https://miner.example",
        callback_url=f"https://validator.example/validator/endpoint-assignments/{identity}/callback",
        search_url=f"https://platform.example/v1/endpoint-assignments/{identity}/search",
        first_scheduled_at=start,
        started_at=start,
        deadline_at=start + timedelta(minutes=1),
        execution_timeout_seconds=60.0,
        attempt_number=1,
    )
    delegation = EndpointDelegation(
        platform_hotkey=platform.ss58_address,
        body_utf8=authority.model_dump_json(),
        signature_hex=platform.sign(delegation_signing_bytes(authority)).hex(),
    )
    return authority.assignment(query, delegation)


async def test_client_and_endpoint_exchange_signed_assignment_and_status() -> None:
    platform = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    miner = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    app = create_endpoint_test_app(
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=miner,
        endpoint_url="https://miner.example",
        block_at_registration=90,
    )
    client = SignedMinerEndpointClient(
        validator_hotkey=platform,
        client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
    )
    assignment = _assignment(miner.ss58_address, platform)

    outcome = await client.send_assignment(
        endpoint_url="https://miner.example",
        expected_hotkey=miner.ss58_address,
        assignment=assignment,
        stop_at=asyncio.get_running_loop().time() + 60,
    )
    status = await client.get_status(
        endpoint_url="https://miner.example",
        expected_hotkey=miner.ss58_address,
        assignment_id=assignment.assignment_id,
        deadline_at=assignment.expires_at,
        stop_at=asyncio.get_running_loop().time() + 60,
    )

    await client.aclose()
    assert outcome.attempted_at is not None
    assert outcome.admission_confirmed
    assert status.state is EndpointMinerStatus.RUNNING


async def test_unreachable_miner_and_http_failure_are_actual_attempts() -> None:
    platform = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    miner = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    assignment = _assignment(miner.ss58_address, platform)

    def cannot_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    connect_client = SignedMinerEndpointClient(
        validator_hotkey=platform,
        client=httpx.AsyncClient(transport=httpx.MockTransport(cannot_connect)),
    )
    assert (
        await connect_client.send_assignment(
            endpoint_url="https://miner.example",
            expected_hotkey=miner.ss58_address,
            assignment=assignment,
            stop_at=asyncio.get_running_loop().time() + 60,
        )
    ).attempted_at is not None
    await connect_client.aclose()

    failed_client = SignedMinerEndpointClient(
        validator_hotkey=platform,
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503))),
    )
    assert (
        await failed_client.send_assignment(
            endpoint_url="https://miner.example",
            expected_hotkey=miner.ss58_address,
            assignment=assignment,
            stop_at=asyncio.get_running_loop().time() + 60,
        )
    ).attempted_at is not None
    await failed_client.aclose()


@pytest.mark.parametrize("acknowledgement", ["unsigned", "malformed", "oversized", "declined"])
async def test_post_send_acknowledgement_failures_remain_uncertain(acknowledgement: str) -> None:
    """Future failure: a bad HTTP 200 acknowledgement must not escape and stop fan-out."""
    platform = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    miner = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    assignment = _assignment(miner.ss58_address, platform)

    def bad_ack(request: httpx.Request) -> httpx.Response:
        if acknowledgement == "oversized":
            return httpx.Response(200, content=b"x" * (8 * 1024 + 1))
        body = b"not-json"
        headers: dict[str, str] = {}
        if acknowledgement == "declined":
            body = b'{"accepted":false}'
        if acknowledgement in {"malformed", "declined"}:
            path = request.url.raw_path.decode("ascii")
            signature = miner.sign(build_canonical_request("POST", path, body)).hex()
            headers["Authorization"] = f'Bittensor ss58="{miner.ss58_address}",sig="{signature}"'
        return httpx.Response(200, content=body, headers=headers)

    client = SignedMinerEndpointClient(
        validator_hotkey=platform,
        client=httpx.AsyncClient(transport=httpx.MockTransport(bad_ack)),
    )

    outcome = await client.send_assignment(
        endpoint_url="https://miner.example",
        expected_hotkey=miner.ss58_address,
        assignment=assignment,
        stop_at=asyncio.get_running_loop().time() + 60,
    )

    await client.aclose()
    assert outcome.attempted_at is not None
    assert not outcome.admission_confirmed


@pytest.mark.parametrize("encoding", [None, "identity"])
async def test_client_signs_and_verifies_the_registered_endpoint_base_path(encoding: str | None) -> None:
    """Future failure: accepted endpoint base paths must be part of every transport signature."""
    platform = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    miner = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    assignment = _assignment(miner.ss58_address, platform)

    def base_path_endpoint(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii")
        assert path.startswith("/base/v1/endpoint-assignments")
        verify_signed_request(
            method=request.method,
            path_qs=path,
            body=request.content,
            authorization_header=request.headers.get("Authorization"),
            allowed_ss58=(platform.ss58_address,),
        )
        body = b'{"accepted":true}' if request.method == "POST" else b'{"state":"running"}'
        signature = miner.sign(build_canonical_request(request.method, path, body)).hex()
        headers = {"Authorization": f'Bittensor ss58="{miner.ss58_address}",sig="{signature}"'}
        if encoding is not None:
            headers["Content-Encoding"] = encoding
        return httpx.Response(
            200,
            content=body,
            headers=headers,
        )

    client = SignedMinerEndpointClient(
        validator_hotkey=platform,
        client=httpx.AsyncClient(transport=httpx.MockTransport(base_path_endpoint)),
    )

    outcome = await client.send_assignment(
        endpoint_url="https://miner.example/base",
        expected_hotkey=miner.ss58_address,
        assignment=assignment,
        stop_at=asyncio.get_running_loop().time() + 60,
    )
    status = await client.get_status(
        endpoint_url="https://miner.example/base",
        expected_hotkey=miner.ss58_address,
        assignment_id=assignment.assignment_id,
        deadline_at=assignment.expires_at,
        stop_at=asyncio.get_running_loop().time() + 60,
    )

    await client.aclose()
    assert (outcome).attempted_at is not None
    assert status.state is EndpointMinerStatus.RUNNING


@pytest.mark.parametrize("method", ["POST", "GET"])
@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "zstd", "identity, gzip", "unknown"])
async def test_encoded_replies_close_without_reading_or_decoding(method: str, encoding: str) -> None:
    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    assignment = _assignment(miner.ss58_address, platform)
    closed = False

    class UnreadBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("encoded reply body must not be read or decoded")
            yield b""  # pragma: no cover

        async def aclose(self):
            nonlocal closed
            closed = True

    def respond(request):
        return httpx.Response(200, headers={"Content-Encoding": encoding}, stream=UnreadBody())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client = SignedMinerEndpointClient(validator_hotkey=platform, client=http_client)
        if method == "POST":
            assert (
                await client.send_assignment(
                    endpoint_url="https://miner.example",
                    expected_hotkey=miner.ss58_address,
                    assignment=assignment,
                    stop_at=asyncio.get_running_loop().time() + 60,
                )
            ).attempted_at is not None
        else:
            with pytest.raises(EndpointStatusUnavailableError):
                await client.get_status(
                    endpoint_url="https://miner.example",
                    expected_hotkey=miner.ss58_address,
                    assignment_id=assignment.assignment_id,
                    deadline_at=assignment.expires_at,
                    stop_at=asyncio.get_running_loop().time() + 60,
                )
        assert closed


async def test_client_rejects_insecure_endpoint_before_sending() -> None:
    platform = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    miner = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    client = SignedMinerEndpointClient(validator_hotkey=platform)

    with pytest.raises(ValueError, match="HTTPS base URL"):
        await client.send_assignment(
            endpoint_url="http://miner.example",
            expected_hotkey=miner.ss58_address,
            assignment=_assignment(miner.ss58_address, platform),
            stop_at=asyncio.get_running_loop().time() + 60,
        )

    await client.aclose()


@pytest.mark.parametrize("failure", ["connect", "http", "oversized", "unsigned", "schema"])
async def test_status_failures_are_gateway_unavailability(failure: str) -> None:
    platform = bt.Keypair.create_from_uri("//Alice")
    miner = bt.Keypair.create_from_uri("//Bob")

    def respond(request: httpx.Request) -> httpx.Response:
        if failure == "connect":
            raise httpx.ConnectError("refused", request=request)
        body = b"x" * 8193 if failure == "oversized" else b'{"state":"invalid"}'
        signature = miner.sign(build_canonical_request("GET", request.url.raw_path.decode(), body)).hex()
        return httpx.Response(
            503 if failure == "http" else 200,
            content=body,
            headers={}
            if failure == "unsigned"
            else {"Authorization": f'Bittensor ss58="{miner.ss58_address}",sig="{signature}"'},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client = SignedMinerEndpointClient(validator_hotkey=platform, client=http_client)
        with pytest.raises(RuntimeError, match="status unavailable"):
            await client.get_status(
                endpoint_url="https://miner.example",
                expected_hotkey=miner.ss58_address,
                assignment_id=uuid4(),
                deadline_at=datetime.now(UTC) + timedelta(seconds=1),
                stop_at=asyncio.get_running_loop().time() + 60,
            )


@pytest.mark.parametrize("method", ["POST", "GET"])
async def test_signing_expiry_never_opens_transport(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    platform = bt.Keypair.create_from_uri("//Alice")
    assignment = _assignment(bt.Keypair.create_from_uri("//Bob").ss58_address)
    loop = asyncio.get_running_loop()
    now = loop.time()
    clock = Mock(return_value=now)
    signer = Mock(ss58_address=platform.ss58_address)

    def sign(body: bytes) -> bytes:
        clock.return_value = now + 120
        return platform.sign(body)

    signer.sign.side_effect = sign
    transport = Mock(return_value=httpx.Response(200))
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http_client:
        client = SignedMinerEndpointClient(validator_hotkey=signer, client=http_client)
        # Signing is synchronous: expire before the next scheduler checkpoint.
        with monkeypatch.context() as patch:
            patch.setattr(loop, "time", clock)
            if method == "POST":
                outcome = await client.send_assignment(
                    endpoint_url="https://miner.example",
                    expected_hotkey=assignment.expected_hotkey,
                    assignment=assignment,
                    stop_at=asyncio.get_running_loop().time() + 60,
                )
                assert (outcome).attempted_at is None
            else:
                with pytest.raises(RuntimeError, match="status unavailable"):
                    await client.get_status(
                        endpoint_url="https://miner.example",
                        expected_hotkey=assignment.expected_hotkey,
                        assignment_id=assignment.assignment_id,
                        deadline_at=assignment.expires_at,
                        stop_at=asyncio.get_running_loop().time() + 60,
                    )
        transport.assert_not_called()


class _WaitingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        yield b" "
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("method", ["POST", "GET"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_request_deadline_or_caller_cancellation_closes_stream(method: str, cancel: bool) -> None:
    assignment = _assignment(bt.Keypair.create_from_uri("//Bob").ss58_address).model_copy(
        update={"expires_at": datetime.now(UTC) + timedelta(seconds=10 if cancel else 0.05)}
    )
    stream = _WaitingStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    ) as http_client:
        client = SignedMinerEndpointClient(validator_hotkey=bt.Keypair.create_from_uri("//Alice"), client=http_client)

        async def invoke() -> object:
            if method == "POST":
                return await client.send_assignment(
                    endpoint_url="https://miner.example",
                    expected_hotkey=assignment.expected_hotkey,
                    assignment=assignment,
                    stop_at=asyncio.get_running_loop().time() + 60,
                )
            return await client.get_status(
                endpoint_url="https://miner.example",
                expected_hotkey=assignment.expected_hotkey,
                assignment_id=assignment.assignment_id,
                deadline_at=assignment.expires_at,
                stop_at=asyncio.get_running_loop().time() + 60,
            )

        async with asyncio.timeout(2):
            task = asyncio.create_task(invoke())
            try:
                await stream.started.wait()
                if cancel:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                elif method == "GET":
                    with pytest.raises(RuntimeError, match="status unavailable"):
                        await task
                else:
                    assert (await task).attempted_at is not None
                assert stream.closed
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
