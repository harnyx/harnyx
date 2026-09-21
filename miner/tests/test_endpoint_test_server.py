"""Protect the runnable miner endpoint's signed and forgetful behavior."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import bittensor as bt
import httpx
import pytest

from harnyx_commons.bittensor import build_canonical_request, verify_signed_request
from harnyx_commons.endpoint_execution import (
    ENDPOINT_REPORT_FORWARDING_ALLOWANCE,
    EndpointAuthority,
    delegation_header,
    delegation_signing_bytes,
    parse_delegation_header,
)
from harnyx_miner.endpoint_test_server import EndpointTestServerState, create_endpoint_test_app
from harnyx_miner_sdk.endpoint_protocol import (
    ENDPOINT_CALLBACK_CONTEXT_HEADER,
    EndpointAssignment,
    EndpointCallback,
    EndpointCallbackAcknowledgement,
    EndpointDelegation,
    EndpointDurableTerminalResult,
    EndpointMinerStatus,
    query_digest,
)
from harnyx_miner_sdk.query import Query, Response

pytestmark = pytest.mark.anyio("asyncio")


async def test_injected_answerer_reuses_structured_answer_for_continuation_and_lost_ack():
    platform, miner, second_validator = (bt.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob", "//Charlie"))
    query = Query(text="Return null", output_schema={"type": "null"})
    original = _assignment(miner)
    assignment = _delegate(original.model_copy(update={"query": query, "query_digest": query_digest(query)}), platform)
    authority = EndpointAuthority.model_validate_json(
        parse_delegation_header(assignment.callback_context).body_utf8
    ).model_copy(
        update={
            "validator_hotkey": second_validator.ss58_address,
            "callback_url": "https://second-validator.example/callback",
            "attempt_number": 2,
        }
    )
    delegation = EndpointDelegation(
        platform_hotkey=platform.ss58_address,
        body_utf8=authority.model_dump_json(),
        signature_hex=platform.sign(delegation_signing_bytes(authority)).hex(),
    )
    continuation = authority.assignment(query, delegation)
    state = EndpointTestServerState()
    started, release = asyncio.Event(), asyncio.Event()
    invocations, delivered = [], []

    async def answerer(received):
        invocations.append(received)
        started.set()
        await release.wait()
        return Response(output=None)

    async def transport(request):
        delivered.append((str(request.url), request.content, request.headers.get(ENDPOINT_CALLBACK_CONTEXT_HEADER)))
        if len(delivered) == 1:
            raise httpx.ReadError("acknowledgment lost")
        return httpx.Response(200, json={"durable_terminal_result": "persisted"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            answerer=answerer,
            client=outbound,
        )
        async with (
            asyncio.timeout(5),
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://miner.example") as client,
        ):
            await _accept(client, platform, assignment)
            await started.wait()
            await _accept(client, platform, assignment)
            await _accept(client, second_validator, continuation)
            release.set()
            await asyncio.gather(*tuple(app.state.endpoint_background_tasks))
            assert invocations == [assignment]
            assert {url: context for url, _, context in delivered} == {
                assignment.callback_url: assignment.callback_context,
                continuation.callback_url: continuation.callback_context,
            }
            assert {url for url, _, _ in delivered} == {assignment.callback_url, continuation.callback_url}
            assert len({body for _, body, _ in delivered}) == 1
            callback = json.loads(delivered[0][1])
            assert callback["response"]["output"] is None
            assert callback["expires_at"] == assignment.model_dump(mode="json")["expires_at"]
            assert callback["assignment_id"] == str(assignment.assignment_id)
            assert len(state.acknowledged_destinations) == 2


@pytest.mark.parametrize("ending", ["failure", "expiry", "shutdown"])
async def test_injected_answer_failure_or_cancellation_cannot_send_success(ending):
    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    assignment, state = _assignment(miner), EndpointTestServerState()
    if ending == "expiry":
        assignment = _delegate(
            assignment.model_copy(update={"expires_at": datetime.now(UTC) + timedelta(milliseconds=80)}), platform
        )
    started, stopped = asyncio.Event(), asyncio.Event()

    async def answerer(_assignment):
        started.set()
        try:
            if ending == "failure":
                raise ValueError("invalid proof")
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def no_callback(request):
        pytest.fail("failed answer must not produce callback")

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_callback)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            answerer=answerer,
            client=outbound,
        )
        async with asyncio.timeout(5):
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://miner.example") as client,
            ):
                await _accept(client, platform, assignment)
                await started.wait()
                if ending != "shutdown":
                    await stopped.wait()
                    await asyncio.gather(*tuple(app.state.endpoint_background_tasks), return_exceptions=True)
                    assert (await _status(client, platform, assignment)).status_code == 503
            assert stopped.is_set()
            assert assignment.assignment_id in state.assignments
            assert not state.callbacks


def _assignment(miner):
    query = Query(text="Find evidence")
    identity = uuid4()
    assignment = EndpointAssignment(
        assignment_id=identity,
        query=query,
        query_digest=query_digest(query),
        expected_hotkey=miner.ss58_address,
        callback_url="https://platform.example/v1/endpoint-assignments/example/callback",
        search_url=f"https://platform.example/v1/endpoint-assignments/{identity}/search",
        endpoint_url="https://miner.example",
        nonce="b" * 64,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    return _delegate(assignment, bt.Keypair.create_from_uri("//Alice"))


def _delegate(assignment, platform):
    start = assignment.expires_at - timedelta(minutes=1)
    authority = EndpointAuthority(
        assignment_id=assignment.assignment_id,
        validator_hotkey=platform.ss58_address,
        miner_hotkey=assignment.expected_hotkey,
        query_digest=assignment.query_digest,
        nonce=assignment.nonce,
        endpoint_url="https://miner.example",
        callback_url=assignment.callback_url,
        search_url=assignment.search_url,
        first_scheduled_at=start,
        started_at=start,
        deadline_at=assignment.expires_at,
        execution_timeout_seconds=60.0,
        attempt_number=1,
    )
    delegation = EndpointDelegation(
        platform_hotkey=platform.ss58_address,
        body_utf8=authority.model_dump_json(),
        signature_hex=platform.sign(delegation_signing_bytes(authority)).hex(),
    )
    return assignment.model_copy(update={"callback_context": delegation_header(delegation)})


async def test_miner_accepts_http_callback_assignment() -> None:
    validator = bt.Keypair.create_from_uri("//Alice")
    miner = bt.Keypair.create_from_uri("//Bob")
    identity = uuid4()
    query = Query(text="Find evidence")
    assignment = EndpointAssignment.model_validate(
        {
            "assignment_id": identity,
            "query": query,
            "query_digest": query_digest(query),
            "expected_hotkey": miner.ss58_address,
            "callback_url": f"http://validator:8100/validator/endpoint-assignments/{identity}/callback",
            "search_url": f"https://platform.example/v1/endpoint-assignments/{identity}/search",
            "endpoint_url": "https://miner.example",
            "nonce": "b" * 64,
            "expires_at": datetime.now(UTC) + timedelta(minutes=1),
        },
    )
    state = EndpointTestServerState()

    async def callback(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"durable_terminal_result": "persisted"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(callback)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            miner_hotkey=miner,
            endpoint_url=assignment.endpoint_url,
            block_at_registration=90,
            state=state,
            answerer=AsyncMock(return_value=Response(text="answer")),
            client=outbound,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=assignment.endpoint_url) as client,
        ):
            await _accept(client, validator, assignment)
            await asyncio.gather(*tuple(app.state.endpoint_background_tasks))

    assert state.acknowledged == {identity}


async def _accept(client, platform, assignment):
    path = "/v1/endpoint-assignments"
    body = assignment.model_dump_json().encode()
    response = await client.post(
        path,
        content=body,
        headers={
            "Authorization": _authorization(platform, "POST", path, body),
        },
    )
    assert response.status_code == 200


async def _status(client, platform, assignment):
    path = f"/v1/endpoint-assignments/{assignment.assignment_id}/status"
    return await client.get(
        path,
        headers={
            "Authorization": _authorization(platform, "GET", path, b""),
        },
    )


def _search_result(request):
    return httpx.Response(
        200,
        json={
            "receipt_id": json.loads(request.content)["receipt_id"],
            "response": {"data": []},
            "results": [],
        },
    )


@pytest.mark.parametrize("terminal", list(EndpointDurableTerminalResult))
async def test_one_task_owns_search_and_delivery_through_duplicate_polls_and_acknowledgement(terminal):
    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    state, assignment = EndpointTestServerState(), _assignment(miner)
    search_started, send_started = asyncio.Event(), asyncio.Event()
    search_release, send_release = asyncio.Event(), asyncio.Event()
    searches = deliveries = 0

    async def transport(request):
        nonlocal searches, deliveries
        if request.url.path.endswith("/search"):
            searches += 1
            search_started.set()
            await search_release.wait()
            return _search_result(request)
        deliveries += 1
        send_started.set()
        await send_release.wait()
        return httpx.Response(200, json={"durable_terminal_result": terminal.value})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            provider="parallel",
            provider_key="secret-key",
            client=outbound,
        )
        async with asyncio.timeout(5), app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://miner.example"
            ) as client:
                await _accept(client, platform, assignment)
                await search_started.wait()
                await asyncio.gather(*(_accept(client, platform, assignment) for _ in range(3)))
                assert searches == 1
                search_release.set()
                await send_started.wait()
                replies = await asyncio.gather(*(_status(client, platform, assignment) for _ in range(4)))
                assert all(reply.json()["state"] == "completed" for reply in replies)
                assert deliveries == 1
                owners = tuple(app.state.endpoint_background_tasks)
                send_release.set()
                await asyncio.gather(*owners)
                await _status(client, platform, assignment)
                await _accept(client, platform, assignment)
                assert deliveries == 1
                assert not app.state.endpoint_background_tasks
                assert assignment.assignment_id in state.callbacks


@pytest.mark.parametrize("output_schema", [{}, {"type": "object", "properties": {"summary": {"type": "string"}}}])
async def test_structured_assignment_is_rejected_before_search_or_retention(output_schema):
    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    state = EndpointTestServerState()
    query = Query(text="Find evidence", output_schema=output_schema)
    assignment = _assignment(miner).model_copy(update={"query": query, "query_digest": query_digest(query)})

    async def no_outbound_request(request):
        pytest.fail("rejected assignment must not search or deliver a callback")

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_outbound_request)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            provider="parallel",
            provider_key="secret-key",
            client=outbound,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://miner.example"
            ) as client:
                path = "/v1/endpoint-assignments"
                body = assignment.model_dump_json().encode()
                replies = await asyncio.gather(
                    *(
                        client.post(
                            path, content=body, headers={"Authorization": _authorization(platform, "POST", path, body)}
                        )
                        for _ in range(2)
                    )
                )
                assert all(reply.status_code == 422 for reply in replies)
                assert (await _status(client, platform, assignment)).json()["state"] == "unknown"
                assert not state.assignments
                assert not state.callbacks
                assert not app.state.endpoint_background_tasks


@pytest.mark.parametrize("failure", ["http", "schema", "transport"])
async def test_failed_search_is_logged_retained_and_never_restarted(failure, caplog):
    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    state, assignment = EndpointTestServerState(), _assignment(miner)
    entered, release = asyncio.Event(), asyncio.Event()
    searches = 0

    async def transport(request):
        nonlocal searches
        searches += 1
        entered.set()
        await release.wait()
        if failure == "transport":
            raise httpx.ReadError("secret-key", request=request)
        return httpx.Response(503 if failure == "http" else 200, content=b"secret-key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            provider="parallel",
            provider_key="secret-key",
            client=outbound,
        )
        async with asyncio.timeout(5), app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://miner.example"
            ) as client:
                await _accept(client, platform, assignment)
                await entered.wait()
                tasks = tuple(app.state.endpoint_background_tasks)
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                assert (await _status(client, platform, assignment)).status_code == 503
                await _accept(client, platform, assignment)
                assert (await _status(client, platform, assignment)).status_code == 503
                assert searches == 1
                assert state.assignments[assignment.assignment_id] == assignment
                assert not state.callbacks and not app.state.endpoint_background_tasks
                assert "search failed" in caplog.text
                assert "secret-key" not in caplog.text + repr(state)


@pytest.mark.parametrize("stage", ["search", "callback"])
async def test_shutdown_joins_execution_or_delivery_and_retains_assignment(stage):
    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    state, assignment = EndpointTestServerState(), _assignment(miner)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def transport(request):
        if stage == "callback" and request.url.path.endswith("/search"):
            return _search_result(request)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            provider="parallel",
            provider_key="secret-key",
            client=outbound,
        )
        async with asyncio.timeout(5):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://miner.example"
                ) as client:
                    await _accept(client, platform, assignment)
                    await entered.wait()
                    tasks = tuple(app.state.endpoint_background_tasks)
            assert cancelled.is_set() and all(task.done() for task in tasks)
            assert not app.state.endpoint_background_tasks
            assert state.assignments[assignment.assignment_id] == assignment
            assert state.statuses.get(assignment.assignment_id) is not EndpointMinerStatus.RUNNING
            assert bool(state.callbacks) == (stage == "callback")
            assert not outbound.is_closed  # Injected client remains caller-owned.


@pytest.mark.parametrize("stage", ["search", "callback"])
async def test_clear_and_reaccept_cannot_be_changed_by_obsolete_task(stage):
    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    state, assignment = EndpointTestServerState(), _assignment(miner)
    entered, release = asyncio.Event(), asyncio.Event()
    replacement_entered, replacement_release = asyncio.Event(), asyncio.Event()
    searches = deliveries = 0

    async def transport(request):
        nonlocal searches, deliveries
        search = request.url.path.endswith("/search")
        if search:
            searches += 1
            if searches == 2:
                replacement_entered.set()
                await replacement_release.wait()
                return _search_result(request)
            if stage == "callback":
                return _search_result(request)
        else:
            deliveries += 1
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Model work that already finished at a remote boundary despite local cancellation.
            await release.wait()
        return _search_result(request) if search else httpx.Response(200, json={"durable_terminal_result": "persisted"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            provider="parallel",
            provider_key="secret-key",
            client=outbound,
        )
        async with asyncio.timeout(5), app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://miner.example"
            ) as client:
                await _accept(client, platform, assignment)
                await entered.wait()
                obsolete = tuple(app.state.endpoint_background_tasks)
                state.clear()
                await _accept(client, platform, assignment)
                await replacement_entered.wait()
                release.set()
                await asyncio.gather(*obsolete, return_exceptions=True)
                assert state.statuses[assignment.assignment_id] is EndpointMinerStatus.RUNNING
                assert not state.callbacks
                assert not state.acknowledged
                assert (await _status(client, platform, assignment)).json()["state"] == "running"
                replacement = tuple(app.state.endpoint_background_tasks)
                assert len(replacement) == 1
                replacement_release.set()
                await asyncio.gather(*replacement)
                assert deliveries == (2 if stage == "callback" else 1)
                assert not app.state.endpoint_background_tasks


async def test_expired_delivery_retains_callback_without_poll_restarting_it(monkeypatch):
    from harnyx_miner import endpoint_test_server

    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    state, assignment = EndpointTestServerState(), _assignment(miner)
    entered, release = asyncio.Event(), asyncio.Event()
    deliveries = 0

    class Expired:
        @staticmethod
        def now(tz):
            return assignment.expires_at + ENDPOINT_REPORT_FORWARDING_ALLOWANCE + timedelta(microseconds=1)

    async def transport(request):
        nonlocal deliveries
        if request.url.path.endswith("/search"):
            return _search_result(request)
        deliveries += 1
        entered.set()
        await release.wait()
        monkeypatch.setattr(endpoint_test_server, "datetime", Expired)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            provider="parallel",
            provider_key="secret-key",
            client=outbound,
            callback_retry_seconds=0,
        )
        async with asyncio.timeout(5), app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://miner.example",
            ) as client:
                await _accept(client, platform, assignment)
                await entered.wait()
                tasks = tuple(app.state.endpoint_background_tasks)
                release.set()
                await asyncio.gather(*tasks)
                assert (await _status(client, platform, assignment)).json()["state"] == "completed"
                assert deliveries == 1 and not app.state.endpoint_background_tasks
                assert state.callbacks and state.assignments and not state.acknowledged


@pytest.mark.parametrize("stall", ["headers", "body"])
async def test_callback_deadline_closes_stalled_attempt_and_retains_result(monkeypatch, caplog, stall):
    from harnyx_miner import endpoint_test_server

    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    state, assignment = EndpointTestServerState(), _assignment(miner)
    entered, closed, stream_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    expired = False

    class NearDeadline:
        @staticmethod
        def now(tz):
            return (
                assignment.expires_at + ENDPOINT_REPORT_FORWARDING_ALLOWANCE + timedelta(microseconds=1)
                if expired
                else assignment.expires_at + ENDPOINT_REPORT_FORWARDING_ALLOWANCE - timedelta(milliseconds=20)
            )

    async def wait_until_cancelled():
        nonlocal expired
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            expired = True
            closed.set()

    class StalledBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"durable_terminal_result":'
            await wait_until_cancelled()

        async def aclose(self):
            stream_closed.set()

    async def transport(request):
        if request.url.path.endswith("/search"):
            monkeypatch.setattr(endpoint_test_server, "datetime", NearDeadline)
            return _search_result(request)
        if stall == "headers":
            await wait_until_cancelled()
        return httpx.Response(200, stream=StalledBody())

    # The assignment deadline must own cancellation even without a client timeout.
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport), timeout=None) as outbound:  # noqa: S113
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            provider="parallel",
            provider_key="secret-key",
            client=outbound,
        )
        async with asyncio.timeout(5), app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://miner.example"
            ) as client:
                await _accept(client, platform, assignment)
                await entered.wait()
                saved_callback = state.callbacks[assignment.assignment_id]
                tasks = tuple(app.state.endpoint_background_tasks)
                await asyncio.gather(*tasks)
                assert closed.is_set()
                if stall == "body":
                    assert stream_closed.is_set()
                assert not state.active_tasks and not app.state.endpoint_background_tasks
                assert state.callbacks[assignment.assignment_id] is saved_callback
                assert state.assignments[assignment.assignment_id].expires_at == assignment.expires_at
                assert not state.acknowledged
                assert (await _status(client, platform, assignment)).json()["state"] == "completed"
                assert state.callback_attempts[assignment.assignment_id] == 1
        assert state.callbacks[assignment.assignment_id] is saved_callback
        assert not [record for record in caplog.records if record.levelno >= 40]


async def test_failed_assignment_does_not_interrupt_independent_execution(caplog):
    platform, miner = bt.Keypair.create_from_uri("//Alice"), bt.Keypair.create_from_uri("//Bob")
    state = EndpointTestServerState()
    failed, successful = _assignment(miner), _assignment(miner)
    entered = asyncio.Queue()
    release = asyncio.Event()

    async def transport(request):
        if request.url.path.endswith("/search"):
            await entered.put(json.loads(request.content)["receipt_id"])
            await release.wait()
            if str(failed.assignment_id) in request.content.decode():
                return httpx.Response(503)
            return _search_result(request)
        return httpx.Response(200, json={"durable_terminal_result": "persisted"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            platform_base_url="https://platform.example",
            miner_hotkey=miner,
            endpoint_url="https://miner.example",
            block_at_registration=90,
            state=state,
            provider="parallel",
            provider_key="secret-key",
            client=outbound,
        )
        async with asyncio.timeout(5), app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://miner.example",
            ) as client:
                await asyncio.gather(_accept(client, platform, failed), _accept(client, platform, successful))
                assert {await entered.get(), await entered.get()} == {
                    f"endpoint-test-{a.assignment_id}" for a in (failed, successful)
                }
                tasks = tuple(app.state.endpoint_background_tasks)
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                assert (await _status(client, platform, failed)).status_code == 503
                assert (await _status(client, platform, successful)).json()["state"] == "completed"
                assert state.acknowledged == {successful.assignment_id}
                assert len(state.assignments) == 2 and "search failed" in caplog.text


@pytest.mark.parametrize("declared_length", [True, False])
async def test_oversized_assignment_is_rejected_before_signature_validation(declared_length: bool) -> None:
    state = EndpointTestServerState()
    app = create_endpoint_test_app(
        endpoint_url="https://miner.example",
        block_at_registration=90,
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=bt.Keypair.create_from_uri("//Bob"),
        state=state,
    )
    chunks_read = 0

    async def chunks():
        nonlocal chunks_read
        for chunk in (b"x" * 500_000, b"x" * 500_001, b"must-not-read"):
            chunks_read += 1
            yield chunk

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://miner.example") as client:
        response = await client.post(
            "/v1/endpoint-assignments",
            content=chunks(),
            headers={"Content-Length": "1000001"} if declared_length else {},
        )
    assert response.status_code == 413
    assert chunks_read == (0 if declared_length else 2)
    assert state.assignments == {}


def _authorization(hotkey: bt.Keypair, method: str, path: str, body: bytes) -> str:
    signature = hotkey.sign(build_canonical_request(method, path, body)).hex()
    return f'Bittensor ss58="{hotkey.ss58_address}",sig="{signature}"'


async def test_assignment_and_status_are_signed_and_unknown_after_memory_clear() -> None:
    platform_hotkey = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    miner_hotkey = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    state = EndpointTestServerState()
    app = create_endpoint_test_app(
        endpoint_url="https://miner.example",
        block_at_registration=90,
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=miner_hotkey,
        state=state,
    )
    query = Query(text="Find evidence")
    identity = uuid4()
    assignment = EndpointAssignment(
        assignment_id=identity,
        query=query,
        query_digest=query_digest(query),
        expected_hotkey=miner_hotkey.ss58_address,
        callback_url="https://platform.example/v1/endpoint-assignments/callback",
        search_url=f"https://platform.example/v1/endpoint-assignments/{identity}/search",
        endpoint_url="https://miner.example",
        nonce="b" * 64,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    assignment = _delegate(assignment, platform_hotkey)
    path = "/v1/endpoint-assignments"
    body = assignment.model_dump_json().encode()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://miner.example") as client:
        accepted = await client.post(
            path,
            content=body,
            headers={"Authorization": _authorization(platform_hotkey, "POST", path, body)},
        )
        assert accepted.status_code == 200
        verify_signed_request(
            method="POST",
            path_qs=path,
            body=accepted.content,
            authorization_header=accepted.headers["Authorization"],
            allowed_ss58=(miner_hotkey.ss58_address,),
        )

        status_path = f"/v1/endpoint-assignments/{assignment.assignment_id}/status"
        known = await client.get(
            status_path,
            headers={
                "Authorization": _authorization(platform_hotkey, "GET", status_path, b""),
            },
        )
        assert known.json()["state"] == EndpointMinerStatus.RUNNING.value

        state.clear()
        unknown = await client.get(
            status_path,
            headers={
                "Authorization": _authorization(platform_hotkey, "GET", status_path, b""),
            },
        )
        assert unknown.json()["state"] == EndpointMinerStatus.UNKNOWN.value


async def test_tampered_assignment_is_rejected_without_retaining_secrets() -> None:
    platform_hotkey = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    miner_hotkey = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    state = EndpointTestServerState()
    app = create_endpoint_test_app(
        endpoint_url="https://miner.example",
        block_at_registration=90,
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=miner_hotkey,
        state=state,
    )
    path = "/v1/endpoint-assignments"
    signed_body = b'{"invalid":"before tamper"}'
    tampered_body = b'{"invalid":"after tamper","provider_key":"must-not-survive"}'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://miner.example") as client:
        response = await client.post(
            path,
            content=tampered_body,
            headers={"Authorization": _authorization(platform_hotkey, "POST", path, signed_body)},
        )

    assert response.status_code in {401, 422}
    assert "must-not-survive" not in repr(state)


@pytest.mark.parametrize("note", ["Evidence summary", None, "", "   "])
async def test_executable_endpoint_searches_and_retries_callback_without_retaining_provider_key(note) -> None:
    platform_hotkey = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    miner_hotkey = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    state = EndpointTestServerState()
    callback_attempts = 0
    observed_provider_keys: list[str] = []

    async def platform(request: httpx.Request) -> httpx.Response:
        nonlocal callback_attempts
        verify_signed_request(
            method=request.method,
            path_qs=request.url.raw_path.decode("ascii"),
            body=request.content,
            authorization_header=request.headers.get("Authorization"),
            allowed_ss58=(miner_hotkey.ss58_address,),
        )
        if request.url.path.endswith("/search"):
            observed_provider_keys.append(request.headers["X-Provider-Api-Key"])
            search_request = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "receipt_id": search_request["receipt_id"],
                    "response": {"data": []},
                    "results": [
                        {
                            "result_id": "result-1",
                            "url": "https://example.com/evidence",
                            "note": note,
                            "title": "Evidence",
                        }
                    ],
                },
            )
        callback_attempts += 1
        answer = json.loads(request.content)["response"]
        assert bool(answer.get("citations")) is bool(note and note.strip())
        assert answer["text"] == (note if note and note.strip() else "Evidence")
        if callback_attempts == 1:
            return httpx.Response(503)
        acknowledgement = EndpointCallbackAcknowledgement(
            durable_terminal_result=EndpointDurableTerminalResult.PERSISTED
        )
        return httpx.Response(200, content=acknowledgement.model_dump_json().encode())

    platform_client = httpx.AsyncClient(transport=httpx.MockTransport(platform))
    app = create_endpoint_test_app(
        endpoint_url="https://miner.example",
        block_at_registration=90,
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=miner_hotkey,
        state=state,
        provider="parallel",
        provider_key="transient-provider-key",
        client=platform_client,
        callback_retry_seconds=0,
    )
    query = Query(text="Find evidence")
    identity = uuid4()
    assignment = EndpointAssignment(
        assignment_id=identity,
        query=query,
        query_digest=query_digest(query),
        expected_hotkey=miner_hotkey.ss58_address,
        callback_url="https://platform.example/v1/endpoint-assignments/00000000-0000-0000-0000-000000000001/callback",
        search_url=f"https://platform.example/v1/endpoint-assignments/{identity}/search",
        endpoint_url="https://miner.example",
        nonce="b" * 64,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    assignment = _delegate(assignment, platform_hotkey)
    path = "/v1/endpoint-assignments"
    body = assignment.model_dump_json().encode()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://miner.example") as client:
        accepted = await client.post(
            path,
            content=body,
            headers={"Authorization": _authorization(platform_hotkey, "POST", path, body)},
        )
        assert accepted.status_code == 200
        for _ in range(100):
            if state.callback_attempts.get(assignment.assignment_id) == 2:
                break
            await asyncio.sleep(0)

    await platform_client.aclose()
    assert observed_provider_keys == ["transient-provider-key"]
    assert callback_attempts == 2
    assert state.statuses[assignment.assignment_id] is EndpointMinerStatus.COMPLETED
    assert "transient-provider-key" not in repr(state)


def _challenge(miner: bt.Keypair, url: str) -> dict[str, object]:
    return {
        "purpose": "harnyx.miner_endpoint_ownership.v1",
        "nonce": "a" * 64,
        "hotkey": miner.ss58_address,
        "url": url,
        "block_at_registration": 90,
        "expires_at": (datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
    }


@pytest.mark.parametrize("prefix", ["", "/base", "/caf%C3%A9", "/base%2F", "/%7Bassignment_id%7D", "/%7Btenant:int%7D"])
async def test_ownership_proof_signs_exact_body_and_registered_path(prefix: str) -> None:
    miner = bt.Keypair.create_from_uri("//Bob")
    url = "https://miner.example" + prefix
    app = create_endpoint_test_app(
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=miner,
        endpoint_url=url,
        block_at_registration=90,
    )
    # Whitespace/order must survive signing; parsing then reserializing is not a proof.
    body = json.dumps(_challenge(miner, url), indent=2).encode() + b"\n"
    path = prefix + "/verify"
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=url) as client:
            response = await client.post("https://miner.example" + path, content=body)
    assert response.status_code == 200
    assert set(response.json()) == {"signature"}
    signature = bytes.fromhex(response.json()["signature"])
    assert len(signature) == 64
    assert miner.verify(build_canonical_request("POST", path, body), signature)
    assert not miner.verify(build_canonical_request("POST", path, body.rstrip()), signature)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("purpose", "another-protocol"),
        ("hotkey", bt.Keypair.create_from_uri("//Alice").ss58_address),
        ("url", "https://attacker.example"),
        ("block_at_registration", 91),
        ("block_at_registration", "90"),
        ("block_at_registration", True),
        ("nonce", "not-a-challenge"),
        ("expires_at", "2000-01-01T00:00:00+00:00"),
        ("expires_at", "2999-01-01T00:00:00"),
        ("expires_at", 32503680000),
        ("extra", "unrecognized"),
    ],
)
async def test_ownership_proof_rejects_invalid_challenge_without_signature(field: str, value: object) -> None:
    miner = bt.Keypair.create_from_uri("//Bob")
    url = "https://miner.example"
    app = create_endpoint_test_app(
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=miner,
        endpoint_url=url,
        block_at_registration=90,
    )
    challenge = _challenge(miner, url)
    challenge[field] = value
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=url) as client:
            response = await client.post("/verify", json=challenge)
    assert 400 <= response.status_code < 500
    assert "signature" not in response.json()


@pytest.mark.parametrize("path", ["/verify", "/base-other/verify"])
async def test_ownership_proof_rejects_requests_outside_registered_base(path: str) -> None:
    miner = bt.Keypair.create_from_uri("//Bob")
    url = "https://miner.example/base"
    app = create_endpoint_test_app(
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=miner,
        endpoint_url=url,
        block_at_registration=90,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            response = await client.post("https://miner.example" + path, json=_challenge(miner, url))
    assert response.status_code == 404


@pytest.mark.parametrize("body", [b"not-json", b"{}", b"[]"])
async def test_ownership_proof_rejects_malformed_or_incomplete_json(body: bytes) -> None:
    app = create_endpoint_test_app(
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=bt.Keypair.create_from_uri("//Bob"),
        endpoint_url="https://miner.example",
        block_at_registration=90,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            response = await client.post("https://miner.example/verify", content=body)
    assert response.status_code == 422
    assert "signature" not in response.json()


@pytest.mark.parametrize("declared_length", [True, False])
async def test_ownership_proof_rejects_oversized_body_without_buffering_remainder(declared_length: bool) -> None:
    app = create_endpoint_test_app(
        validator_eligibility=AsyncMock(return_value=True),
        platform_base_url="https://platform.example",
        miner_hotkey=bt.Keypair.create_from_uri("//Bob"),
        endpoint_url="https://miner.example",
        block_at_registration=90,
    )
    chunks_read = 0

    async def chunks():
        nonlocal chunks_read
        for chunk in (b"x" * 4096, b"x" * 4097, b"must-not-read"):
            chunks_read += 1
            yield chunk

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://miner.example",
        ) as client:
            response = await client.post(
                "/verify",
                content=chunks(),
                headers={"Content-Length": "8193"} if declared_length else {},
            )
    assert response.status_code == 413
    assert chunks_read == (0 if declared_length else 2)
    assert "signature" not in response.json()


@pytest.mark.parametrize("failure", ["status", "connection", "headers", "body"])
async def test_callback_backoff_recovers_with_original_result_after_failed_attempts(monkeypatch, failure):
    from harnyx_miner import endpoint_test_server

    miner = bt.Keypair.create_from_uri("//Bob")
    assignment = _assignment(miner)
    state = EndpointTestServerState()
    state.assignments[assignment.assignment_id] = assignment
    callback = EndpointCallback(
        assignment_id=assignment.assignment_id,
        query_digest=assignment.query_digest,
        nonce=assignment.nonce,
        expires_at=assignment.expires_at,
        response=Response(text="Saved answer"),
    )
    delays, requests, closed = [], [], []

    async def sleep(delay):
        delays.append(delay)

    class StalledBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{"
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    async def transport(request):
        requests.append(request)
        if len(requests) <= 4:
            if failure == "connection":
                raise httpx.ConnectError("validator unavailable", request=request)
            if failure == "headers":
                await asyncio.Event().wait()
            if failure == "body":
                return httpx.Response(200, stream=StalledBody())
            return httpx.Response(503)
        return httpx.Response(200, json={"durable_terminal_result": "persisted"})

    monkeypatch.setattr(endpoint_test_server.asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        await endpoint_test_server._deliver_callback(
            assignment=assignment,
            callback=callback,
            miner_hotkey=miner,
            state=state,
            client=client,
            retry_seconds=1,
            retry_max_seconds=3,
            attempt_timeout_seconds=0.02,
        )

    assert delays == [1, 2, 3, 3]
    assert len(requests) == 5
    assert {request.content for request in requests} == {callback.model_dump_json().encode()}
    assert state.acknowledged_destinations == {(assignment.assignment_id, assignment.callback_url)}
    assert state.assignments[assignment.assignment_id].expires_at == assignment.expires_at
    if failure == "body":
        assert len(closed) == 4


async def test_callback_backoff_stops_at_original_delivery_cutoff(monkeypatch):
    from harnyx_miner import endpoint_test_server

    miner = bt.Keypair.create_from_uri("//Bob")
    assignment = _assignment(miner)
    cutoff = assignment.expires_at + ENDPOINT_REPORT_FORWARDING_ALLOWANCE
    now = cutoff - timedelta(seconds=0.5)
    state = EndpointTestServerState()
    state.assignments[assignment.assignment_id] = assignment
    callback = EndpointCallback(
        assignment_id=assignment.assignment_id,
        query_digest=assignment.query_digest,
        nonce=assignment.nonce,
        expires_at=assignment.expires_at,
        response=Response(text="Saved"),
    )
    delays, requests = [], []

    class Clock:
        @staticmethod
        def now(tz):
            return now

    async def sleep(delay):
        nonlocal now
        delays.append(delay)
        now += timedelta(seconds=delay, microseconds=1)

    async def transport(request):
        requests.append(request)
        return httpx.Response(503)

    monkeypatch.setattr(endpoint_test_server, "datetime", Clock)
    monkeypatch.setattr(endpoint_test_server.asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        await endpoint_test_server._deliver_callback(
            assignment=assignment,
            callback=callback,
            miner_hotkey=miner,
            state=state,
            client=client,
            retry_seconds=1,
            retry_max_seconds=10,
            attempt_timeout_seconds=5,
        )
    assert delays == [0.5]
    assert len(requests) == 1
    assert requests[0].extensions["timeout"]["read"] == 0.5
    assert not state.acknowledged


@pytest.mark.parametrize(
    "context",
    [None, "", "opaque context; NOT a Platform document", "x" * 24_000],
    ids=["absent", "empty", "opaque", "maximum"],
)
async def test_direct_request_returns_exact_optional_context_and_signed_answer(context):
    validator, miner = (bt.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob"))
    assignment = _assignment(miner).model_copy(update={"callback_context": context})
    received = []

    async def callback(request):
        verify_signed_request(
            method="POST",
            path_qs=request.url.raw_path.decode(),
            body=request.content,
            authorization_header=request.headers["Authorization"],
            allowed_ss58=(miner.ss58_address,),
        )
        received.append(request)
        return httpx.Response(200, json={"durable_terminal_result": "persisted"})

    eligibility = AsyncMock(return_value=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(callback)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=eligibility,
            miner_hotkey=miner,
            endpoint_url=assignment.endpoint_url,
            block_at_registration=90,
            answerer=AsyncMock(return_value=Response(text="answer")),
            client=outbound,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=assignment.endpoint_url) as client,
        ):
            await _accept(client, validator, assignment)
            await asyncio.gather(*tuple(app.state.endpoint_background_tasks))
            assert (await _status(client, validator, assignment)).status_code == 200
    eligibility.assert_awaited_once_with(validator.ss58_address)
    assert len(received) == 1
    assert received[0].headers.get(ENDPOINT_CALLBACK_CONTEXT_HEADER) == context
    assert EndpointCallback.model_validate_json(received[0].content).response.text == "answer"


async def test_replacement_requires_matching_nonce_and_eligibility_and_memory_loss_rechecks():
    original, replacement, miner = (bt.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Charlie", "//Bob"))
    assignment = _assignment(miner).model_copy(update={"callback_context": "original"})
    continuation = assignment.model_copy(
        update={"callback_context": "replacement", "callback_url": "https://other.example/callback"}
    )
    state = EndpointTestServerState()
    eligibility = AsyncMock(return_value=True)
    app = create_endpoint_test_app(
        validator_eligibility=eligibility,
        miner_hotkey=miner,
        endpoint_url=assignment.endpoint_url,
        block_at_registration=90,
        state=state,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=assignment.endpoint_url) as client,
    ):
        await _accept(client, original, assignment)
        eligibility.return_value = False
        await _accept(client, original, assignment)
        assert (await _status(client, original, assignment)).status_code == 200
        assert (await _status(client, replacement, assignment)).status_code == 403
        path = "/v1/endpoint-assignments"

        async def post(key, request):
            body = request.model_dump_json().encode()
            return await client.post(
                path, content=body, headers={"Authorization": _authorization(key, "POST", path, body)}
            )

        assert (await post(replacement, continuation)).status_code == 403
        eligibility.return_value = True
        assert (await post(replacement, continuation.model_copy(update={"nonce": "z" * 64}))).status_code == 409
        await _accept(client, replacement, continuation)
        assert state.assignments[assignment.assignment_id] == assignment
        assert set(state.destinations[assignment.assignment_id]) == {assignment.callback_url, continuation.callback_url}
        assert (
            await post(replacement, assignment.model_copy(update={"callback_context": "overwrite"}))
        ).status_code == 409
        state.clear()
        eligibility.return_value = False
        assert (await _status(client, original, assignment)).status_code == 403
        assert (await post(original, assignment)).status_code == 403
        eligibility.return_value = True
        assert (await _status(client, replacement, assignment)).json()["state"] == "unknown"
        await _accept(client, replacement, continuation)
        assert state.assignments[assignment.assignment_id].expires_at == assignment.expires_at


@pytest.mark.parametrize("eligibility_result, expected", [(False, 403), (RuntimeError("chain unavailable"), 503)])
async def test_failed_eligibility_never_admits_assignment_or_unknown_status(eligibility_result, expected):
    validator, miner = (bt.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob"))
    assignment = _assignment(miner)
    eligibility = (
        AsyncMock(side_effect=eligibility_result)
        if isinstance(eligibility_result, Exception)
        else AsyncMock(return_value=False)
    )
    state = EndpointTestServerState()
    app = create_endpoint_test_app(
        validator_eligibility=eligibility,
        miner_hotkey=miner,
        endpoint_url=assignment.endpoint_url,
        block_at_registration=90,
        state=state,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=assignment.endpoint_url) as client,
    ):
        body = assignment.model_dump_json().encode()
        path = "/v1/endpoint-assignments"
        result = await client.post(
            path, content=body, headers={"Authorization": _authorization(validator, "POST", path, body)}
        )
        assert result.status_code == expected
        assert (await _status(client, validator, assignment)).status_code == expected
        assert not state.assignments and not state.signers


@pytest.mark.parametrize("operation", ["status", "assignment"])
async def test_chain_lookup_does_not_block_admitted_work_or_bypass_retained_status_admission(operation):
    original, newcomer, miner = (bt.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Charlie", "//Bob"))
    blocked, release = asyncio.Event(), asyncio.Event()
    state = EndpointTestServerState()
    assignment = _assignment(miner)

    async def eligibility(signer):
        if signer == newcomer.ss58_address:
            blocked.set()
            await release.wait()
        return True

    app = create_endpoint_test_app(
        validator_eligibility=eligibility,
        miner_hotkey=miner,
        endpoint_url=assignment.endpoint_url,
        block_at_registration=90,
        state=state,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=assignment.endpoint_url) as client,
    ):
        if operation == "status":
            pending = asyncio.create_task(_status(client, newcomer, assignment))
        else:
            # Two concurrent first admissions must retain one execution and both destinations.
            continuation = assignment.model_copy(
                update={"callback_url": "https://new.example/callback", "callback_context": "new"}
            )
            pending = asyncio.create_task(_accept(client, newcomer, continuation))
        try:
            await asyncio.wait_for(blocked.wait(), 1)
            async with asyncio.timeout(1):
                await _accept(client, original, assignment)
                await _accept(client, original, assignment)
            release.set()
            result = await pending
            if operation == "status":
                assert result.status_code == 403
            else:
                assert len(state.assignments) == 1
                assert len(state.destinations[assignment.assignment_id]) == 2
                assert state.assignments[assignment.assignment_id] == assignment
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


async def test_signed_request_tampering_cannot_reach_eligibility_or_execute():
    validator, miner = (bt.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob"))
    assignment = _assignment(miner)
    eligibility = AsyncMock(return_value=True)
    state = EndpointTestServerState()
    app = create_endpoint_test_app(
        validator_eligibility=eligibility,
        miner_hotkey=miner,
        endpoint_url=assignment.endpoint_url,
        block_at_registration=90,
        state=state,
    )
    path = "/v1/endpoint-assignments"
    signature = _authorization(validator, "POST", path, assignment.model_dump_json().encode())
    tampered = assignment.model_copy(update={"callback_context": "modified after signing"})
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=assignment.endpoint_url) as client,
    ):
        response = await client.post(path, content=tampered.model_dump_json(), headers={"Authorization": signature})
        assert response.status_code == 401
    eligibility.assert_not_awaited()
    assert not state.assignments


async def test_public_search_adapter_rejects_untrusted_destination_before_sending_credentials():
    from harnyx_miner.endpoint_test_server import _search_snippet

    miner = bt.Keypair.create_from_uri("//Bob")
    assignment = _assignment(miner).model_copy(update={"search_url": "https://attacker.example/search"})

    async def outbound(request):
        pytest.fail("credentials must not leave the miner")

    async with httpx.AsyncClient(transport=httpx.MockTransport(outbound)) as client:
        with pytest.raises(ValueError, match="configured assignment tooling route"):
            await _search_snippet(assignment, "parallel", "provider-secret", miner, client, "https://platform.example")


@pytest.mark.parametrize(
    "callback_url",
    [
        "https://validator.example:invalid/callback",
        "https://[invalid/callback",
        "https:///callback",
        "https://validator.example:99999/callback",
    ],
)
async def test_malformed_replacement_callback_is_rejected_without_interrupting_original(callback_url):
    original, replacement, miner = (bt.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Charlie", "//Bob"))
    assignment = _assignment(miner)
    state = EndpointTestServerState()
    release = asyncio.Event()
    delivered = []

    async def answer(request):
        await release.wait()
        return Response(text="original answer")

    async def callback(request):
        delivered.append(str(request.url))
        return httpx.Response(200, json={"durable_terminal_result": "persisted"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(callback)) as outbound:
        app = create_endpoint_test_app(
            validator_eligibility=AsyncMock(return_value=True),
            miner_hotkey=miner,
            endpoint_url=assignment.endpoint_url,
            block_at_registration=90,
            state=state,
            answerer=answer,
            client=outbound,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=assignment.endpoint_url) as client,
        ):
            await _accept(client, original, assignment)
            bad = assignment.model_copy(update={"callback_url": callback_url, "callback_context": "replacement"})
            body = bad.model_dump_json().encode()
            path = "/v1/endpoint-assignments"
            rejected = await client.post(
                path, content=body, headers={"Authorization": _authorization(replacement, "POST", path, body)}
            )
            assert rejected.status_code == 422
            assert list(state.destinations[assignment.assignment_id]) == [assignment.callback_url]
            release.set()
            await asyncio.gather(*tuple(app.state.endpoint_background_tasks))
            assert delivered == [assignment.callback_url]
            assert assignment.assignment_id in state.acknowledged
