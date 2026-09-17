"""Protect validator continuity, callback durability and failure responsibility."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import bittensor as bt
import pytest

from harnyx_commons.bittensor import build_canonical_request
from harnyx_commons.endpoint_execution import (
    EndpointAssignmentSnapshot,
    EndpointAuthority,
    EndpointExecutionWork,
    delegation_signing_bytes,
    verify_delegation,
)
from harnyx_miner_sdk.endpoint_protocol import (
    EndpointCallback,
    EndpointCallbackAcknowledgement,
    EndpointDelegation,
    EndpointDurableTerminalResult,
    EndpointMinerStatus,
    EndpointStatusResponse,
    query_digest,
)
from harnyx_miner_sdk.query import Query, Response
from harnyx_validator.application.endpoint_execution import (
    EndpointReportRejectedError,
    MinerRequestAttempt,
    ValidatorEndpointExecution,
)

pytestmark = pytest.mark.anyio


async def test_miner_restart_unknown_status_reuses_original_assignment_and_timing():
    service, work, authority, _, _ = _setup()
    service.miner.get_status.return_value = EndpointStatusResponse(state=EndpointMinerStatus.UNKNOWN)
    retried = asyncio.Event()

    async def sent(**kwargs):
        if service.miner.send_assignment.await_count == 2:
            retried.set()
            service.miner.get_status.return_value = EndpointStatusResponse(state=EndpointMinerStatus.RUNNING)
        return MinerRequestAttempt(attempted_at=datetime.now(UTC))

    service.miner.send_assignment.side_effect = sent
    try:
        await service.execute(work)
        await asyncio.wait_for(retried.wait(), 2)
        first, second = service.miner.send_assignment.call_args_list
        assert first.kwargs["assignment"] == second.kwargs["assignment"]
        assert second.kwargs["assignment"].expires_at == authority.deadline_at
        service.platform.start_endpoint.assert_not_awaited()
    finally:
        await service.stop()


async def test_active_callback_uses_authenticated_signer_without_waiting_on_owner_lookup():
    service, work, authority, _, miner = _setup()
    try:
        await service.execute(work)
        await asyncio.sleep(0)
        service.authorize_delegation = AsyncMock(side_effect=TimeoutError("owner lookup unavailable"))
        result = await service.accept_callback(**_callback(authority, work, miner))
        assert result.durable_terminal_result == EndpointDurableTerminalResult.PERSISTED
        service.authorize_delegation.assert_not_awaited()
        service.platform.fail_endpoint.assert_not_awaited()
    finally:
        await service.stop()


async def test_restart_callback_authentication_blocks_concurrent_new_miner_attempt():
    service, work, authority, _, miner = _setup()
    entered, release = asyncio.Event(), asyncio.Event()
    original = service.authorize_delegation

    async def authorize(delegation):
        if asyncio.current_task().get_name() == "recover-callback":
            entered.set()
            await release.wait()
        return await original(delegation)

    service.authorize_delegation = authorize
    callback = asyncio.create_task(
        service.accept_callback(**_callback(authority, work, miner)), name="recover-callback"
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(RuntimeError, match="callback recovery"):
            await service.execute(work)
        service.miner.send_assignment.assert_not_awaited()
        service.platform.fail_endpoint.assert_not_awaited()
        release.set()
        assert (await callback).durable_terminal_result == EndpointDurableTerminalResult.PERSISTED
    finally:
        release.set()
        await asyncio.gather(callback, return_exceptions=True)
        await service.stop()


@pytest.mark.parametrize("answer", ["unknown", "invalid", "absent", "not_attempted"])
async def test_failure_requires_actual_attempt_and_definitive_absence_of_valid_answer(answer):
    service, work, authority, _, miner = _setup(duration=0.2)
    if answer == "not_attempted":
        service.miner.send_assignment.return_value = MinerRequestAttempt()
    try:
        await service.execute(work)
        for _ in range(5):
            await asyncio.sleep(0)
        if answer in {"unknown", "invalid"}:
            error = (
                TimeoutError("Platform unavailable")
                if answer == "unknown"
                else EndpointReportRejectedError("invalid receipts")
            )
            service.platform.report_endpoint.side_effect = error
            with pytest.raises(type(error)):
                await service.accept_callback(**_callback(authority, work, miner))
        await asyncio.sleep(1.1)
        if answer in {"invalid", "absent"}:
            report = service.platform.fail_endpoint.call_args.args[1]
            assert authority.started_at <= report.attempted_at < authority.deadline_at <= report.observed_through
        else:
            service.platform.fail_endpoint.assert_not_awaited()
    finally:
        await service.stop()


def _setup(*, started=True, duration=30):
    platform_key = bt.Keypair.create_from_uri("//Alice")
    validator_key = bt.Keypair.create_from_uri("//Bob")
    miner_key = bt.Keypair.create_from_uri("//Charlie")
    identity = uuid4()
    query = Query(text="Find evidence")
    now = datetime.now(UTC)
    authority = EndpointAuthority(
        assignment_id=identity,
        validator_hotkey=validator_key.ss58_address,
        miner_hotkey=miner_key.ss58_address,
        query_digest=query_digest(query),
        nonce="a" * 64,
        endpoint_url="https://miner.example",
        callback_url=f"https://validator.example/validator/endpoint-assignments/{identity}/callback",
        search_url=f"https://platform.example/v1/endpoint-assignments/{identity}/search",
        first_scheduled_at=now,
        execution_timeout_seconds=float(duration),
        attempt_number=1,
        started_at=now if started else None,
        deadline_at=now + timedelta(seconds=duration) if started else None,
    )

    def work(value):
        return EndpointExecutionWork(
            query=query,
            delegation=EndpointDelegation(
                platform_hotkey=platform_key.ss58_address,
                body_utf8=value.model_dump_json(),
                signature_hex=platform_key.sign(delegation_signing_bytes(value)).hex(),
            ),
        )

    async def authorize(delegation):
        return verify_delegation(delegation, platform_key.ss58_address)

    initial = work(authority)
    platform, miner = AsyncMock(), AsyncMock()
    platform.saved_endpoint.return_value = EndpointAssignmentSnapshot(
        work=initial, status="awaiting_callback" if started else "created"
    )
    platform.report_endpoint.return_value = EndpointCallbackAcknowledgement(
        durable_terminal_result=EndpointDurableTerminalResult.PERSISTED
    )
    miner.send_assignment.return_value = MinerRequestAttempt(attempted_at=now)
    service = ValidatorEndpointExecution(
        platform, miner, authorize, validator_key.ss58_address, "https://validator.example"
    )
    service.start()
    return service, initial, authority, work, miner_key


def _callback(authority, work, miner_key, *, received_at=None, text="answer"):
    body = (
        EndpointCallback(
            assignment_id=authority.assignment_id,
            query_digest=authority.query_digest,
            nonce=authority.nonce,
            expires_at=authority.deadline_at,
            response=Response(text=text),
        )
        .model_dump_json()
        .encode()
    )
    path = urlsplit(authority.callback_url).path
    signature = miner_key.sign(build_canonical_request("POST", path, body)).hex()
    return dict(
        assignment_id=authority.assignment_id,
        delegation=work.delegation,
        raw_body=body,
        authorization_header=f'Bittensor ss58="{miner_key.ss58_address}",sig="{signature}"',
        received_at=received_at or datetime.now(UTC),
        signed_path=path,
    )


async def test_healthy_progress_keeps_one_task_even_when_capacity_is_full():
    service, work, authority, _, _ = _setup()
    service.capacity = 1
    try:
        await service.execute(work)
        for _ in range(5):
            await asyncio.sleep(0)
            await service.execute(work)
        assert service.miner.send_assignment.await_count == 1
        assert service.platform.start_endpoint.await_count == 0
    finally:
        await service.stop()


async def test_no_miner_send_until_start_is_durable_and_saved_timing_wins():
    service, work, authority, sign_work, _ = _setup(started=False)
    entered, release = asyncio.Event(), asyncio.Event()
    saved_start = datetime.now(UTC) - timedelta(seconds=2)
    final = authority.model_copy(update={"started_at": saved_start, "deadline_at": saved_start + timedelta(seconds=30)})

    async def start(*args):
        entered.set()
        await release.wait()
        return sign_work(final)

    service.platform.start_endpoint.side_effect = start
    try:
        await service.execute(work)
        await asyncio.wait_for(entered.wait(), 1)
        service.miner.send_assignment.assert_not_awaited()
        release.set()
        for _ in range(20):
            await asyncio.sleep(0)
            if service.miner.send_assignment.await_count:
                break
        sent = service.miner.send_assignment.call_args.kwargs["assignment"]
        assert sent.expires_at == final.deadline_at
    finally:
        await service.stop()


async def test_restart_retains_on_time_report_during_platform_outage_and_recovers_ack():
    service, work, authority, _, miner_key = _setup()
    service.platform.report_endpoint.side_effect = ConnectionError("Platform unavailable")
    callback = _callback(authority, work, miner_key)
    try:
        with pytest.raises(ConnectionError):
            await service.accept_callback(**callback)
        service.platform.saved_endpoint.assert_not_awaited()
        item = service._executions[authority.assignment_id]
        assert item.pending[0].received_at == callback["received_at"]
        for _ in range(8):
            with pytest.raises(ConnectionError):
                await service.accept_callback(**_callback(authority, work, miner_key))
        assert len(item.pending) == 1
        service.platform.report_endpoint.side_effect = None
        ack = await service.accept_callback(**callback)
        assert ack.durable_terminal_result == EndpointDurableTerminalResult.PERSISTED
        assert not item.pending
    finally:
        await service.stop()


async def test_distinct_pending_answers_are_bounded_and_invalid_candidates_release_capacity():
    service, work, authority, _, miner_key = _setup()
    service.platform.report_endpoint.side_effect = ConnectionError("Platform unavailable")
    try:
        for index in range(4):
            with pytest.raises(ConnectionError):
                await service.accept_callback(**_callback(authority, work, miner_key, text=str(index)))
        with pytest.raises(RuntimeError, match="capacity"):
            await service.accept_callback(**_callback(authority, work, miner_key, text="fifth"))
        service.platform.report_endpoint.side_effect = EndpointReportRejectedError("invalid receipts")
        with pytest.raises(EndpointReportRejectedError):
            await service.accept_callback(**_callback(authority, work, miner_key, text="0"))
        assert len(service._executions[authority.assignment_id].pending) == 3
    finally:
        await service.stop()


async def test_late_retry_does_not_invent_an_on_time_receive_event():
    service, work, authority, _, miner_key = _setup()
    try:
        with pytest.raises(ValueError, match="deadline"):
            await service.accept_callback(**_callback(authority, work, miner_key, received_at=authority.deadline_at))
        service.platform.report_endpoint.assert_not_awaited()
    finally:
        await service.stop()


async def test_saved_success_after_deadline_does_not_contact_miner():
    service, work, authority, _, _ = _setup()
    service.platform.saved_endpoint.return_value = EndpointAssignmentSnapshot(
        work=work, status="succeeded", response_json={"saved": True}
    )
    try:
        await service.execute(work)
        await asyncio.sleep(0)
        service.miner.send_assignment.assert_not_awaited()
        assert (await service.execute(work)).state == "saved"
    finally:
        await service.stop()


@pytest.mark.parametrize("admission", ["request", "callback"])
async def test_completed_executions_release_capacity_for_new_work(admission):
    service, work, authority, sign_work, miner = _setup()
    service.capacity = 1
    service.platform.saved_endpoint.return_value = EndpointAssignmentSnapshot(work=work, status="succeeded")
    try:
        await service.execute(work)
        await service._executions[authority.assignment_id].task
        identity = uuid4()
        next_authority = authority.model_copy(
            update={
                "assignment_id": identity,
                "callback_url": authority.callback_url.replace(str(authority.assignment_id), str(identity)),
            }
        )
        next_work = sign_work(next_authority)
        if admission == "request":
            assert (await service.execute(next_work)).assignment_id == identity
        else:
            result = await service.accept_callback(**_callback(next_authority, next_work, miner))
            assert result.durable_terminal_result == EndpointDurableTerminalResult.PERSISTED
        assert authority.assignment_id not in service._executions
        assert len(service._executions) == service.capacity
    finally:
        await service.stop()


@pytest.mark.parametrize("admission", ["request", "callback"])
@pytest.mark.parametrize("retained", ["active", "pending"])
async def test_admission_preserves_active_work_and_forwardable_answers(admission, retained):
    service, work, authority, sign_work, miner = _setup()
    service.capacity = 1
    try:
        if retained == "active":
            await service.execute(work)
        else:
            service.platform.report_endpoint.side_effect = ConnectionError("Platform unavailable")
            with pytest.raises(ConnectionError):
                await service.accept_callback(**_callback(authority, work, miner))
            task = service._executions[authority.assignment_id].task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # A repeated request must not replace the receive event after its task exits.
            assert (await service.execute(work)).state == "reporting"
        item = service._executions[authority.assignment_id]
        identity = uuid4()
        next_authority = authority.model_copy(
            update={
                "assignment_id": identity,
                "callback_url": authority.callback_url.replace(str(authority.assignment_id), str(identity)),
            }
        )
        next_work = sign_work(next_authority)
        with pytest.raises(RuntimeError, match="capacity"):
            if admission == "request":
                await service.execute(next_work)
            else:
                await service.accept_callback(**_callback(next_authority, next_work, miner))
        assert service._executions[authority.assignment_id] is item
        if retained == "pending":
            original_received_at = item.pending[0].received_at
            service.platform.report_endpoint.side_effect = None
            ack = await service.accept_callback(**_callback(authority, work, miner))
            assert ack.durable_terminal_result == EndpointDurableTerminalResult.PERSISTED
            assert service.platform.report_endpoint.call_args.args[1].received_at == original_received_at
    finally:
        await service.stop()


@pytest.mark.parametrize("admission", ["request", "callback"])
async def test_expired_forwarding_does_not_hold_capacity(admission, monkeypatch):
    import harnyx_validator.application.endpoint_execution as execution_module

    service, work, authority, sign_work, miner = _setup()
    service.capacity = 1
    service.platform.report_endpoint.side_effect = ConnectionError("Platform unavailable")
    try:
        with pytest.raises(ConnectionError):
            await service.accept_callback(**_callback(authority, work, miner))
        task = service._executions[authority.assignment_id].task
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        class AfterForwarding(datetime):
            @classmethod
            def now(cls, tz=None):
                return authority.report_cutoff + timedelta(seconds=1)

        monkeypatch.setattr(execution_module, "datetime", AfterForwarding)
        identity = uuid4()
        next_authority = authority.model_copy(
            update={
                "assignment_id": identity,
                "callback_url": authority.callback_url.replace(str(authority.assignment_id), str(identity)),
                "deadline_at": authority.deadline_at + timedelta(minutes=5),
                "execution_timeout_seconds": authority.execution_timeout_seconds + 300,
            }
        )
        next_work = sign_work(next_authority)
        service.platform.report_endpoint.side_effect = None
        if admission == "request":
            assert (await service.execute(next_work)).assignment_id == identity
        else:
            ack = await service.accept_callback(**_callback(next_authority, next_work, miner))
            assert ack.durable_terminal_result == EndpointDurableTerminalResult.PERSISTED
        assert authority.assignment_id not in service._executions
    finally:
        await service.stop()
