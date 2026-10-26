"""Admission ownership must survive the separation of HTTP client and server tasks."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from typing import cast

import httpx
import pytest
import uvicorn
from fastapi import HTTPException
from harnyx_sandbox import app as sandbox_app
from harnyx_sandbox.sandbox.harness import EntrypointRequest, SandboxHarness, _EntrypointContext

from harnyx_commons.sandbox.docker import HttpSandboxClient
from harnyx_miner_sdk.sandbox_protocol import AdmissionRequest, SandboxAdmission
from harnyx_validator.application.dto.evaluation import MinerTaskWorkAssignment
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.scheduler import SchedulerConfig
from harnyx_validator.application.services import evaluation_runner as runner_module
from harnyx_validator.runtime.platform_work_worker import _ArtifactGroupState, _AssignedArtifactGroup
from validator.tests.application.test_evaluation_runner import _assigned_task_test_context, _successful_execution

pytestmark = [pytest.mark.anyio("asyncio"), pytest.mark.integration, pytest.mark.local_integration]
_ASSIGNMENT_TOKEN = "test-assignment"  # noqa: S105 - fixed test-only assignment credential
_CONTROL_TOKEN = "test-control"  # noqa: S105 - fixed test-only control credential


@pytest.fixture
async def admission_server(tmp_path, monkeypatch):
    source = tmp_path / "agent.py"
    source.write_text(
        "import asyncio\n"
        "from harnyx_miner_sdk.decorators import entrypoint\n"
        "@entrypoint('probe')\n"
        "async def probe(request: dict[str, object]) -> dict[str, object]:\n"
        "    await asyncio.sleep(0.05)\n"
        "    return {'ok': True}\n"
    )
    harness = SandboxHarness(artifact_path=str(source))
    warm = await harness.reserve(AdmissionRequest(limit_seconds=5.0), {})
    harness.release(warm)
    harness._slots = asyncio.Semaphore(1)
    monkeypatch.setattr(sandbox_app, "sandbox_harness", harness)
    monkeypatch.setattr(sandbox_app.app.state, "control_token", _CONTROL_TOKEN, raising=False)
    ready = asyncio.Event()
    sending = asyncio.Event()
    response_gate = asyncio.Event()

    async def app(scope, receive, send):
        async def send_response(message):
            if (b"x-test-pause-response", b"true") in scope.get("headers", []) and message[
                "type"
            ] == "http.response.start":
                sending.set()
                await response_gate.wait()
            await send(message)

        await sandbox_app.app(scope, receive, send_response)

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            ready.set()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        server = ReadyServer(uvicorn.Config(app, log_level="error", access_log=False, lifespan="off", ws="none"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await asyncio.wait_for(ready.wait(), 10)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{listener.getsockname()[1]}",
                headers={"x-sandbox-control-token": _CONTROL_TOKEN, "x-platform-token": _ASSIGNMENT_TOKEN},
                timeout=5.0,
            ) as client:
                yield harness, client, sending
        finally:
            response_gate.set()
            server.should_exit = True
            # A failed ownership assertion must not leave a paused server task behind.
            for task in tuple(server.server_state.tasks):
                task.cancel()
            await asyncio.wait_for(serving, 10)
            await harness.close()


@pytest.mark.parametrize("stage", ("waiting", "sending"))
async def test_http_disconnect_releases_admission_ownership(admission_server, monkeypatch, stage):
    """A cancelled client cannot retain either a pending waiter or an unsent allocation."""
    harness, client, sending = admission_server
    occupied = await harness.reserve(AdmissionRequest(limit_seconds=5.0), {}) if stage == "waiting" else None
    entered = asyncio.Event()
    settled = asyncio.Event()
    reserve = harness.reserve

    async def observed_reserve(*args, **kwargs):
        entered.set()
        try:
            return await reserve(*args, **kwargs)
        finally:
            settled.set()

    monkeypatch.setattr(harness, "reserve", observed_reserve)
    request = asyncio.create_task(
        client.post(
            "/admission",
            json={"limit_seconds": 5.0},
            headers={"x-test-pause-response": "true"} if stage == "sending" else {},
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if stage == "sending":
            await asyncio.wait_for(sending.wait(), 2)
            assert len(harness._reservations) == 1
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        async with asyncio.timeout(0.5):
            await settled.wait()
            while len(harness._reservations) != (1 if stage == "waiting" else 0):
                await asyncio.sleep(0.005)
        if occupied is not None:
            assert set(harness._reservations) == {occupied.reservation_id}
            harness.release(occupied)
        response = await asyncio.wait_for(client.post("/admission", json={"limit_seconds": 1.0}), 0.5)
        response.raise_for_status()
        admission = SandboxAdmission.model_validate(response.json())
        assert (admission.deadline_monotonic_ns - time.monotonic_ns()) / 1e9 > 0.8
        release = await client.post("/admission/release", json=admission.model_dump())
        release.raise_for_status()
        assert not harness._reservations
        assert not harness._slots.locked()
    finally:
        request.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await request
        if occupied is not None:
            harness.release(occupied)


@pytest.mark.parametrize("status", (504, 503))
async def test_http_admission_errors_keep_their_existing_response(admission_server, monkeypatch, status):
    """Moving admission into response ownership must preserve settled timeout and OOM errors."""
    harness, client, _ = admission_server
    detail = (
        {"error": "admission deadline exceeded", "exception": "TimeoutError"}
        if status == 504
        else {"code": "SandboxOomDraining"}
    )

    async def failed(*_args, **_kwargs):
        raise HTTPException(status, detail=detail)

    monkeypatch.setattr(harness, "reserve", failed)
    response = await client.post("/admission", json={"limit_seconds": 1.0})
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert not harness._reservations
    assert not harness._slots.locked()


async def test_successful_http_admission_remains_usable_after_connection_closes(admission_server):
    """Normal HTTP disconnect after response completion must not revoke a usable reservation."""
    harness, client, _ = admission_server
    response = await client.post("/admission", json={"limit_seconds": 2.0}, headers={"Connection": "close"})
    response.raise_for_status()
    admission = SandboxAdmission.model_validate(response.json())
    assert set(harness._reservations) == {admission.reservation_id}
    result = await harness.invoke(
        "probe",
        EntrypointRequest(
            payload={},
            context=_EntrypointContext.model_validate({"time_budget": {"limit_seconds": 2.0}}),
            admission=admission,
        ),
    )
    assert result == {"ok": True}
    assert not harness._reservations
    assert not harness._slots.locked()


async def test_runner_window_cancellation_does_not_block_later_http_admission(
    admission_server,
    tmp_path,
    monkeypatch,
):
    """The abandoned first-window retry cannot steal capacity from a live second window."""
    harness, http, _ = admission_server
    occupied = await harness.reserve(AdmissionRequest(limit_seconds=5.0), {})
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(tmp_path)
    runner._config = SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=0.6)
    monkeypatch.setattr(runner_module, "TASK_START_WAIT_SECONDS", 1.2)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    group = _AssignedArtifactGroup(
        artifact_id=artifact.artifact_id,
        state=_ArtifactGroupState.OPEN,
        assignment_queue=asyncio.Queue(),
        close_requested=asyncio.Event(),
        result_queue=asyncio.Queue(),
        assignment_records={},
        starting_records={},
        monotonic_clock=time.monotonic,
        dispatch_start_lease_seconds=1.65,
    )
    reserve = harness.reserve
    requests = 0

    async def observed_reserve(*args, **kwargs):
        nonlocal requests
        requests += 1
        if requests == 3:
            # The first window's retry is now disconnected; the second window is live.
            harness.release(occupied)
        return await reserve(*args, **kwargs)

    monkeypatch.setattr(harness, "reserve", observed_reserve)

    class Orchestrator:
        async def execute(self, request, **_kwargs):
            assert request.execution_time_limit_seconds == 0.6
            return _successful_execution(request)

    group.put_nowait(assignment)
    group.mark_dispatch_ready()
    claim = group.claim_nowait_for_dispatch()
    try:
        outcome = await asyncio.wait_for(
            runner._run_admitted_assignment(
                runner._assigned_attempt_lifecycle(batch_id=batch_id, artifact=artifact, claimed=claim),
                orchestrator=cast(TaskRunOrchestrator, Orchestrator()),
                sandbox_client=HttpSandboxClient(str(http.base_url), control_token=_CONTROL_TOKEN, client=http),
            ),
            3.0,
        )
        assert outcome.execution_to_queue is not None
        assert claim.record.session_id is not None
        assert requests == 3
        assert group.result_queue.empty()
        assert not harness._reservations
        assert not harness._slots.locked()
    finally:
        harness.release(occupied)
