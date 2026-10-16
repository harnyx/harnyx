from __future__ import annotations

import asyncio
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import ScriptArtifactSpec
from harnyx_validator.application.scheduler import EvaluationScheduler, SchedulerConfig
from validator.tests.application.test_scheduler import (
    DummyEvaluationRecordStore,
    DummyProgressRecorder,
    DummyReceiptLog,
    _task,
)
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = [pytest.mark.anyio("asyncio"), pytest.mark.integration]


async def test_scheduler_removes_unresponsive_real_deployment(sandbox_launcher):
    deployment = sandbox_launcher("validator.tests.integration.sandbox.recovery_agent")
    manager = DockerSandboxManager()
    stopped = []
    invocations = []

    class DeploymentManager:
        def start(self, options):
            return deployment

        def stop(self, target):
            confirmed = manager.stop(target)
            stopped.append(confirmed)
            return confirmed

    class Orchestrator:
        async def evaluate(self, request, **kwargs):
            invocations.append(request.session_id)
            invocation = asyncio.create_task(
                deployment.client.invoke(
                    "query",
                    payload={"text": "unavailable"},
                    context=_context(3),
                    token=request.token,
                    session_id=request.session_id,
                )
            )
            docker = shutil.which("docker") or "docker"
            try:
                async with asyncio.timeout(10):
                    while True:
                        marker = await asyncio.to_thread(
                            subprocess.run,
                            [docker, "logs", deployment.identifier],
                            capture_output=True,
                        )
                        if b"RECOVERY_QUERY_STARTED" in marker.stdout:
                            break
                        await asyncio.sleep(0.02)
                    await asyncio.to_thread(
                        subprocess.run,
                        [docker, "pause", deployment.identifier],
                        check=True,
                        capture_output=True,
                    )
                await invocation
            finally:
                if not invocation.done():
                    invocation.cancel()
                await asyncio.gather(invocation, return_exceptions=True)
            raise AssertionError("unresponsive container returned a result")

    task = _task("recovery")
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0, miner_hotkey_ss58="miner")
    with ThreadPoolExecutor(max_workers=2) as executor:
        scheduler = EvaluationScheduler(
            tasks=(task,),
            subtensor_client=FakeSubtensorClient(),
            sandbox_manager=DeploymentManager(),
            session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
            evaluation_records=DummyEvaluationRecordStore(),
            receipt_log=DummyReceiptLog(),
            blocking_executor=executor,
            orchestrator_factory=lambda _client: Orchestrator(),
            sandbox_options_factory=lambda _artifact: SandboxOptions(image="unused", container_name="unused"),
            clock=lambda: datetime.now(UTC),
            config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=3),
            progress=DummyProgressRecorder(),
        )
        async with asyncio.timeout(40):
            result = await scheduler.run_assigned_task(
                batch_id=uuid4(),
                artifact=artifact,
                task=task,
                attempt_number=1,
                max_attempts=1,
                assignment_token=uuid4().hex,
            )
    assert result.result is None
    assert len(invocations) == 1
    assert stopped == [True]
    inspection = subprocess.run(["docker", "inspect", deployment.identifier], capture_output=True)  # noqa: S603,S607
    assert inspection.returncode != 0
    replacement = sandbox_launcher("validator.tests.integration.sandbox.recovery_agent")
    assert replacement.identifier != deployment.identifier
    recovered = await replacement.client.invoke(
        "query",
        payload={"text": "recovered"},
        context=_context(5),
        token=uuid4().hex,
        session_id=uuid4(),
    )
    assert recovered["text"] == "recovered"


def _context(seconds):
    return {
        "time_budget": {"limit_seconds": seconds},
        "cost_budget": {
            "session_budget_usd": 0.75,
            "session_hard_limit_usd": 0.75,
            "session_used_budget_usd": 0.0,
            "session_remaining_budget_usd": 0.75,
        },
    }


_OOM_SOURCE = b"""
import asyncio
import os
from harnyx_miner_sdk.decorators import entrypoint

@entrypoint("probe")
async def probe(request: dict[str, object]) -> dict[str, object]:
    mode = request.get("mode")
    if mode == "oom":
        await asyncio.sleep(1)
        blocks = []
        while True:
            blocks.append(bytearray(4 * 1024 * 1024))
    if mode == "sleep":
        print("OOM_PROBE_SLEEP_STARTED", flush=True)
        await asyncio.sleep(8)
    if mode == "sigkill":
        os.kill(os.getpid(), 9)
    return {"mode": mode, "pid": os.getpid()}
"""


async def test_real_worker_oom_drains_without_losing_healthy_sibling(sandbox_launcher):
    """A kernel OOM must close admission, preserve survivors and leave replacements clean."""
    from harnyx_commons.sandbox.client import SandboxInvokeError
    from harnyx_miner_sdk.sandbox_protocol import SANDBOX_OOM_AFFECTED, SANDBOX_OOM_DRAINING

    deployment = sandbox_launcher(
        "validator.tests.integration.sandbox.recovery_agent",
        source=_OOM_SOURCE,
        evaluation_policy=True,
    )
    assert (await deployment.client.resource_status()).counter_status == "ready"

    async def invoke(mode):
        return await deployment.client.invoke(
            "probe",
            payload={"mode": mode},
            context=_context(20),
            token=uuid4().hex,
            session_id=uuid4(),
        )

    failed, healthy = await asyncio.gather(invoke("oom"), invoke("sleep"), return_exceptions=True)
    assert isinstance(failed, SandboxInvokeError)
    assert failed.detail_code == SANDBOX_OOM_AFFECTED
    assert isinstance(healthy, dict) and healthy["mode"] == "sleep"
    status = await deployment.client.resource_status()
    assert status.draining and status.oom_kill_count > 0
    with pytest.raises(SandboxInvokeError) as rejected:
        async with deployment.client.admission(5, uuid4().hex):
            pytest.fail("drained sandbox admitted work")
    assert rejected.value.detail_code == SANDBOX_OOM_DRAINING
    assert DockerSandboxManager().stop(deployment)
    replacement = sandbox_launcher(
        "validator.tests.integration.sandbox.recovery_agent",
        source=_OOM_SOURCE,
        evaluation_policy=True,
    )
    assert replacement.identifier != deployment.identifier
    status = await replacement.client.resource_status()
    assert status.counter_status == "ready" and not status.draining and status.oom_kill_count == 0
    result = await replacement.client.invoke(
        "probe",
        payload={"mode": "healthy"},
        context=_context(5),
        token=uuid4().hex,
        session_id=uuid4(),
    )
    assert result["mode"] == "healthy"


async def test_real_sigkill_without_oom_remains_infrastructure_failure(sandbox_launcher):
    """Exit137 and a broken result socket alone must never become miner OOM evidence."""
    from harnyx_commons.sandbox.client import SandboxInvokeError
    from harnyx_miner_sdk.sandbox_protocol import SANDBOX_OOM_AFFECTED

    deployment = sandbox_launcher(
        "validator.tests.integration.sandbox.recovery_agent",
        source=_OOM_SOURCE,
        evaluation_policy=True,
    )
    with pytest.raises(SandboxInvokeError) as failed:
        await deployment.client.invoke(
            "probe",
            payload={"mode": "sigkill"},
            context=_context(10),
            token=uuid4().hex,
            session_id=uuid4(),
        )
    assert failed.value.detail_code != SANDBOX_OOM_AFFECTED
    status = await deployment.client.resource_status()
    assert status.counter_status == "ready" and status.oom_kill_count == 0 and not status.draining


@pytest.mark.parametrize("victim", ["compiler", "supervisor"])
async def test_real_critical_process_oom_uses_positive_daemon_evidence(sandbox_launcher, victim):
    """Compiler OOM preserves sockets; supervisor OOM classifies only the lost active result."""
    from harnyx_commons.sandbox.client import SandboxInvokeError
    from harnyx_miner_sdk.sandbox_protocol import SANDBOX_OOM_AFFECTED

    deployment = sandbox_launcher(
        "validator.tests.integration.sandbox.recovery_agent",
        source=_OOM_SOURCE,
        evaluation_policy=True,
    )
    active = asyncio.create_task(
        deployment.client.invoke(
            "probe",
            payload={"mode": "sleep"},
            context=_context(25),
            token=uuid4().hex,
            session_id=uuid4(),
        )
    )
    try:
        async with asyncio.timeout(10):
            while True:
                logs = await asyncio.to_thread(
                    subprocess.run, ["docker", "logs", deployment.identifier], capture_output=True
                )
                if b"OOM_PROBE_SLEEP_STARTED" in logs.stdout:
                    break
                await asyncio.sleep(0.05)
        # This is a trusted local fault injector, outside the restricted miner worker.
        # Raising the intended victim's score makes kernel OOM selection reproducible.
        selection = (
            "from pathlib import Path; p=next(p.parent for p in Path('/proc').glob('[0-9]*/cmdline') "
            "if b'harnyx_sandbox.sandbox.compiler' in p.read_bytes() and b'-c' not in p.read_bytes()); "
            "(p/'oom_score_adj').write_text('1000')"
            if victim == "compiler"
            else "from pathlib import Path; Path('/proc/1/oom_score_adj').write_text('1000')"
        )
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", "--user", "0", deployment.identifier, "python", "-c", selection],
            capture_output=True,
            check=True,
            timeout=10,
        )
        allocated = await asyncio.to_thread(
            subprocess.run,
            [
                "docker",
                "exec",
                deployment.identifier,
                "python",
                "-c",
                "blocks=[]\nwhile True: blocks.append(bytearray(16*1024*1024))",
            ],
            capture_output=True,
            timeout=20,
        )
        assert allocated.returncode != 0
        outcome = (await asyncio.gather(active, return_exceptions=True))[0]
        if victim == "compiler":
            assert isinstance(outcome, dict) and outcome["mode"] == "sleep"
        else:
            assert isinstance(outcome, SandboxInvokeError)
            assert outcome.detail_code == SANDBOX_OOM_AFFECTED
        assert (await deployment.client.resource_status()).draining
    finally:
        if not active.done():
            active.cancel()
        await asyncio.gather(active, return_exceptions=True)
