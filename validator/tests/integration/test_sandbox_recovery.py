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
