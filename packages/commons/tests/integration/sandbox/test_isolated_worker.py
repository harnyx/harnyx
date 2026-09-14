from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from harnyx_commons.sandbox.client import SandboxInvokeError

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]

INTROSPECTION_SOURCE = """
import asyncio
from harnyx_miner_sdk.decorators import entrypoint
@entrypoint("inspect_tasks")
async def inspect_tasks(request: dict[str, str]) -> dict[str, bool]:
    await asyncio.sleep(0.1 if request["role"] == "observer" else 0.2)
    found = False
    for task in list(asyncio.tasks._all_tasks):
        for frame in task.get_stack():
            headers = frame.f_locals.get("headers")
            if headers is not None and headers.get("x-platform-token") == "synthetic-victim-token":
                found = True
    return {"sibling_token_readable": found}
"""


async def test_worker_cannot_inspect_other_query_headers(sandbox_launcher) -> None:
    deployment = sandbox_launcher(
        agent_module="commons.tests.integration.sandbox.isolated_worker_agent",
        source=INTROSPECTION_SOURCE.encode(),
    )
    results = await asyncio.gather(
        *(
            deployment.client.invoke(
                "inspect_tasks",
                payload={"role": role},
                context={"time_budget": {"limit_seconds": 5.0}},
                token="synthetic-victim-token" if role == "victim" else "synthetic-observer-token",
                session_id=uuid4(),
            )
            for role in ("victim", "observer")
        )
    )
    assert results == [{"sibling_token_readable": False}, {"sibling_token_readable": False}]


async def test_ten_queries_have_separate_workers_in_real_container(sandbox_launcher) -> None:
    deployment = sandbox_launcher(agent_module="commons.tests.integration.sandbox.isolated_worker_agent")
    results = await asyncio.gather(
        *(
            deployment.client.invoke(
                "isolated",
                payload={},
                context={"time_budget": {"limit_seconds": 5.0}},
                token=uuid4().hex,
                session_id=uuid4(),
            )
            for _ in range(10)
        )
    )
    assert len({result["pid"] for result in results}) == 10
    assert {result["loads"] for result in results} == {1}


async def test_blocked_loop_times_out_and_replaces_in_real_container(sandbox_launcher) -> None:
    deployment = sandbox_launcher(agent_module="commons.tests.integration.sandbox.isolated_worker_agent")
    client = deployment.client
    first = await client.invoke(
        "isolated", payload={}, context={"time_budget": {"limit_seconds": 5.0}}, token=uuid4().hex, session_id=uuid4()
    )
    async with client.admission(0.4, uuid4().hex) as admission:
        with pytest.raises(SandboxInvokeError) as error:
            await client.invoke(
                "isolated",
                payload={"block": True},
                context={"time_budget": {"limit_seconds": 0.4}},
                token=uuid4().hex,
                session_id=uuid4(),
                admission=admission,
            )
        assert error.value.status_code == 504
    replacement = await client.invoke(
        "isolated", payload={}, context={"time_budget": {"limit_seconds": 5.0}}, token=uuid4().hex, session_id=uuid4()
    )
    assert replacement["pid"] != first["pid"]
    assert replacement["loads"] == 1


async def test_worker_cannot_authorize_host_control_requests(sandbox_launcher) -> None:
    source = b"""
import os
import httpx
from harnyx_miner_sdk.decorators import entrypoint
@entrypoint("check_control")
async def check_control(request: dict[str, str]) -> dict[str, object]:
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        health = await client.get("/healthz")
        codes = []
        for path in ("/admission", "/admission/release", "/entry/check_control"):
            response = await client.post(path, json={}, headers={"x-platform-token": "test-tool-token"})
            codes.append(response.status_code)
    return {"health": health.status_code, "control_statuses": codes,
            "credential_in_environment": "SANDBOX_CONTROL_TOKEN" in os.environ}
"""
    deployment = sandbox_launcher(agent_module="commons.tests.integration.sandbox.isolated_worker_agent", source=source)
    result = await deployment.client.invoke(
        "check_control",
        payload={},
        context={"time_budget": {"limit_seconds": 5}},
        token=uuid4().hex,
        session_id=uuid4(),
    )
    assert result == {"health": 200, "control_statuses": [401, 401, 401], "credential_in_environment": False}


async def test_stopped_compiler_cannot_prevent_container_worker_timeout(sandbox_launcher) -> None:
    import shutil
    import subprocess

    source = b"""
import asyncio
import os
from harnyx_miner_sdk.decorators import entrypoint
@entrypoint("wait")
async def wait(request: dict[str, bool]) -> dict[str, int]:
    if request.get("block"):
        print("RECOVERY_QUERY_STARTED", flush=True)
        await asyncio.sleep(60)
    return {"compiler": os.getppid()}
"""
    deployment = sandbox_launcher(agent_module="commons.tests.integration.sandbox.isolated_worker_agent", source=source)
    client = deployment.client
    docker = shutil.which("docker") or "docker"

    async def invoke(payload, limit=5):
        return await client.invoke(
            "wait",
            payload=payload,
            context={"time_budget": {"limit_seconds": limit}},
            token=uuid4().hex,
            session_id=uuid4(),
        )

    first = await invoke({})
    query = asyncio.create_task(invoke({"block": True}, 3))
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
                [
                    docker,
                    "exec",
                    deployment.identifier,
                    "python",
                    "-c",
                    "import os,signal,sys; os.kill(int(sys.argv[1]), signal.SIGSTOP)",
                    str(first["compiler"]),
                ],
                check=True,
                capture_output=True,
            )
            with pytest.raises(SandboxInvokeError) as failure:
                await query
            assert failure.value.status_code == 504
            replacement = await invoke({})
            assert replacement["compiler"] != first["compiler"]
    finally:
        if not query.done():
            query.cancel()
        await asyncio.gather(query, return_exceptions=True)
