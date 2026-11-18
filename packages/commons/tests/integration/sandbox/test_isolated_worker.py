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


async def test_worker_cannot_create_tcp_socket_or_read_control_credential(sandbox_launcher) -> None:
    source = b"""
import os
import socket
from harnyx_miner_sdk.decorators import entrypoint
@entrypoint("check_control")
async def check_control(request: dict[str, str]) -> dict[str, object]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            pass
    except PermissionError:
        network_denied = True
    else:
        network_denied = False
    return {"network_denied": network_denied,
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
    assert result == {"network_denied": True, "credential_in_environment": False}


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


async def test_workspace_volume_lifecycle_in_real_container(sandbox_launcher) -> None:
    import json
    import os
    import subprocess

    from harnyx_commons.sandbox.docker import DockerSandboxManager

    docker = os.getenv("DOCKER_CLI", "docker")

    def run(*args):
        return subprocess.run(  # noqa: S603 - Docker CLI and arguments are owned by this test
            [docker, *args], capture_output=True, text=True, check=True,
        ).stdout

    deployment = sandbox_launcher(
        agent_module="commons.tests.integration.sandbox.isolated_worker_agent", evaluation_policy=True,
    )
    inspected = json.loads(run("inspect", deployment.identifier))[0]
    volume = next(m["Name"] for m in inspected["Mounts"] if m["Destination"] == "/sandbox/queries")
    state = next(m for m in inspected["Mounts"] if m["Destination"] != "/sandbox/queries")
    manager = DockerSandboxManager(docker_binary=docker)
    results = await asyncio.gather(*(
        deployment.client.invoke(
            "workspace", payload={}, context={"time_budget": {"limit_seconds": 5.0}},
            token=uuid4().hex, session_id=uuid4(),
        ) for _ in range(10)
    ))
    assert len({r["cwd"] for r in results}) == 10
    for result in results:
        assert result["cwd"] == result["temp"] == result["module_temp"]
        assert result["module"] == "initialized"
        assert result["contents"] == "abc" and result["listed"] == ["output"]
        assert result["root_read_only"] and result["uid"] != 0
    for payload, limit, status in [({"fail": True}, 5.0, 500), ({"delay": 5}, 0.3, 504)]:
        with pytest.raises(SandboxInvokeError) as exc:
            await deployment.client.invoke(
                "workspace", payload=payload, context={"time_budget": {"limit_seconds": limit}},
                token=uuid4().hex, session_id=uuid4(),
            )
        assert exc.value.status_code == status
    async with asyncio.timeout(5):
        while run("exec", deployment.identifier, "ls", "-A", "/sandbox/queries").strip():
            await asyncio.sleep(0.05)
    assert manager.stop(deployment)
    assert volume not in run("volume", "ls", "--format", "{{.Name}}").splitlines()
    if state["Type"] == "volume":
        run("volume", "inspect", state["Name"])
