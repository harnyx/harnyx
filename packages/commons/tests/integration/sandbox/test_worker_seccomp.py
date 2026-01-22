from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]

_AGENT_MODULE = "commons.tests.integration.sandbox.seccomp_agent"


async def test_worker_seccomp_blocks_thread_creation(sandbox_launcher) -> None:
    deployment = sandbox_launcher(agent_module=_AGENT_MODULE)
    client = deployment.client

    with pytest.raises(RuntimeError) as excinfo:
        await client.invoke(
            "spawn_thread",
            payload={},
            context={},
            token=str(uuid.uuid4()),
            session_id=uuid.uuid4(),
        )

    message = str(excinfo.value)
    expected_tokens = ("PermissionError", "EPERM", "can't start new thread")
    assert any(token in message for token in expected_tokens)
