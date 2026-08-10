from __future__ import annotations

import asyncio
import uuid

import pytest

from harnyx_commons.sandbox.client import SandboxInvokeError


async def _invoke(sandbox, payload: dict[str, object], *, entrypoint: str = "probe"):
    return await sandbox.invoke(
        entrypoint,
        payload=payload,
        context={},
        token=str(uuid.uuid4()),
        session_id=uuid.uuid4(),
    )


@pytest.mark.security
@pytest.mark.anyio("asyncio")
async def test_miner_sdk_response_round_trips_through_real_sandbox(sandbox) -> None:
    response = await _invoke(sandbox, {"text": "safe response"}, entrypoint="query")

    assert response == {"text": "safe response", "citations": None}


@pytest.mark.security
@pytest.mark.anyio("asyncio")
async def test_malicious_result_is_rejected_without_affecting_concurrent_invocation(
    sandbox,
    result_ipc_barrier,
) -> None:
    barrier_id = uuid.uuid4().hex
    malicious, healthy = await asyncio.gather(
        _invoke(
            sandbox,
            {
                "mode": "result_ipc_overlap",
                "barrier_id": barrier_id,
                "barrier_url": result_ipc_barrier.url,
                "role": "malicious",
            },
        ),
        _invoke(
            sandbox,
            {
                "mode": "result_ipc_overlap",
                "barrier_id": barrier_id,
                "barrier_url": result_ipc_barrier.url,
                "role": "healthy",
            },
        ),
        return_exceptions=True,
    )

    assert isinstance(malicious, SandboxInvokeError)
    assert malicious.status_code == 500
    assert malicious.detail_code == "UnhandledException"
    assert healthy == {"role": "healthy"}
    assert result_ipc_barrier.observed(barrier_id, "malicious")
    assert result_ipc_barrier.observed(barrier_id, "healthy")
    assert not result_ipc_barrier.observed(barrier_id, "reduce")
