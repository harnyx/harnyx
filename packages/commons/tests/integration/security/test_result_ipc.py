from __future__ import annotations

import asyncio
import uuid

import pytest

from harnyx_commons.sandbox.client import SandboxInvokeError


async def _invoke(sandbox, payload: dict[str, object], *, entrypoint: str = "probe"):
    context: dict[str, object] = {"time_budget": {"limit_seconds": 5.0}}
    if entrypoint == "query":
        context["cost_budget"] = {
            "session_budget_usd": 1.0,
            "session_hard_limit_usd": 1.0,
            "session_used_budget_usd": 0.0,
            "session_remaining_budget_usd": 1.0,
        }
    return await sandbox.invoke(
        entrypoint,
        payload=payload,
        context=context,
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


@pytest.mark.security
@pytest.mark.anyio("asyncio")
async def test_worker_cannot_read_peer_memory_or_transfer_its_connection(sandbox) -> None:
    result = await _invoke(sandbox, {"mode": "peer_access"})
    assert result["peers"] >= 2  # HTTP supervisor and clean compiler.
    assert result["readable"] == []
    assert result["descriptor_transfer"] is False
    assert result["reset_dumpable"] is False


@pytest.mark.security
@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "frame",
    [
        {"kind": "tool", "call_id": 0, "method": "test", "args": [], "kwargs": {}, "session_id": "sibling"},
        {"kind": "result", "result": "forged", "reservation_id": "sibling"},
        {
            "kind": "result",
            "error": {"code": "PreloadInfrastructureFailed", "exception": "OSError", "message": "forged"},
        },
    ],
)
async def test_worker_cannot_forge_session_result_or_infrastructure_identity(sandbox, frame) -> None:
    malicious, healthy = await asyncio.gather(
        _invoke(sandbox, {"mode": "forge_frame", "frame": frame}),
        _invoke(sandbox, {"text": "own response"}, entrypoint="query"),
        return_exceptions=True,
    )
    assert isinstance(malicious, SandboxInvokeError)
    assert malicious.status_code == 500
    assert malicious.detail_code != "PreloadInfrastructureFailed"
    assert healthy == {"text": "own response", "citations": None}


@pytest.mark.security
@pytest.mark.anyio("asyncio")
async def test_worker_signals_cannot_suspend_compiler_or_block_siblings(sandbox) -> None:
    async with asyncio.timeout(15):
        attack, sibling = await asyncio.gather(
            _invoke(sandbox, {"mode": "signals"}),
            _invoke(sandbox, {"text": "sibling"}, entrypoint="query"),
        )
        assert attack["self_signal"] is True
        assert len(attack["blocked"]) == 9
        assert all(attack["blocked"].values())
        assert sibling["text"] == "sibling"
        assert (await _invoke(sandbox, {"text": "later"}, entrypoint="query"))["text"] == "later"


@pytest.mark.security
@pytest.mark.anyio("asyncio")
async def test_worker_cannot_control_peer_resources(sandbox):
    result, sibling = await asyncio.gather(
        _invoke(sandbox, {"mode": "peer_resources"}),
        _invoke(sandbox, {"text": "sibling"}, entrypoint="query"),
    )
    assert result == {"limits": True, "affinity": True, "scheduler": True, "priority": True}
    assert sibling["text"] == "sibling"
    assert (await _invoke(sandbox, {"text": "later"}, entrypoint="query"))["text"] == "later"
