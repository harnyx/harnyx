from __future__ import annotations

import uuid

import pytest


@pytest.mark.security
@pytest.mark.anyio("asyncio")
async def test_pids_limit_enforced(sandbox) -> None:
    response = await sandbox.invoke(
        "probe",
        payload={"mode": "pids"},
        context={"time_budget": {"limit_seconds": 5.0}},
        token=str(uuid.uuid4()),
        session_id=uuid.uuid4(),
    )
    assert isinstance(response["spawned"], str)
    assert response["spawned"].startswith("err:")
