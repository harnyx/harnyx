from __future__ import annotations

import uuid

import pytest


@pytest.mark.security
@pytest.mark.anyio("asyncio")
async def test_timeout_kills_long_handler(sandbox) -> None:
    with pytest.raises(RuntimeError, match="status 504"):
        await sandbox.invoke(
            "probe",
            payload={"mode": "sleep", "secs": 30},
            context={},
            token=str(uuid.uuid4()),
            session_id=uuid.uuid4(),
        )
