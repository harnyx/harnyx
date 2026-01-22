from __future__ import annotations

import uuid

import pytest


@pytest.mark.security
@pytest.mark.anyio("asyncio")
async def test_root_fs_is_readonly(sandbox) -> None:
    response = await sandbox.invoke(
        "probe",
        payload={"mode": "fs"},
        context={},
        token=str(uuid.uuid4()),
        session_id=uuid.uuid4(),
    )
    assert response["ok_tmp"] is True
    err = response["err_root"]
    assert isinstance(err, str) and err.startswith("err:")
