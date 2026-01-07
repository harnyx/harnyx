from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    # Force AnyIO-managed tests in validator suite (unit + integration) to use asyncio only
    return "asyncio"
