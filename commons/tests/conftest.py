from __future__ import annotations

from collections.abc import Generator

import pytest

from caster_commons.tools.decorators import clear_entrypoints


@pytest.fixture(autouse=True)
def reset_entrypoints() -> Generator[None, None, None]:
    clear_entrypoints()
    yield
    clear_entrypoints()

@pytest.fixture
def anyio_backend() -> str:
    # Force AnyIO-managed tests in commons suite (unit + integration) to use asyncio only
    return "asyncio"
