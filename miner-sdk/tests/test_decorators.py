from __future__ import annotations

import pytest

from caster_miner_sdk.decorators import (
    clear_entrypoints,
    entrypoint,
    entrypoint_exists,
    get_entrypoint,
)

pytestmark = pytest.mark.anyio("asyncio")

async def test_entrypoint_registration_and_lookup() -> None:
    clear_entrypoints()

    @entrypoint("evaluate_criterion")
    async def agent(request: dict[str, str]) -> dict[str, str]:
        return {"echo": request["message"]}

    assert entrypoint_exists("evaluate_criterion")
    handler = get_entrypoint("evaluate_criterion")
    assert await handler({"message": "hello"}) == {"echo": "hello"}


async def test_duplicate_entrypoint_raises() -> None:
    clear_entrypoints()

    @entrypoint("dup")
    async def handler_a(request: object) -> None:  # pragma: no cover - simple registration
        del request

    with pytest.raises(ValueError):
        @entrypoint("dup")
        async def handler_b(request: object) -> None:  # pragma: no cover - never executed
            del request


async def test_get_entrypoint_missing_raises_key_error() -> None:
    clear_entrypoints()
    with pytest.raises(KeyError):
        get_entrypoint("missing")


async def test_entrypoint_rejects_sync_functions() -> None:
    clear_entrypoints()

    with pytest.raises(TypeError):
        @entrypoint("bad")
        def bad(request: object) -> None:  # pragma: no cover - rejected at registration
            del request
