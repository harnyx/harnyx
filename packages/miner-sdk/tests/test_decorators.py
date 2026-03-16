from __future__ import annotations

import pytest

from harnyx_miner_sdk.decorators import (
    clear_entrypoints,
    entrypoint,
    entrypoint_exists,
    get_entrypoint,
)
from harnyx_miner_sdk.query import Query, Response

pytestmark = pytest.mark.anyio("asyncio")


async def test_entrypoint_registration_and_lookup() -> None:
    clear_entrypoints()

    @entrypoint("query")
    async def query(query: Query) -> Response:
        return Response(text=query.text)

    assert entrypoint_exists("query")
    handler = get_entrypoint("query")
    assert await handler({"text": "hello"}) == Response(text="hello")


async def test_query_entrypoint_allows_domain_named_parameter() -> None:
    clear_entrypoints()

    @entrypoint("query")
    async def query(request: Query) -> Response:
        return Response(text=request.text)

    handler = get_entrypoint("query")
    assert await handler({"text": "hello"}) == Response(text="hello")


async def test_query_entrypoint_rejects_wrong_parameter_type() -> None:
    clear_entrypoints()

    with pytest.raises(
        TypeError,
        match="query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query",
    ):
        @entrypoint("query")
        async def query(request: str) -> Response:
            return Response(text=request)


async def test_query_entrypoint_rejects_wrong_return_type() -> None:
    clear_entrypoints()

    with pytest.raises(
        TypeError,
        match="query entrypoint return type must be harnyx_miner_sdk.query.Response",
    ):
        @entrypoint("query")
        async def query(request: Query) -> str:
            return request.text


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
