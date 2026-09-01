from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.decorators import (
    clear_entrypoints,
    entrypoint,
    entrypoint_exists,
    get_entrypoint,
)
from harnyx_miner_sdk.query import Query, Response

pytestmark = pytest.mark.anyio("asyncio")

_CONTEXT = {
    "cost_budget": {
        "session_budget_usd": 1.0,
        "session_hard_limit_usd": 1.0,
        "session_used_budget_usd": 0.0,
        "session_remaining_budget_usd": 1.0,
    },
    "time_budget": {"limit_seconds": 300.0},
}


async def test_entrypoint_registration_and_lookup() -> None:
    clear_entrypoints()

    @entrypoint("query")
    async def query(query: Query, context: ContextSnapshot) -> Response:
        assert context.time_budget.limit_seconds == 300.0
        return Response(text=query.text)

    assert entrypoint_exists("query")
    handler = get_entrypoint("query")
    assert await handler({"text": "hello"}, _CONTEXT) == Response(text="hello")


async def test_query_entrypoint_allows_domain_named_parameter() -> None:
    clear_entrypoints()

    @entrypoint("query")
    async def query(request: Query, execution_context: ContextSnapshot) -> Response:
        assert execution_context.cost_budget.session_budget_usd == 1.0
        return Response(text=request.text)

    handler = get_entrypoint("query")
    assert await handler({"text": "hello"}, _CONTEXT) == Response(text="hello")


async def test_query_entrypoint_accepts_structured_envelope() -> None:
    clear_entrypoints()

    @entrypoint("query")
    async def query(request: Query, context: ContextSnapshot) -> Response:
        del context
        return Response(output={"answer": request.text})

    handler = get_entrypoint("query")
    assert await handler(
        {"text": "hello", "output_schema": {"type": "object"}},
        _CONTEXT,
    ) == Response(output={"answer": "hello"})


async def test_query_entrypoint_does_not_apply_relational_schema_conformance() -> None:
    clear_entrypoints()

    @entrypoint("query")
    async def query(request: Query, context: ContextSnapshot) -> Response:
        del context
        return Response(output={"answer": request.text})

    handler = get_entrypoint("query")
    response = await handler(
        {"text": "hello", "output_schema": {"type": "array"}},
        _CONTEXT,
    )

    assert response == Response(output={"answer": "hello"})


async def test_invalid_query_schema_prevents_miner_invocation() -> None:
    clear_entrypoints()
    invoked = False

    @entrypoint("query")
    async def query(request: Query, context: ContextSnapshot) -> Response:
        nonlocal invoked
        del context
        invoked = True
        return Response(text=request.text)

    handler = get_entrypoint("query")
    with pytest.raises(ValidationError):
        await handler(
            {"text": "hello", "output_schema": {"$ref": "https://example.com/schema"}},
            _CONTEXT,
        )

    assert invoked is False


async def test_query_entrypoint_rejects_wrong_parameter_type() -> None:
    clear_entrypoints()

    with pytest.raises(
        TypeError,
        match="query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query",
    ):
        @entrypoint("query")
        async def query(request: str, context: ContextSnapshot) -> Response:
            del context
            return Response(text=request)


async def test_query_entrypoint_rejects_wrong_context_type() -> None:
    clear_entrypoints()

    with pytest.raises(
        TypeError,
        match="query entrypoint context must be annotated as harnyx_miner_sdk.context.ContextSnapshot",
    ):
        @entrypoint("query")
        async def query(request: Query, context: dict[str, object]) -> Response:
            del context
            return Response(text=request.text)


async def test_legacy_query_receives_validated_request_only() -> None:
    clear_entrypoints()
    observed: list[Query] = []

    @entrypoint("query")
    async def query(request: Query) -> Response:
        observed.append(request)
        return Response(text=request.text)

    handler = get_entrypoint("query")
    response = await handler({"text": "hello"}, _CONTEXT)

    assert observed == [Query(text="hello")]
    assert response == Response(text="hello")


async def test_legacy_query_still_validates_context_before_invocation() -> None:
    clear_entrypoints()
    observed: list[Query] = []

    @entrypoint("query")
    async def query(request: Query) -> Response:
        observed.append(request)
        return Response(text=request.text)

    handler = get_entrypoint("query")
    with pytest.raises(ValueError):
        await handler({"text": "hello"}, {"time_budget": {"limit_seconds": 300.0}})

    assert observed == []


async def _query_with_zero_parameters() -> Response:
    return Response(text="unused")


async def _query_with_three_parameters(
    request: Query,
    context: ContextSnapshot,
    extra: object,
) -> Response:
    del context, extra
    return Response(text=request.text)


@pytest.mark.parametrize(
    "query",
    [_query_with_zero_parameters, _query_with_three_parameters],
)
async def test_query_rejects_arities_outside_supported_forms(query: object) -> None:
    clear_entrypoints()

    with pytest.raises(TypeError, match="one or two"):
        entrypoint("query")(query)  # type: ignore[arg-type]


async def test_query_entrypoint_rejects_wrong_return_type() -> None:
    clear_entrypoints()

    with pytest.raises(
        TypeError,
        match="query entrypoint return type must be harnyx_miner_sdk.query.Response",
    ):
        @entrypoint("query")
        async def query(request: Query, context: ContextSnapshot) -> str:
            del context
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
