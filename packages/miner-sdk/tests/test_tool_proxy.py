from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import pytest
from pydantic import ValidationError

from harnyx_commons.tools.api import embed_text, fetch_page, llm_chat, search_web, tooling_info
from harnyx_commons.tools.api import (
    test_tool as invoke_test_tool,
)
from harnyx_commons.tools.proxy import ToolProxy
from harnyx_miner_sdk._internal.tool_invoker import bind_tool_invoker
from harnyx_miner_sdk.tools.embedding_models import QWEN3_CHUTES_EMBEDDING_MODEL

TEST_TOKEN = "token-123"  # noqa: S105
SESSION_ID = "00000000-0000-0000-0000-000000000001"

pytestmark = pytest.mark.anyio("asyncio")


_BUDGET = {
    "session_budget_usd": 1.0,
    "session_hard_limit_usd": 1.0,
    "session_used_budget_usd": 0.0,
    "session_remaining_budget_usd": 1.0,
}


class _ResponseInvoker:
    async def invoke(self, name: str, *, args: tuple[object, ...] = (), kwargs: dict[str, object] | None = None):
        del args, kwargs
        responses = {
            "test_tool": {"status": "ok", "echo": "ping"},
            "tooling_info": {},
            "search_web": {"data": []},
            "fetch_page": {"data": []},
            "embed_text": {
                "provider": "chutes",
                "model": QWEN3_CHUTES_EMBEDDING_MODEL,
                "input_type": "query",
                "data": [{"index": 0, "embedding": [0.1]}],
                "dimensions": 1,
            },
            "llm_chat": {"id": "response-1", "choices": [], "usage": {}},
        }
        return {
            "receipt_id": f"receipt-{name}",
            "response": responses[name],
            "results": [],
            "result_policy": "log_only",
            "budget": _BUDGET,
        }


async def test_all_tool_helpers_return_monetary_budget_without_time_budget() -> None:
    invoker = _ResponseInvoker()
    with bind_tool_invoker(invoker):
        responses = (
            await invoke_test_tool("ping"),
            await tooling_info(),
            await search_web("query", provider="parallel"),
            await fetch_page("https://example.com", provider="parallel"),
            await embed_text(
                "query",
                input_type="query",
                provider="chutes",
                model=QWEN3_CHUTES_EMBEDDING_MODEL,
            ),
            await llm_chat(
                provider="chutes",
                messages=[{"role": "user", "content": "hi"}],
                model="demo-model",
            ),
        )

    assert all(not hasattr(response, "time_budget") for response in responses)
    assert all(response.budget.session_remaining_budget_usd == 1.0 for response in responses)


async def _call_search_web_with_timeout(timeout: object) -> object:
    return await search_web("harnyx subnet", provider="parallel", timeout=timeout)


async def _call_fetch_page_with_timeout(timeout: object) -> object:
    return await fetch_page("https://example.com", provider="parallel", timeout=timeout)


async def _call_llm_chat_with_timeout(timeout: object) -> object:
    return await llm_chat(
        provider="chutes",
        messages=[{"role": "user", "content": "hi"}],
        model="demo-model",
        timeout=timeout,
    )


async def _call_tooling_info_with_timeout(timeout: object) -> object:
    return await tooling_info(timeout=timeout)


async def _call_test_tool_with_timeout(timeout: object) -> object:
    return await invoke_test_tool("ping", timeout=timeout)


@pytest.mark.parametrize(
    ("invoke_helper", "timeout"),
    (
        (_call_search_web_with_timeout, 0),
        (_call_fetch_page_with_timeout, -1.0),
        (_call_llm_chat_with_timeout, float("nan")),
        (_call_tooling_info_with_timeout, float("inf")),
        (_call_test_tool_with_timeout, float("-inf")),
        (_call_search_web_with_timeout, "5"),
        (_call_search_web_with_timeout, True),
    ),
)
async def test_tool_helpers_reject_invalid_timeout_values(
    invoke_helper: Callable[[object], Awaitable[object]],
    timeout: object,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("tool helper should reject invalid timeout before invoking the tool proxy")

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            with pytest.raises(ValidationError):
                await invoke_helper(timeout)
    finally:
        await proxy.aclose()
