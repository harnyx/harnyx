from __future__ import annotations

import json

import httpx
import pytest
from caster_sandbox.tools.proxy import ToolInvocationError, ToolProxy

from caster_miner_sdk._internal.tool_invoker import bind_tool_invoker
from caster_miner_sdk.api import LlmChatResult, llm_chat, search_ai, search_web

TEST_TOKEN = "token-123"  # noqa: S105
ERROR_TOKEN = "bad-token"  # noqa: S105
SESSION_ID = "00000000-0000-0000-0000-000000000001"

pytestmark = pytest.mark.anyio("asyncio")


async def test_tool_proxy_invokes_endpoint_with_token() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "result": 5})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="http://validator", transport=transport)

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=client,
    )
    try:
        result = await proxy.invoke("search_web", args=["query"], kwargs={"foo": "bar"})
    finally:
        await proxy.aclose()

    assert result == {"ok": True, "result": 5}
    assert captured["payload"] == {
        "session_id": SESSION_ID,
        "token": TEST_TOKEN,
        "tool": "search_web",
        "args": ["query"],
        "kwargs": {"foo": "bar"},
    }
    assert captured["headers"]["x-caster-token"] == "token-123"


async def test_tool_proxy_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - executed via proxy
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="http://validator", transport=transport)

    proxy = ToolProxy(
        base_url="http://validator",
        token=ERROR_TOKEN,
        session_id=SESSION_ID,
        client=client,
    )
    with pytest.raises(ToolInvocationError):
        await proxy.invoke("broken")
    await proxy.aclose()


async def test_search_web_helper_invokes_tool_proxy() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "receipt_id": "r1",
                "response": {"data": []},
                "results": [
                    {
                        "index": 0,
                        "result_id": "result-0",
                        "url": "https://example.com",
                        "note": "Snippet",
                        "title": "Example",
                    }
                ],
                "result_policy": "referenceable",
                "budget": {
                    "session_budget_usd": 1.0,
                    "session_used_budget_usd": 0.0,
                    "session_remaining_budget_usd": 1.0,
                },
            },
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await search_web("caster subnet", num=3)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "r1"
    assert result.result_policy == "referenceable"
    assert result.results[0].url == "https://example.com"
    payload = captured["payload"]
    assert payload["tool"] == "search_web"
    assert payload["kwargs"] == {"query": "caster subnet", "num": 3}


async def test_search_ai_helper_invokes_tool_proxy() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "receipt_id": "r-ai",
                "response": {"data": [{"url": "https://example.com", "note": "Snippet"}]},
                "results": [
                    {
                        "index": 0,
                        "result_id": "result-ai-0",
                        "url": "https://example.com",
                        "note": "Snippet",
                        "title": None,
                    }
                ],
                "result_policy": "referenceable",
                "cost_usd": 0.004,
                "budget": {
                    "session_budget_usd": 1.0,
                    "session_used_budget_usd": 0.0,
                    "session_remaining_budget_usd": 1.0,
                },
            },
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await search_ai("caster subnet", tools=("web",), count=3)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "r-ai"
    assert result.result_policy == "referenceable"
    assert result.results[0].url == "https://example.com"

    payload = captured["payload"]
    assert payload["tool"] == "search_ai"
    assert payload["kwargs"]["prompt"] == "caster subnet"
    assert payload["kwargs"]["tools"] == ["web"]
    assert payload["kwargs"]["count"] == 3


async def test_llm_chat_helper_invokes_tool_proxy() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "receipt_id": "chat-1",
                "response": {
                    "id": "resp-1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "hi",
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "citations": [],
                },
                "results": [
                    {
                        "index": 0,
                        "result_id": "chat-result",
                        "raw": "hi",
                    }
                ],
                "result_policy": "log_only",
                "budget": {
                    "session_budget_usd": 1.0,
                    "session_used_budget_usd": 0.0,
                    "session_remaining_budget_usd": 1.0,
                },
            },
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await llm_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="demo-model",
                temperature=0.2,
            )
    finally:
        await proxy.aclose()

    assert isinstance(result, LlmChatResult)
    assert result.receipt_id == "chat-1"
    assert result.result_policy == "log_only"
    assert result.llm.choices[0].message.content[0].text == "hi"
    assert result.llm.usage.total_tokens == 2
    assert result.results[0].raw == "hi"
    payload = captured["payload"]
    assert payload["tool"] == "llm_chat"
    assert payload["kwargs"]["model"] == "demo-model"
    assert payload["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["kwargs"]["temperature"] == 0.2
