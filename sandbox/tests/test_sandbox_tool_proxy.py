from __future__ import annotations

import json

import harnyx_sandbox.app as app_module
import harnyx_sandbox.sandbox.harness as harness_module
import httpx
import pytest
from harnyx_sandbox.tools.proxy import ToolInvocationError, ToolProxy
from pydantic import ValidationError

from harnyx_miner_sdk._internal.tool_invoker import bind_tool_invoker
from harnyx_miner_sdk.api import LlmChatResult, llm_chat, search_ai, search_web
from harnyx_miner_sdk.sandbox_headers import SESSION_ID_HEADER

TEST_TOKEN = "token-123"  # noqa: S105
ERROR_TOKEN = "bad-token"  # noqa: S105
SESSION_ID = "00000000-0000-0000-0000-000000000001"

pytestmark = pytest.mark.anyio("asyncio")


def test_tool_factory_uses_entrypoint_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class CapturingProxy:
        def __init__(
            self,
            base_url: str,
            token: str,
            *,
            session_id: str,
            timeout: float,
        ) -> None:
            captured.update(
                {
                    "base_url": base_url,
                    "token": token,
                    "session_id": session_id,
                    "timeout": timeout,
                }
            )

    monkeypatch.setattr(app_module, "ToolProxy", CapturingProxy)

    proxy = app_module._tool_factory(
        None,
        {
            "x-host-container-url": "http://validator",
            "x-platform-token": TEST_TOKEN,
            "x-session-id": SESSION_ID,
        },
    )

    assert proxy is not None
    assert captured == {
        "base_url": "http://validator",
        "token": TEST_TOKEN,
        "session_id": SESSION_ID,
        "timeout": harness_module.ENTRYPOINT_TIMEOUT_SECONDS,
    }


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
        "tool": "search_web",
        "args": ["query"],
        "kwargs": {"foo": "bar"},
    }
    assert captured["headers"]["x-platform-token"] == "token-123"
    assert captured["headers"][SESSION_ID_HEADER] == SESSION_ID


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
                    "session_hard_limit_usd": 1.0,
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
            result = await search_web(("harnyx", "subnet"), provider="parallel", num=3)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "r1"
    assert result.result_policy == "referenceable"
    assert result.results[0].url == "https://example.com"
    payload = captured["payload"]
    assert payload["tool"] == "search_web"
    assert payload["kwargs"] == {"search_queries": ["harnyx", "subnet"], "provider": "parallel", "num": 3}


async def test_search_web_helper_normalizes_plain_string_query() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "receipt_id": "r2",
                "response": {"data": []},
                "results": [],
                "result_policy": "referenceable",
                "budget": {
                    "session_budget_usd": 1.0,
                    "session_hard_limit_usd": 1.0,
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
            result = await search_web("harnyx subnet", provider="parallel", num=3)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "r2"
    payload = captured["payload"]
    assert payload["tool"] == "search_web"
    assert payload["kwargs"] == {"search_queries": ["harnyx subnet"], "provider": "parallel", "num": 3}


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
                    "session_hard_limit_usd": 1.0,
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
            result = await search_ai("harnyx subnet", provider="parallel", count=10)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "r-ai"
    assert result.result_policy == "referenceable"
    assert result.results[0].url == "https://example.com"

    payload = captured["payload"]
    assert payload["tool"] == "search_ai"
    assert payload["kwargs"]["prompt"] == "harnyx subnet"
    assert payload["kwargs"]["provider"] == "parallel"
    assert payload["kwargs"]["count"] == 10


async def test_search_ai_helper_rejects_count_below_provider_minimum() -> None:
    with pytest.raises(ValidationError) as excinfo:
        await search_ai("harnyx subnet", provider="parallel", count=3)

    assert any(
        err.get("type") == "greater_than_equal" and err.get("loc") == ("count",) and err.get("ctx") == {"ge": 10}
        for err in excinfo.value.errors()
    )


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
                    "session_hard_limit_usd": 1.0,
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
                provider="chutes",
                messages=[{"role": "user", "content": "hi"}],
                model="demo-model",
                temperature=0.2,
                thinking={"enabled": True, "effort": "high"},
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
    assert payload["kwargs"]["provider"] == "chutes"
    assert payload["kwargs"]["model"] == "demo-model"
    assert payload["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["kwargs"]["temperature"] == 0.2
    assert payload["kwargs"]["thinking"] == {"enabled": True, "effort": "high"}


async def test_llm_chat_helper_rejects_thinking_effort_and_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("llm_chat should reject invalid thinking before invoking the tool proxy")

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            with pytest.raises(ValidationError):
                await llm_chat(
                    provider="chutes",
                    messages=[{"role": "user", "content": "hi"}],
                    model="demo-model",
                    thinking={"enabled": True, "effort": "high", "budget": 1024},
                )
    finally:
        await proxy.aclose()


async def test_llm_chat_helper_rejects_coerced_thinking_scalars() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("llm_chat should reject invalid thinking before invoking the tool proxy")

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            with pytest.raises(ValidationError):
                await llm_chat(
                    provider="chutes",
                    messages=[{"role": "user", "content": "hi"}],
                    model="demo-model",
                    thinking={"enabled": "false", "budget": True},
                )
    finally:
        await proxy.aclose()


async def test_llm_chat_full_tool_loop_crosses_sandbox_proxy() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "receipt_id": "chat-loop",
                "response": {
                    "id": "resp-loop",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": [],
                                "tool_calls": [
                                    {
                                        "id": "call-2",
                                        "type": "function",
                                        "name": "lookup_weather",
                                        "arguments": '{"city":"Rome"}',
                                    }
                                ],
                                "reasoning_details": [
                                    {"type": "reasoning.encrypted", "data": "opaque-response"}
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
                "results": [],
                "result_policy": "log_only",
                "budget": {
                    "session_budget_usd": 1.0,
                    "session_hard_limit_usd": 1.0,
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
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "name": "lookup_weather",
                    "arguments": '{"city":"Paris"}',
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"temperature":19}'},
    ]
    try:
        with bind_tool_invoker(proxy):
            result = await llm_chat(
                provider="openrouter",
                model="openai/gpt-oss-20b",
                messages=messages,
                tools=[{"type": "function", "function": {"name": "lookup_weather"}}],
                tool_choice="auto",
                parallel_tool_calls=True,
            )
    finally:
        await proxy.aclose()

    assert captured["payload"]["kwargs"]["messages"] == messages
    tool_calls = result.llm.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].id == "call-2"
    assert result.llm.choices[0].message.reasoning_details == [
        {"type": "reasoning.encrypted", "data": "opaque-response"}
    ]
