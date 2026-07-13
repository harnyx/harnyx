from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx
import pytest
from pydantic import ValidationError

from harnyx_commons.tools.api import (
    LlmChatResult,
    embed_text,
    fetch_page,
    llm_chat,
    search_ai,
    search_web,
    tooling_info,
)
from harnyx_commons.tools.api import (
    test_tool as invoke_test_tool,
)
from harnyx_commons.tools.proxy import ToolInvocationError, ToolProxy
from harnyx_miner_sdk._internal.tool_invoker import bind_tool_invoker
from harnyx_miner_sdk.llm import LlmInputImageData, LlmInputImagePart, LlmInputTextPart, LlmMessage
from harnyx_miner_sdk.sandbox_headers import SESSION_ID_HEADER
from harnyx_miner_sdk.tools import proxy as proxy_module

TEST_TOKEN = "token-123"  # noqa: S105
ERROR_TOKEN = "bad-token"  # noqa: S105
SESSION_ID = "00000000-0000-0000-0000-000000000001"

pytestmark = pytest.mark.anyio("asyncio")


def _tool_response_payload(
    *,
    receipt_id: str,
    response: object,
    result_policy: str = "log_only",
) -> dict[str, object]:
    return {
        "receipt_id": receipt_id,
        "response": response,
        "results": [],
        "result_policy": result_policy,
        "budget": {
            "session_budget_usd": 1.0,
            "session_hard_limit_usd": 1.0,
            "session_used_budget_usd": 0.0,
            "session_remaining_budget_usd": 1.0,
        },
    }


def test_resolve_base_url_host_rewrites_localhost_to_ip_literal() -> None:
    resolved = proxy_module._resolve_base_url_host("http://localhost:43211")

    assert resolved.startswith("http://")
    assert ":43211" in resolved
    assert "localhost" not in resolved


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
    assert payload["kwargs"] == {"provider": "parallel", "search_queries": ["harnyx", "subnet"], "num": 3}


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
    assert payload["kwargs"] == {"provider": "parallel", "search_queries": ["harnyx subnet"], "num": 3}


async def test_search_web_helper_invokes_tool_proxy_with_timeout() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="r3",
                response={"data": []},
                result_policy="referenceable",
            ),
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await search_web("harnyx subnet", provider="parallel", num=3, timeout=5)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "r3"
    payload = captured["payload"]
    assert payload["tool"] == "search_web"
    assert payload["kwargs"] == {
        "provider": "parallel",
        "search_queries": ["harnyx subnet"],
        "num": 3,
        "timeout": 5.0,
    }


async def test_embed_text_helper_invokes_tool_proxy() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="emb-1",
                response={
                    "provider": "openrouter",
                    "model": "qwen/qwen3-embedding-8b",
                    "input_type": "query",
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "dimensions": 3,
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                },
            ),
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await embed_text(
                "What is Harnyx?",
                input_type="query",
                provider="openrouter",
                model="qwen/qwen3-embedding-8b",
                instruction="Given a web search query, retrieve relevant passages that answer the query",
                timeout=5,
            )
    finally:
        await proxy.aclose()

    assert result.receipt_id == "emb-1"
    assert result.response.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.response.usage is not None
    assert result.response.usage.total_tokens == 8
    payload = captured["payload"]
    assert payload["tool"] == "embed_text"
    assert payload["kwargs"] == {
        "provider": "openrouter",
        "model": "qwen/qwen3-embedding-8b",
        "texts": ["What is Harnyx?"],
        "input_type": "query",
        "instruction": "Given a web search query, retrieve relevant passages that answer the query",
        "timeout": 5.0,
    }


async def test_embed_text_helper_forwards_openrouter_provider_extra() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="emb-openrouter-extra",
                response={
                    "provider": "openrouter",
                    "model": "qwen/qwen3-embedding-8b",
                    "input_type": "query",
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "dimensions": 3,
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                },
            ),
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await embed_text(
                "What is Harnyx?",
                input_type="query",
                provider="openrouter",
                model="qwen/qwen3-embedding-8b",
                provider_extra={"provider": {"only": ["nebius"], "allow_fallbacks": False}},
            )
    finally:
        await proxy.aclose()

    assert result.receipt_id == "emb-openrouter-extra"
    payload = captured["payload"]
    assert payload["tool"] == "embed_text"
    assert payload["kwargs"]["provider"] == "openrouter"
    assert payload["kwargs"]["provider_extra"] == {
        "provider": {"only": ["nebius"], "allow_fallbacks": False}
    }


async def test_embed_text_helper_rejects_chutes_provider_extra_before_proxy_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("embed_text should reject chutes provider_extra before invoking the tool proxy")

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            with pytest.raises(ValidationError):
                await embed_text(
                    "What is Harnyx?",
                    input_type="query",
                    provider="chutes",
                    model="Qwen/Qwen3-Embedding-8B-TEE",
                    provider_extra={"provider": {"only": ["nebius"]}},
                )
    finally:
        await proxy.aclose()


async def test_search_web_helper_rejects_removed_start_pagination() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("search_web should reject unsupported start before invoking the tool proxy")

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            with pytest.raises(ValidationError):
                await search_web("harnyx subnet", provider="parallel", start=10)
    finally:
        await proxy.aclose()


async def test_tooling_info_helper_invokes_tool_proxy() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "receipt_id": "info-1",
                "response": {
                    "tool_names": ["search_web"],
                    "pricing": {
                        "search_web": {
                            "kind": "per_referenceable_result",
                            "usd_per_referenceable_result": 0.0001,
                        }
                    },
                },
                "results": [
                    {
                        "index": 0,
                        "result_id": "info-result",
                        "raw": {"tool_names": ["search_web"]},
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
            result = await tooling_info()
    finally:
        await proxy.aclose()

    assert result.receipt_id == "info-1"
    assert result.response["tool_names"] == ["search_web"]
    payload = captured["payload"]
    assert payload["tool"] == "tooling_info"
    assert payload["kwargs"] == {}


async def test_tooling_info_helper_invokes_tool_proxy_with_timeout() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="info-2",
                response={"tool_names": ["search_web"], "pricing": {}},
            ),
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await tooling_info(timeout=5)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "info-2"
    payload = captured["payload"]
    assert payload["tool"] == "tooling_info"
    assert payload["kwargs"] == {"timeout": 5.0}


async def test_test_tool_helper_invokes_tool_proxy() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="test-1",
                response={"status": "ok", "echo": "ping"},
            ),
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await invoke_test_tool("ping")
    finally:
        await proxy.aclose()

    assert result.receipt_id == "test-1"
    assert result.response.echo == "ping"
    payload = captured["payload"]
    assert payload["tool"] == "test_tool"
    assert payload["args"] == ["ping"]
    assert payload["kwargs"] == {}


async def test_test_tool_helper_invokes_tool_proxy_with_timeout() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="test-2",
                response={"status": "ok", "echo": "ping"},
            ),
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await invoke_test_tool("ping", timeout=5)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "test-2"
    payload = captured["payload"]
    assert payload["tool"] == "test_tool"
    assert payload["args"] == ["ping"]
    assert payload["kwargs"] == {"timeout": 5.0}


async def test_fetch_page_helper_invokes_tool_proxy() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "receipt_id": "page-1",
                "response": {
                    "data": [{"url": "https://example.com", "content": "page body", "title": "Example"}]
                },
                "results": [
                    {
                        "index": 0,
                        "result_id": "page-result",
                        "url": "https://example.com",
                        "note": "page body",
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
            result = await fetch_page("https://example.com", provider="parallel")
    finally:
        await proxy.aclose()

    assert result.receipt_id == "page-1"
    assert result.results[0].url == "https://example.com"
    assert result.response.data[0].content == "page body"
    payload = captured["payload"]
    assert payload["tool"] == "fetch_page"
    assert payload["kwargs"] == {"provider": "parallel", "url": "https://example.com"}


async def test_fetch_page_helper_invokes_tool_proxy_with_timeout() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "receipt_id": "page-1",
                "response": {
                    "data": [{"url": "https://example.com", "content": "page body", "title": "Example"}]
                },
                "results": [
                    {
                        "index": 0,
                        "result_id": "page-result",
                        "url": "https://example.com",
                        "note": "page body",
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
            result = await fetch_page("https://example.com", provider="parallel", timeout=5)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "page-1"
    payload = captured["payload"]
    assert payload["tool"] == "fetch_page"
    assert payload["kwargs"] == {"provider": "parallel", "url": "https://example.com", "timeout": 5.0}


async def test_search_ai_helper_invokes_tool_proxy_with_timeout() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="ai-1",
                response={"data": []},
                result_policy="referenceable",
            ),
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            result = await search_ai("harnyx subnet", provider="parallel", count=10, timeout=5)
    finally:
        await proxy.aclose()

    assert result.receipt_id == "ai-1"
    payload = captured["payload"]
    assert payload["tool"] == "search_ai"
    assert payload["kwargs"] == {
        "provider": "parallel",
        "prompt": "harnyx subnet",
        "count": 10,
        "timeout": 5.0,
    }


async def _call_search_web_with_timeout(timeout: object) -> object:
    return await search_web("harnyx subnet", provider="parallel", timeout=timeout)


async def _call_search_ai_with_timeout(timeout: object) -> object:
    return await search_ai("harnyx subnet", provider="parallel", timeout=timeout)


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
    "invoke_helper",
    [
        _call_search_web_with_timeout,
        _call_search_ai_with_timeout,
        _call_fetch_page_with_timeout,
        _call_llm_chat_with_timeout,
        _call_tooling_info_with_timeout,
        _call_test_tool_with_timeout,
    ],
)
@pytest.mark.parametrize("timeout", [0, -1.0, float("nan"), float("inf"), float("-inf"), "5", True])
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


async def test_llm_chat_helper_invokes_tool_proxy_with_timeout() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="chat-2",
                response={
                    "id": "resp-2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "hi"}],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
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
                timeout=5,
            )
    finally:
        await proxy.aclose()

    assert result.receipt_id == "chat-2"
    payload = captured["payload"]
    assert payload["tool"] == "llm_chat"
    assert payload["kwargs"]["provider"] == "chutes"
    assert payload["kwargs"]["model"] == "demo-model"
    assert payload["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["kwargs"]["timeout"] == 5.0


async def test_llm_chat_helper_forwards_openrouter_provider_extra() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="chat-openrouter-extra",
                response={
                    "id": "resp-openrouter-extra",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "hi"}],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
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
                provider="openrouter",
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-oss-120b",
                provider_extra={"provider": {"only": ["cerebras"]}},
            )
    finally:
        await proxy.aclose()

    assert result.receipt_id == "chat-openrouter-extra"
    payload = captured["payload"]
    assert payload["tool"] == "llm_chat"
    assert payload["kwargs"]["provider"] == "openrouter"
    assert payload["kwargs"]["provider_extra"] == {"provider": {"only": ["cerebras"]}}


async def test_llm_chat_helper_forwards_ai_gateway_provider_extra() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="chat-ai-gateway-extra",
                response={
                    "id": "resp-ai-gateway-extra",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "hi"}],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
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
                provider="ai_gateway",
                messages=[{"role": "user", "content": "hi"}],
                model="zai/glm-5.2-fast",
                provider_extra={"providerOptions": {"gateway": {"only": ["cerebras"]}}},
            )
    finally:
        await proxy.aclose()

    assert result.receipt_id == "chat-ai-gateway-extra"
    payload = captured["payload"]
    assert payload["tool"] == "llm_chat"
    assert payload["kwargs"]["provider"] == "ai_gateway"
    assert payload["kwargs"]["provider_extra"] == {"providerOptions": {"gateway": {"only": ["cerebras"]}}}


async def test_llm_chat_helper_rejects_chutes_provider_extra_before_proxy_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("llm_chat should reject chutes provider_extra before invoking the tool proxy")

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
                    provider_extra={"provider": {"only": ["cerebras"]}},
                )
    finally:
        await proxy.aclose()


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


async def test_llm_chat_helper_serializes_complete_tool_loop_request() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_response_payload(
                receipt_id="chat-tool-loop",
                response={
                    "id": "resp-tool-loop",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "Paris is 19C."}],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                },
            ),
        )

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    messages = [
        {"role": "user", "content": "Weather in Paris?"},
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
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"temperature":19}'},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "strict": True,
            },
        }
    ]
    try:
        with bind_tool_invoker(proxy):
            await llm_chat(
                provider="openrouter",
                model="openai/gpt-oss-20b",
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
                parallel_tool_calls=True,
                max_tokens=512,
            )
    finally:
        await proxy.aclose()

    kwargs = captured["payload"]["kwargs"]
    assert kwargs["messages"] == messages
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "lookup_weather"}}
    assert kwargs["parallel_tool_calls"] is True
    assert kwargs["max_output_tokens"] == 512
    assert "max_tokens" not in kwargs


async def test_llm_chat_helper_rejects_typed_message_parts_it_cannot_serialize() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unsupported typed message parts must fail before proxy I/O")

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            with pytest.raises(ValueError, match="only input_text"):
                await llm_chat(
                    provider="chutes",
                    model="demo-model",
                    messages=[
                        LlmMessage(
                            role="user",
                            content=(
                                LlmInputTextPart(text="Describe this image."),
                                LlmInputImagePart(data=LlmInputImageData(url="https://example.com/image.png")),
                            ),
                        )
                    ],
                )
    finally:
        await proxy.aclose()


@pytest.mark.parametrize(
    ("field", "value"),
    (("include", ["sources"]), ("response_format", "json")),
)
async def test_llm_chat_helper_rejects_removed_fields_before_proxy_io(field: str, value: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("removed llm_chat fields must fail before proxy I/O")

    proxy = ToolProxy(
        base_url="http://validator",
        token=TEST_TOKEN,
        session_id=SESSION_ID,
        client=httpx.AsyncClient(base_url="http://validator", transport=httpx.MockTransport(handler)),
    )
    try:
        with bind_tool_invoker(proxy):
            with pytest.raises(ValidationError, match=field):
                await llm_chat(
                    provider="chutes",
                    model="demo-model",
                    messages=[{"role": "user", "content": "hi"}],
                    **{field: value},
                )
    finally:
        await proxy.aclose()
