from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from caster_commons.tools.desearch import DeSearchAiDateFilter, DeSearchClient
from caster_commons.tools.search_models import SearchWebSearchRequest, SearchXSearchRequest

pytestmark = pytest.mark.anyio("asyncio")


def _capture_request() -> tuple[dict[str, Any], httpx.MockTransport]:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["params"] = request.url.params
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    return captured, transport


async def test_desearch_client_posts_payload() -> None:
    captured, transport = _capture_request()
    client = httpx.AsyncClient(base_url="https://api.desearch.ai", transport=transport)

    adapter = DeSearchClient(
        base_url="https://api.desearch.ai",
        api_key="test-key",
        client=client,
    )

    request = SearchWebSearchRequest(query="caster subnet", num=5)
    result = await adapter.search_links_web(request)

    assert result.data == []
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/web?query=caster+subnet&num=5"
    assert captured["headers"]["authorization"] == "test-key"


async def test_desearch_client_raises_on_error_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "failure"})

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="test-key", client=client)

    with pytest.raises(RuntimeError):
        await adapter.search_links_web(SearchWebSearchRequest(query="caster subnet"))


async def test_desearch_client_twitter_search() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["params"] = request.url.params
        return httpx.Response(200, json=[{"text": "hello", "user": {"username": "foo"}}])

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    response = await adapter.search_links_twitter(SearchXSearchRequest(query="#caster", count=3))

    assert response.data[0].text == "hello"
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/twitter?query=%23caster&count=3"


async def test_desearch_client_ai_search_twitter_posts_posts_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "key"

        if request.url.path == "/desearch/ai/search":
            assert request.method == "POST"
            payload = json.loads(request.content)
            assert payload["prompt"] == "caster subnet"
            assert payload["tools"] == ["twitter"]
            assert payload["result_type"] == "LINKS_WITH_FINAL_SUMMARY"
            assert payload["system_message"] == ""
            assert payload["streaming"] is False
            assert payload["count"] == 200
            assert payload["date_filter"] == "PAST_24_HOURS"
            assert "start_date" not in payload
            assert "end_date" not in payload
            assert "model" not in payload

            return httpx.Response(
                200,
                json={
                    "tweets": [
                        {
                            "id": "123",
                            "url": "https://x.com/foo/status/123",
                            "text": "hi",
                            "user": {"username": "foo"},
                        }
                    ],
                    "completion": "hello",
                },
            )

        if request.url.path == "/twitter/post":
            raise AssertionError("ai_search_twitter_posts should not call /twitter/post when tweets are present")

        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    tweets = await adapter.ai_search_twitter_posts(
        prompt="caster subnet",
        count=300,
        date_filter=DeSearchAiDateFilter.PAST_24_HOURS,
    )

    assert len(tweets) == 1
    assert tweets[0].id == "123"
