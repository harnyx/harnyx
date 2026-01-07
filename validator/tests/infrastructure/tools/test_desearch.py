from __future__ import annotations

from typing import Any

import httpx
import pytest

from caster_commons.tools.desearch import DeSearchClient
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
