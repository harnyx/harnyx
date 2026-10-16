from __future__ import annotations

import json

import httpx
import pytest

from harnyx_commons.tools.search_models import FetchPageRequest, SearchWebSearchRequest
from harnyx_commons.tools.tavily import TavilyClient

pytestmark = pytest.mark.anyio("asyncio")


async def test_tavily_search_pins_generated_outputs_off_and_retains_usage() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "request_id": "tavily-1",
                "usage": {"credits": 2},
                "results": [{"url": "https://example.com", "title": "Example", "content": "snippet"}],
            },
        )

    http_client = httpx.AsyncClient(base_url="https://api.tavily.com", transport=httpx.MockTransport(handler))
    client = TavilyClient(base_url="https://api.tavily.com", api_key="tvly-key", client=http_client)
    result = await client.search_web(
        SearchWebSearchRequest.model_validate(
            {
                "provider": "tavily",
                "search_queries": ["one", "two"],
                "num": 5,
                "provider_extra": {
                    "search_depth": "basic",
                    "chunks_per_source": 2,
                    "time_range": "d",
                },
            }
        )
    )

    payload = json.loads(requests[0].content)
    assert payload == {
        "query": "(one) OR (two)",
        "search_depth": "basic",
        "chunks_per_source": 2,
        "time_range": "d",
        "topic": "general",
        "auto_parameters": False,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "include_usage": True,
        "max_results": 5,
    }
    assert result.response.data[0].snippet == "snippet"
    assert result.billing.usage_count == 2
    await http_client.aclose()


async def test_tavily_search_defaults_to_fast_retrieved_chunks() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    http_client = httpx.AsyncClient(base_url="https://api.tavily.com", transport=httpx.MockTransport(handler))
    client = TavilyClient(base_url="https://api.tavily.com", api_key="tvly-key", client=http_client)
    await client.search_web(SearchWebSearchRequest(provider="tavily", search_queries=("harnyx",)))

    assert isinstance(payloads[0], dict)
    assert payloads[0]["search_depth"] == "fast"
    await http_client.aclose()


async def test_tavily_fetch_maps_extract_controls() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [{"url": "https://example.com", "raw_content": "page"}],
                "request_id": "tavily-extract-1",
                "usage": {"credits": 1},
            },
        )

    http_client = httpx.AsyncClient(base_url="https://api.tavily.com", transport=httpx.MockTransport(handler))
    client = TavilyClient(base_url="https://api.tavily.com", api_key="tvly-key", client=http_client)
    result = await client.fetch_page(
        FetchPageRequest.model_validate(
            {
                "provider": "tavily",
                "url": "https://example.com",
                "provider_extra": {"query": "important", "chunks_per_source": 2, "format": "text"},
            }
        )
    )

    assert payloads == [
        {
            "urls": ["https://example.com"],
            "query": "important",
            "chunks_per_source": 2,
            "extract_depth": "basic",
            "format": "text",
            "include_images": False,
            "include_favicon": False,
            "include_usage": True,
        }
    ]
    assert result.response.data[0].content == "page"
    await http_client.aclose()


@pytest.mark.parametrize("status", [429, 502, 504, 529, 403])
async def test_shared_transport_keeps_existing_retry_and_auth_behavior_without_classifier(monkeypatch, status):
    from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
    from harnyx_commons.llm.retry_utils import RetryPolicy

    attempts, delays = [], []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("harnyx_commons.tools.search_http.asyncio.sleep", sleep)

    def handle(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(status, headers={"Retry-After": "2"}, json={"error": "provider failure"})
        return httpx.Response(200, json={"results": [{"url": "https://example.com", "raw_content": "Evidence"}]})

    async with httpx.AsyncClient(base_url="https://api.tavily.com", transport=httpx.MockTransport(handle)) as http:
        client = TavilyClient(
            base_url="https://api.tavily.com", api_key="key", client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        request = FetchPageRequest(provider="tavily", url="https://example.com")
        if status in {429, 502, 504}:
            assert (await client.fetch_page(request)).response.data[0].content == "Evidence"
            assert len(attempts) == 2
            assert delays == [2.0]
        else:
            with pytest.raises(ToolProviderError) as error:
                await client.fetch_page(request)
            expected = (ToolProviderFailureCode.AUTHENTICATION_FAILED if status == 403
                        else ToolProviderFailureCode.PROVIDER_FAILED)
            assert error.value.failure_code == expected
            assert len(attempts) == 1
            assert delays == []
