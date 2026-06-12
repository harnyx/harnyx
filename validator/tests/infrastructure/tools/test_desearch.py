from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.tools.desearch import DeSearchAiDateFilter, DeSearchClient
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    SearchAiSearchRequest,
    SearchWebSearchRequest,
    SearchXSearchRequest,
)

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

    request = SearchWebSearchRequest(provider="desearch", search_queries=("harnyx", "subnet"), num=5)
    result = await adapter.search_links_web(request)

    assert result.data == []
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/web?query=%28harnyx%29+OR+%28subnet%29&num=5"
    assert captured["headers"]["authorization"] == "test-key"


async def test_desearch_client_captures_json_object_cost_usd_body_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [{"link": "https://example.com", "title": "Example", "snippet": "Summary"}],
                "cost_usd": 0.00015,
                "usage_count": 1,
                "service": "web",
                "currency": "USD",
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="test-key", client=client)

    result = await adapter.search_web(
        SearchWebSearchRequest(provider="desearch", search_queries=("harnyx",), num=5)
    )

    assert result.billing is not None
    assert result.billing.actual_cost_usd == pytest.approx(0.00015)
    assert result.billing.usage_count == 1
    assert result.billing.service == "web"
    assert result.billing.currency == "USD"
    assert result.billing.source == "response_body"
    assert "cost_usd" not in result.response.model_dump(exclude_none=True)


async def test_desearch_client_captures_array_cost_from_headers() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"link": "https://example.com", "title": "Example", "snippet": "Summary"}],
            headers={
                "X-Desearch-Cost-Usd": "0.00017",
                "X-Desearch-Usage-Count": "2",
                "X-Desearch-Service": "web",
                "X-Desearch-Currency": "USD",
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="test-key", client=client)

    result = await adapter.search_web(
        SearchWebSearchRequest(provider="desearch", search_queries=("harnyx",), num=5)
    )

    assert result.billing is not None
    assert result.billing.actual_cost_usd == pytest.approx(0.00017)
    assert result.billing.usage_count == 2
    assert result.billing.source == "response_headers"


async def test_desearch_client_preserves_single_search_term() -> None:
    captured, transport = _capture_request()
    client = httpx.AsyncClient(base_url="https://api.desearch.ai", transport=transport)

    adapter = DeSearchClient(
        base_url="https://api.desearch.ai",
        api_key="test-key",
        client=client,
    )

    request = SearchWebSearchRequest(provider="desearch", search_queries=("United States",), num=5)
    result = await adapter.search_links_web(request)

    assert result.data == []
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/web?query=United+States&num=5"
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
        await adapter.search_links_web(
            SearchWebSearchRequest(provider="desearch", search_queries=("harnyx", "subnet"))
        )


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

    response = await adapter.search_links_twitter(SearchXSearchRequest(query="#harnyx", count=3))

    assert response.data[0].text == "hello"
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/twitter?query=%23harnyx&count=3"


async def test_desearch_client_ai_search_twitter_posts_posts_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "key"

        if request.url.path == "/desearch/ai/search":
            assert request.method == "POST"
            payload = json.loads(request.content)
            assert payload["prompt"] == "harnyx subnet"
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

    response = await adapter.ai_search_twitter_posts(
        prompt="harnyx subnet",
        count=300,
        date_filter=DeSearchAiDateFilter.PAST_24_HOURS,
    )

    assert response.tweets and len(response.tweets) == 1
    assert response.tweets[0].id == "123"
    assert response.completion == "hello"


async def test_desearch_client_search_ai_clamps_count_and_preserves_retry_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/desearch/ai/search":
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        payload = json.loads(request.content)
        assert payload["prompt"] == "harnyx subnet"
        assert payload["tools"] == ["web", "hackernews", "reddit", "wikipedia", "youtube", "arxiv"]
        assert payload["result_type"] == "LINKS_WITH_FINAL_SUMMARY"
        assert payload["system_message"] == ""
        assert payload["count"] == 10
        return httpx.Response(
            200,
            json={
                "search": [
                    {
                        "link": "https://example.com",
                        "title": "Example",
                        "snippet": "Snippet",
                    }
                ],
                "cost_usd": 0.00031,
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    result = await adapter.search_ai(
        SearchAiSearchRequest(provider="desearch", prompt="harnyx subnet", count=10)
    )
    response = result.response

    assert [item.model_dump(exclude_none=True) for item in response.data] == [
        {
            "url": "https://example.com",
            "title": "Example",
            "note": "Snippet",
        }
    ]
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert result.billing.actual_cost_usd == pytest.approx(0.00031)


async def test_desearch_client_search_ai_accepts_summary_and_results_shape() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/desearch/ai/search":
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return httpx.Response(
            200,
            json={
                "summary": "Hamlet is a tragedy.",
                "results": [
                    {
                        "url": "https://example.com/hamlet",
                        "title": "Hamlet",
                        "snippet": "Plot summary",
                    }
                ],
                "cost_usd": 0.00032,
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    result = await adapter.search_ai(SearchAiSearchRequest(provider="desearch", prompt="hamlet", count=10))
    response = result.response

    assert [item.model_dump(exclude_none=True) for item in response.data] == [
        {
            "url": "https://example.com/hamlet",
            "title": "Hamlet",
            "note": "Plot summary",
        }
    ]


async def test_desearch_client_search_ai_accepts_sdk_search_results_shape() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/desearch/ai/search":
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return httpx.Response(
            200,
            json={
                "youtube_search_results": {
                    "organic_results": [
                        {
                            "link": "https://example.com/video",
                            "title": "Hamlet video",
                            "summary_description": "Video summary",
                        }
                    ]
                },
                "cost_usd": 0.00033,
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    result = await adapter.search_ai(SearchAiSearchRequest(provider="desearch", prompt="hamlet", count=10))
    response = result.response

    assert [item.model_dump(exclude_none=True) for item in response.data] == [
        {
            "url": "https://example.com/video",
            "title": "Hamlet video",
            "note": "Video summary",
        }
    ]


async def test_desearch_client_search_ai_summary_only_shape_returns_empty_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/desearch/ai/search":
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return httpx.Response(
            200,
            json={"summary": "Hamlet is a tragedy by Shakespeare.", "cost_usd": 0.00034},
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    result = await adapter.search_ai(SearchAiSearchRequest(provider="desearch", prompt="hamlet", count=10))
    response = result.response

    assert response.data == []
    assert response.attempts == 1
    assert response.retry_reasons == ()


async def test_desearch_client_fetch_page_text() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        return httpx.Response(200, text="example page content", headers={"X-Desearch-Cost-Usd": "0.00021"})

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    result = await adapter.fetch_page(FetchPageRequest(provider="desearch", url="https://example.com"))
    response = result.response

    assert response.data[0].url == "https://example.com"
    assert response.data[0].content == "example page content"
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/web/crawl?url=https%3A%2F%2Fexample.com&format=text"


async def test_desearch_client_fetch_page_captures_text_cost_from_headers() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="example page content",
            headers={
                "X-Desearch-Cost-Usd": "0.00021",
                "X-Desearch-Usage-Count": "1",
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    result = await adapter.fetch_page(FetchPageRequest(provider="desearch", url="https://example.com"))

    assert result.response.data[0].content == "example page content"
    assert result.billing is not None
    assert result.billing.actual_cost_usd == pytest.approx(0.00021)
    assert result.billing.usage_count == 1
    assert result.billing.source == "response_headers"


async def test_desearch_client_malformed_billing_metadata_returns_response_for_static_pricing() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"link": "https://example.com"}], "cost_usd": "not-a-number"},
            headers={"X-Desearch-Usage-Count": "also-not-a-number"},
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    result = await adapter.search_web(
        SearchWebSearchRequest(provider="desearch", search_queries=("harnyx",), num=5)
    )

    assert result.response.data
    assert result.billing.actual_cost_provider == "desearch"
    assert result.billing.actual_cost_usd is None
    assert result.billing.billable_units == 1
    assert result.billing.source == "missing_provider_metadata"


async def test_desearch_client_rejects_nonfinite_billing_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"link": "https://example.com"}]},
            headers={"X-Desearch-Cost-Usd": "nan"},
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    with pytest.raises(ToolProviderError) as exc_info:
        await adapter.search_web(
            SearchWebSearchRequest(provider="desearch", search_queries=("harnyx",), num=5)
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "cost metadata must be finite" in str(exc_info.value.__cause__)


@pytest.mark.parametrize(
    ("json_body", "headers"),
    [
        ({"data": [{"link": "https://example.com"}], "cost_usd": -1}, {}),
        ({"data": [{"link": "https://example.com"}]}, {"X-Desearch-Cost-Usd": "-1"}),
    ],
)
async def test_desearch_client_rejects_negative_billing_metadata(
    json_body: dict[str, object],
    headers: dict[str, str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=json_body, headers=headers)

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    with pytest.raises(ToolProviderError) as exc_info:
        await adapter.search_web(
            SearchWebSearchRequest(provider="desearch", search_queries=("harnyx",), num=5)
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "cost metadata must be non-negative" in str(exc_info.value.__cause__)


async def test_desearch_client_partial_billing_metadata_returns_response_for_static_pricing() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"link": "https://example.com"}], "usage_count": 1},
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    result = await adapter.search_web(
        SearchWebSearchRequest(provider="desearch", search_queries=("harnyx",), num=5)
    )

    assert result.response.data
    assert result.billing.actual_cost_provider == "desearch"
    assert result.billing.actual_cost_usd is None
    assert result.billing.billable_units == 1
    assert result.billing.source == "response_body"


async def test_desearch_client_fetch_page_raises_on_error_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "failure"})

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    with pytest.raises(RuntimeError):
        await adapter.fetch_page(FetchPageRequest(provider="desearch", url="https://example.com"))


async def test_desearch_client_fetch_page_rejects_blank_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="   \n  ")

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    with pytest.raises(ValueError):
        await adapter.fetch_page(FetchPageRequest(provider="desearch", url="https://example.com"))
