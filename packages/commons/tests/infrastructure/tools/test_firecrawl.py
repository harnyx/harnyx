from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.tools import firecrawl as firecrawl_module
from harnyx_commons.tools.firecrawl import FirecrawlClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchWebSearchRequest

pytestmark = pytest.mark.anyio


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    attempts: int = 1,
) -> tuple[FirecrawlClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(
        base_url="https://api.firecrawl.dev",
        transport=httpx.MockTransport(handler),
    )
    return (
        FirecrawlClient(
            base_url="https://api.firecrawl.dev",
            api_key="firecrawl-key",
            client=http_client,
            retry_policy=RetryPolicy(attempts=attempts, initial_ms=0, max_ms=0, jitter=0.0),
            include_payloads_in_logs=False,
        ),
        http_client,
    )


async def test_search_web_normalizes_results_and_optional_billing_metadata() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "id": "search-123",
                "creditsUsed": 2,
                "data": {
                    "web": [
                        {
                            "url": "https://example.com/article",
                            "title": "Example",
                            "description": "Result summary",
                        }
                    ]
                },
            },
        )

    client, http_client = _client(handler)
    try:
        result = await client.search_web(
            SearchWebSearchRequest(provider="firecrawl", search_queries=("one", "two"))
        )
    finally:
        await http_client.aclose()

    assert len(observed) == 1
    request = observed[0]
    assert request.url.path == "/v2/search"
    assert request.headers["authorization"] == "Bearer firecrawl-key"
    assert json.loads(request.content) == {"query": "(one) OR (two)", "sources": ["web"]}
    assert result.response.data[0].model_dump() == {
        "link": "https://example.com/article",
        "snippet": "Result summary",
        "title": "Example",
    }
    assert result.billing.provider_request_id == "search-123"
    assert result.billing.usage_count == 2
    assert result.billing.actual_cost_usd is None


async def test_search_web_zero_limit_returns_without_provider_request() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("Firecrawl must not be called for num=0")

    client, http_client = _client(handler)
    try:
        result = await client.search_web(
            SearchWebSearchRequest(provider="firecrawl", search_queries=("harnyx",), num=0)
        )
    finally:
        await http_client.aclose()

    assert result.response.data == []
    assert result.billing.source == "missing_provider_metadata"


async def test_search_web_clamps_limit_and_accepts_empty_results() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "data": {"web": []}})

    client, http_client = _client(handler)
    try:
        result = await client.search_web(
            SearchWebSearchRequest(provider="firecrawl", search_queries=("harnyx",), num=101)
        )
    finally:
        await http_client.aclose()

    assert payloads == [{"query": "(harnyx)", "sources": ["web"], "limit": 100}]
    assert result.response.data == []
    assert result.billing.source == "missing_provider_metadata"


@pytest.mark.parametrize(
    "response_json",
    [
        {"success": True, "data": {}},
        {"success": True, "data": {"web": {}}},
        {"success": True, "data": {"web": [{"url": "   "}]}},
    ],
)
async def test_search_web_rejects_malformed_result_contract(response_json: object) -> None:
    client, http_client = _client(lambda _request: httpx.Response(200, json=response_json))
    try:
        with pytest.raises(ToolProviderError, match="response invalid"):
            await client.search_web(
                SearchWebSearchRequest(provider="firecrawl", search_queries=("harnyx",))
            )
    finally:
        await http_client.aclose()


async def test_search_web_rejects_query_over_500_characters_without_request() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("overlong Firecrawl query must fail before the request")

    client, http_client = _client(handler)
    try:
        with pytest.raises(ValueError, match="500 characters"):
            await client.search_web(
                SearchWebSearchRequest(provider="firecrawl", search_queries=("x" * 499,))
            )
    finally:
        await http_client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"{"),
        httpx.Response(200, json=[]),
    ],
)
async def test_search_web_maps_malformed_json_to_provider_failure(response: httpx.Response) -> None:
    client, http_client = _client(lambda _request: response)
    try:
        with pytest.raises(ToolProviderError, match="response invalid"):
            await client.search_web(
                SearchWebSearchRequest(provider="firecrawl", search_queries=("harnyx",))
            )
    finally:
        await http_client.aclose()


async def test_fetch_page_normalizes_documented_scrape_shape_without_billing_metadata() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "# Example\n\nContent",
                    "metadata": {"title": "Example", "sourceURL": "https://example.com/final"},
                },
            },
        )

    client, http_client = _client(handler)
    try:
        result = await client.fetch_page(
            FetchPageRequest(provider="firecrawl", url="https://example.com")
        )
    finally:
        await http_client.aclose()

    assert payloads == [
        {"url": "https://example.com", "formats": ["markdown"], "onlyMainContent": True}
    ]
    assert result.response.data[0].model_dump() == {
        "url": "https://example.com/final",
        "content": "# Example\n\nContent",
        "title": "Example",
    }
    assert result.billing.source == "missing_provider_metadata"


async def test_fetch_page_forwards_raw_html_format_and_normalizes_raw_html() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "rawHtml": "<main>Raw content</main>",
                    "metadata": {"title": "Example"},
                },
            },
        )

    client, http_client = _client(handler)
    try:
        result = await client.fetch_page(
            FetchPageRequest.model_validate(
                {
                    "provider": "firecrawl",
                    "url": "https://example.com",
                    "provider_extra": {"formats": ["rawHtml"]},
                }
            )
        )
    finally:
        await http_client.aclose()

    assert payloads == [
        {"url": "https://example.com", "formats": ["rawHtml"], "onlyMainContent": True}
    ]
    assert [item.content for item in result.response.data] == ["<main>Raw content</main>"]


async def test_fetch_page_returns_every_requested_format_in_request_order() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "Markdown content",
                    "rawHtml": "<main>Raw content</main>",
                    "metadata": {
                        "title": "Example",
                        "sourceURL": "https://example.com/final",
                    },
                },
            },
        )

    client, http_client = _client(handler)
    try:
        result = await client.fetch_page(
            FetchPageRequest.model_validate(
                {
                    "provider": "firecrawl",
                    "url": "https://example.com",
                    "provider_extra": {"formats": ["rawHtml", "markdown"]},
                }
            )
        )
    finally:
        await http_client.aclose()

    assert payloads == [
        {
            "url": "https://example.com",
            "formats": ["rawHtml", "markdown"],
            "onlyMainContent": True,
        }
    ]
    assert [item.model_dump() for item in result.response.data] == [
        {
            "url": "https://example.com/final",
            "content": "<main>Raw content</main>",
            "title": "Example",
        },
        {
            "url": "https://example.com/final",
            "content": "Markdown content",
            "title": "Example",
        },
    ]


@pytest.mark.parametrize("raw_html", [None, "", "   "])
async def test_fetch_page_rejects_missing_or_blank_requested_raw_html(raw_html: object) -> None:
    client, http_client = _client(
        lambda _request: httpx.Response(
            200,
            json={
                "success": True,
                "id": "scrape-raw-html",
                "creditsUsed": 1,
                "data": {
                    "markdown": "Markdown fallback must not be used",
                    "rawHtml": raw_html,
                    "metadata": {},
                },
            },
        )
    )
    try:
        with pytest.raises(ToolProviderError, match="response invalid") as exc_info:
            await client.fetch_page(
                FetchPageRequest.model_validate(
                    {
                        "provider": "firecrawl",
                        "url": "https://example.com",
                        "provider_extra": {"formats": ["rawHtml"]},
                    }
                )
            )
    finally:
        await http_client.aclose()

    assert exc_info.value.billing is not None
    assert exc_info.value.billing.provider_request_id == "scrape-raw-html"
    assert exc_info.value.billing.usage_count == 1


async def test_fetch_page_maps_markdown_compatible_pdf_and_proxy_controls() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {"markdown": "parsed content", "metadata": {}},
            },
        )

    client, http_client = _client(handler)
    try:
        await client.fetch_page(
            FetchPageRequest.model_validate(
                {
                    "provider": "firecrawl",
                    "url": "https://example.com/document.pdf",
                    "provider_extra": {"parse_pdf": True, "proxy": "enhanced"},
                }
            )
        )
    finally:
        await http_client.aclose()

    assert payloads == [
        {
            "url": "https://example.com/document.pdf",
            "formats": ["markdown"],
            "onlyMainContent": True,
            "parsers": ["pdf"],
            "proxy": "enhanced",
        }
    ]


@pytest.mark.parametrize("markdown", [None, "", "   "])
async def test_fetch_page_rejects_missing_or_blank_markdown(markdown: object) -> None:
    client, http_client = _client(
        lambda _request: httpx.Response(
            200,
            json={"success": True, "data": {"markdown": markdown, "metadata": {}}},
        )
    )
    try:
        with pytest.raises(ToolProviderError, match="response invalid"):
            await client.fetch_page(FetchPageRequest(provider="firecrawl", url="https://example.com"))
    finally:
        await http_client.aclose()


@pytest.mark.parametrize("metadata", [{"sourceURL": {}}, {"url": []}, {"sourceURL": 42}])
async def test_fetch_page_rejects_non_string_metadata_urls(metadata: object) -> None:
    client, http_client = _client(
        lambda _request: httpx.Response(
            200,
            json={"success": True, "data": {"markdown": "content", "metadata": metadata}},
        )
    )
    try:
        with pytest.raises(ToolProviderError, match="response invalid"):
            await client.fetch_page(FetchPageRequest(provider="firecrawl", url="https://example.com"))
    finally:
        await http_client.aclose()


async def test_rate_limit_respects_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json={"success": True, "data": {"web": []}}),
        ]
    )
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(firecrawl_module.asyncio, "sleep", record_sleep)
    client, http_client = _client(lambda _request: next(responses), attempts=2)
    try:
        await client.search_web(SearchWebSearchRequest(provider="firecrawl", search_queries=("harnyx",)))
    finally:
        await http_client.aclose()

    assert sleeps == [2.0]


@pytest.mark.parametrize("status", [401, 402, 403])
async def test_non_retryable_statuses_fail_once(status: int) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    client, http_client = _client(handler, attempts=3)
    try:
        with pytest.raises(ToolProviderError) as exc_info:
            await client.search_web(SearchWebSearchRequest(provider="firecrawl", search_queries=("harnyx",)))
    finally:
        await http_client.aclose()

    assert calls == 1
    assert exc_info.value.failure_code is (
        ToolProviderFailureCode.AUTHENTICATION_FAILED
        if status == 401
        else ToolProviderFailureCode.PROVIDER_FAILED
    )


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
async def test_documented_transient_statuses_retry(status: int) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status)
        return httpx.Response(200, json={"success": True, "data": {"web": []}})

    client, http_client = _client(handler, attempts=2)
    try:
        result = await client.search_web(
            SearchWebSearchRequest(provider="firecrawl", search_queries=("harnyx",))
        )
    finally:
        await http_client.aclose()

    assert calls == 2
    assert result.response.data == []
