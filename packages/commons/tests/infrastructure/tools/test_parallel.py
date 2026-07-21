from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.pricing import price_parallel_extract, price_parallel_search
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchAiSearchRequest, SearchWebSearchRequest

pytestmark = pytest.mark.anyio("asyncio")


async def test_parallel_client_can_suppress_request_and_raw_response_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="harnyx_commons.tools.parallel.calls")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "search_id": "raw-provider-envelope",
                "results": [{"url": "https://example.com/private"}],
            },
        )

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
        include_payloads_in_logs=False,
    )

    await adapter.search_web(
        SearchWebSearchRequest(provider="parallel", search_queries=("private-query",))
    )

    record = next(record for record in caplog.records if record.msg == "parallel.request.complete")
    assert not hasattr(record, "json_fields")
    assert "private-query" not in str(record.__dict__)
    assert "raw-provider-envelope" not in str(record.__dict__)


async def test_parallel_client_search_web_posts_keyword_list() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "search_id": "search-1",
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "Alpha",
                        "excerpts": ["alpha snippet"],
                        "publish_date": "2026-03-24T00:00:00Z",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    result = await adapter.search_web(
        SearchWebSearchRequest(provider="parallel", search_queries=("alpha", "beta"), num=3)
    )
    response = result.response

    assert response.data[0].link == "https://example.com/a"
    assert response.data[0].snippet == "alpha snippet"
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert result.billing.actual_cost_usd == pytest.approx(price_parallel_search(billable_results=1))
    assert result.billing.actual_cost_provider == "parallel"
    assert result.billing.billable_units == 1
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.parallel.ai/v1beta/search"
    assert captured["headers"]["x-api-key"] == "parallel-key"
    assert captured["json"] == {
        "search_queries": ["alpha", "beta"],
        "max_results": 3,
    }


async def test_parallel_client_search_web_applies_request_timeout_to_provider_call() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, json={"search_id": "search-1", "results": []})

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        timeout=60.0,
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    await adapter.search_web(
        SearchWebSearchRequest(
            provider="parallel",
            search_queries=("timeout parity",),
            timeout=180.0,
        )
    )

    assert captured["timeout"] == {
        "connect": 190.0,
        "read": 190.0,
        "write": 190.0,
        "pool": 190.0,
    }


async def test_parallel_client_search_ai_uses_objective() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "search_id": "search-2",
                "results": [
                    {
                        "url": "https://example.com/b",
                        "title": "Beta",
                        "excerpts": ["beta summary"],
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    result = await adapter.search_ai(SearchAiSearchRequest(provider="parallel", prompt="find beta", count=10))
    response = result.response

    assert response.data[0].url == "https://example.com/b"
    assert response.data[0].note == "beta summary"
    assert result.billing.actual_cost_usd == pytest.approx(price_parallel_search(billable_results=1))
    assert result.billing.actual_cost_provider == "parallel"
    assert captured["json"] == {
        "objective": "find beta",
        "max_results": 10,
    }


async def test_parallel_client_fetch_page_uses_extract() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-1",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                        "full_content": "full page text",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    result = await adapter.fetch_page(FetchPageRequest(provider="parallel", url="https://example.com"))
    response = result.response

    assert response.data[0].url == "https://example.com"
    assert response.data[0].content == "full page text"
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert result.billing.actual_cost_usd == pytest.approx(price_parallel_extract(url_count=1))
    assert result.billing.actual_cost_provider == "parallel"
    assert captured["json"] == {
        "urls": ["https://example.com"],
        "full_content": True,
        "excerpts": False,
    }


async def test_parallel_client_raises_on_error_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "failure"})

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    with pytest.raises(ToolProviderError):
        await adapter.fetch_page(FetchPageRequest(provider="parallel", url="https://example.com"))


async def test_parallel_401_is_typed_as_authentication_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "raw-provider-envelope"})

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await adapter.fetch_page(FetchPageRequest(provider="parallel", url="https://example.com"))

    assert exc_info.value.failure_code is ToolProviderFailureCode.AUTHENTICATION_FAILED
    assert exc_info.value.http_status == 401


async def test_parallel_client_fetch_page_raises_on_empty_extract_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"extract_id": "extract-1", "results": []})

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    with pytest.raises(ToolProviderError):
        await adapter.fetch_page(FetchPageRequest(provider="parallel", url="https://example.com"))
