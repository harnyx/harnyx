from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.pricing import price_parallel_extract, price_parallel_search
from harnyx_commons.tools.extraction_models import ExtractPagesRequest
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

    await adapter.search_web(SearchWebSearchRequest(provider="parallel", search_queries=("private-query",)))

    record = next(record for record in caplog.records if record.msg == "parallel.request.complete")
    assert not hasattr(record, "json_fields")
    assert "private-query" not in str(record.__dict__)
    assert "raw-provider-envelope" not in str(record.__dict__)


async def test_parallel_client_search_web_posts_keyword_list_and_turbo_mode() -> None:
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
        SearchWebSearchRequest.model_validate(
            {
                "provider": "parallel",
                "search_queries": ["alpha", "beta"],
                "num": 3,
                "provider_extra": {"mode": "turbo"},
            }
        )
    )
    response = result.response

    assert response.data[0].link == "https://example.com/a"
    assert response.data[0].snippet == "alpha snippet"
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert result.billing.actual_cost_usd == pytest.approx(
        price_parallel_search(billable_results=1, mode="turbo")
    )
    assert result.billing.actual_cost_provider == "parallel"
    assert result.billing.billable_units == 1
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.parallel.ai/v1/search"
    assert captured["headers"]["x-api-key"] == "parallel-key"
    assert captured["json"] == {
        "search_queries": ["alpha", "beta"],
        "mode": "turbo",
        "advanced_settings": {"max_results": 3},
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


async def test_parallel_client_fetch_page_uses_extract_with_top_level_total_character_limit() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-1",
                "session_id": "session-1",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                        "full_content": "full page text",
                    }
                ],
                "errors": [],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    result = await adapter.fetch_page(
        FetchPageRequest.model_validate(
            {
                "provider": "parallel",
                "url": "https://example.com",
                "provider_extra": {"max_chars_total": 12_000},
            }
        )
    )
    response = result.response

    assert response.data[0].url == "https://example.com"
    assert response.data[0].content == "full page text"
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert result.billing.actual_cost_usd == pytest.approx(price_parallel_extract(url_count=1))
    assert result.billing.actual_cost_provider == "parallel"
    assert result.billing.source == "request_body"
    assert captured["json"] == {
        "urls": ["https://example.com/"],
        "advanced_settings": {
            "full_content": True,
        },
        "max_chars_total": 12_000,
    }


async def test_parallel_client_fetch_page_returns_excerpts_when_full_content_is_disabled() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-excerpts",
                "session_id": "session-excerpts",
                "results": [
                    {
                        "url": "https://example.com/",
                        "title": "Example",
                        "excerpts": ["first relevant excerpt", "second relevant excerpt"],
                    }
                ],
                "errors": [],
            },
        )

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = await adapter.fetch_page(
        FetchPageRequest.model_validate(
            {
                "provider": "parallel",
                "url": "https://example.com",
                "provider_extra": {"full_content": False},
            }
        )
    )

    assert result.response.data[0].content == "first relevant excerpt\n\nsecond relevant excerpt"
    assert captured["json"] == {
        "urls": ["https://example.com/"],
        "advanced_settings": {"full_content": False},
    }


async def test_parallel_client_fetch_page_applies_request_timeout_to_provider_call() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions["timeout"]
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-timeout",
                "session_id": "session-timeout",
                "results": [
                    {
                        "url": "https://example.com/",
                        "full_content": "full page text",
                    }
                ],
                "errors": [],
            },
        )

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        timeout=60.0,
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    await adapter.fetch_page(
        FetchPageRequest(
            provider="parallel",
            url="https://example.com",
            timeout=180.0,
        )
    )

    assert captured["timeout"] == {
        "connect": 190.0,
        "read": 190.0,
        "write": 190.0,
        "pool": 190.0,
    }


async def test_parallel_client_batch_extract_preserves_mixed_result_and_error() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-mixed",
                "session_id": "session-mixed",
                "results": [
                    {
                        "url": "https://example.com/good",
                        "title": "Good",
                        "full_content": "retrieved body",
                    }
                ],
                "errors": [
                    {
                        "url": "https://example.com/missing",
                        "error_type": "fetch_error",
                        "http_status_code": 404,
                        "content": "Not Found",
                    }
                ],
                "warnings": [{"type": "robots", "message": "One URL was unavailable."}],
                "usage": [{"name": "sku_extract_excerpts", "count": 2}],
            },
        )

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = await adapter.extract_pages(
        ExtractPagesRequest(
            urls=("https://example.com/good", "https://example.com/missing"),
            objective="reconstruct the answer",
            max_chars_per_result=200_000,
            max_age_seconds=600,
            disable_cache_fallback=True,
            client_model="gemini-test",
        )
    )

    assert captured["url"] == "https://api.parallel.ai/v1/extract"
    assert captured["json"] == {
        "urls": ["https://example.com/good", "https://example.com/missing"],
        "advanced_settings": {
            "fetch_policy": {
                "max_age_seconds": 600,
                "disable_cache_fallback": True,
            },
            "full_content": {"max_chars_per_result": 200_000},
        },
        "objective": "reconstruct the answer",
        "client_model": "gemini-test",
    }
    assert [page.url for page in result.response.pages] == ["https://example.com/good"]
    assert result.response.errors[0].http_status_code == 404
    assert result.response.warnings[0].warning_type == "robots"
    assert result.billing.billable_units == 2
    assert result.billing.source == "response_body"
    assert result.billing.actual_cost_usd == pytest.approx(price_parallel_extract(url_count=2))


async def test_parallel_mixed_extract_without_supported_usage_bills_submitted_urls() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-fallback",
                "session_id": "session-fallback",
                "results": [{"url": "https://example.com/good", "full_content": "retrieved body"}],
                "errors": [
                    {
                        "url": "https://example.com/missing",
                        "error_type": "fetch_error",
                        "http_status_code": 404,
                    }
                ],
            },
        )

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = await adapter.extract_pages(
        ExtractPagesRequest(urls=("https://example.com/good", "https://example.com/missing"))
    )

    assert result.billing.billable_units == 2
    assert result.billing.source == "request_body"


async def test_parallel_exact_zero_usage_does_not_fall_back_to_submitted_urls() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-zero",
                "session_id": "session-zero",
                "results": [],
                "errors": [
                    {
                        "url": "https://example.com/missing",
                        "error_type": "fetch_error",
                        "http_status_code": 404,
                    }
                ],
                "usage": [{"name": "sku_extract_excerpts", "count": 0}],
            },
        )

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = await adapter.extract_pages(
        ExtractPagesRequest(urls=("https://example.com/missing",))
    )

    assert result.billing.billable_units == 0
    assert result.billing.source == "response_body"


@pytest.mark.parametrize(
    "response_json",
    [
        {
            "extract_id": "extract-invalid-content",
            "session_id": "session-invalid-content",
            "results": [
                {"url": "https://example.com", "full_content": ["not", "text"]}
            ],
            "errors": [],
        },
        {
            "extract_id": "extract-invalid-excerpt",
            "session_id": "session-invalid-excerpt",
            "results": [
                {"url": "https://example.com", "excerpts": [{"not": "text"}]}
            ],
            "errors": [],
        },
        {"results": [], "errors": []},
    ],
)
async def test_parallel_rejects_extract_response_shape_drift(
    response_json: dict[str, Any],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=response_json)

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ToolProviderError):
        await adapter.extract_pages(ExtractPagesRequest(urls=("https://example.com",)))


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"<html>not json</html>", "text/html"),
        (b"[]", "application/json"),
    ],
)
async def test_parallel_rejects_undecodable_or_non_object_extract_response(
    content: bytes,
    content_type: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, content=content, headers={"content-type": content_type})

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ToolProviderError, match="response invalid"):
        await adapter.extract_pages(ExtractPagesRequest(urls=("https://example.com",)))


@pytest.mark.parametrize(
    ("results", "errors"),
    [
        (
            [{"url": "https://example.com/one", "full_content": "one"}],
            [],
        ),
        (
            [
                {"url": "https://example.com/one", "full_content": "one"},
                {"url": "https://example.com/one", "full_content": "duplicate"},
            ],
            [
                {
                    "url": "https://example.com/two",
                    "error_type": "fetch_error",
                }
            ],
        ),
        (
            [{"url": "https://example.com/one", "full_content": "one"}],
            [
                {
                    "url": "https://example.com/one",
                    "error_type": "fetch_error",
                },
                {
                    "url": "https://example.com/two",
                    "error_type": "fetch_error",
                },
            ],
        ),
        (
            [
                {"url": "https://example.com/one", "full_content": "one"},
                {"url": "https://example.com/unknown", "full_content": "unknown"},
            ],
            [
                {
                    "url": "https://example.com/two",
                    "error_type": "fetch_error",
                }
            ],
        ),
        (
            [
                {"url": "https://example.com/one", "full_content": "one"},
                {"url": "/relative", "full_content": "relative"},
            ],
            [
                {
                    "url": "https://example.com/two",
                    "error_type": "fetch_error",
                }
            ],
        ),
    ],
)
async def test_parallel_rejects_invalid_extract_url_partition(
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-invalid-partition",
                "session_id": "session-invalid-partition",
                "results": results,
                "errors": errors,
            },
        )

    adapter = ParallelClient(
        base_url="https://api.parallel.ai",
        api_key="parallel-key",
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ToolProviderError, match="response invalid"):
        await adapter.extract_pages(
            ExtractPagesRequest(
                urls=("https://example.com/one", "https://example.com/two")
            )
        )


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
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-1",
                "session_id": "session-1",
                "results": [],
                "errors": [],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    with pytest.raises(ToolProviderError):
        await adapter.fetch_page(FetchPageRequest(provider="parallel", url="https://example.com"))
