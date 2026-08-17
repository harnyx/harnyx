from __future__ import annotations

import pytest

from harnyx_commons.clients import PARALLEL
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.pricing import price_parallel_extract, price_parallel_search
from harnyx_commons.tools.extraction_models import ExtractPagesRequest
from harnyx_commons.tools.invocation_clients import build_miner_paid_web_search_provider
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchAiSearchRequest, SearchWebSearchRequest

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


def _build_parallel_client(settings: LlmSettings) -> ParallelClient:
    assert settings.parallel_api_key_value, "PARALLEL_API_KEY must be set"
    return ParallelClient(
        base_url=settings.parallel_base_url,
        api_key=settings.parallel_api_key_value,
        timeout=PARALLEL.timeout_seconds,
        max_concurrent=settings.parallel_max_concurrent,
    )


async def test_parallel_search_web_live() -> None:
    settings = LlmSettings()
    client = _build_parallel_client(settings)
    request = SearchWebSearchRequest.model_validate(
        {
            "provider": "parallel",
            "search_queries": ["python", "documentation"],
            "num": 3,
            "provider_extra": {"mode": "turbo"},
        }
    )
    try:
        billing_response = await client.search_web(request)
        assert isinstance(billing_response.response.data, list)
        assert billing_response.billing is not None
        assert billing_response.billing.billable_units == len(billing_response.response.data)
        assert billing_response.billing.provider_request_id is not None
        assert billing_response.billing.source == "response_results"
        assert billing_response.billing.actual_cost_usd == pytest.approx(
            price_parallel_search(
                billable_results=billing_response.billing.billable_units,
                mode="turbo",
            )
        )
        assert (
            price_parallel_search(
                billable_results=billing_response.billing.billable_units,
                mode="turbo",
            )
            >= 0.001
        )
    finally:
        await client.aclose()


@pytest.mark.flaky(reruns=1)
@pytest.mark.expensive
async def test_parallel_batch_extract_live_preserves_success_and_unavailable_url() -> None:
    settings = LlmSettings()
    client = _build_parallel_client(settings)
    urls = (
        "https://www.rfc-editor.org/rfc/rfc2606.html",
        "https://www.rfc-editor.org/rfc/harnyx-intentionally-missing-source-847.html",
    )
    try:
        result = await client.extract_pages(
            ExtractPagesRequest(
                urls=urls,
                objective="Retrieve the public page bodies for an integration contract check.",
                max_chars_per_result=20_000,
                max_age_seconds=600,
                disable_cache_fallback=True,
                client_model="integration-test",
            )
        )
    finally:
        await client.aclose()

    extraction_summary = {
        "pages": [(page.url, bool(page.content)) for page in result.response.pages],
        "errors": [
            (error.url, error.error_type, error.http_status_code) for error in result.response.errors
        ],
    }
    assert any(
        page.url.rstrip("/") == urls[0].rstrip("/") and page.content for page in result.response.pages
    ), extraction_summary
    assert any(
        error.url == urls[1] and error.error_type and error.http_status_code is not None
        for error in result.response.errors
    ), extraction_summary
    assert result.billing.source in {"response_body", "request_body"}
    expected_units = 1 if result.billing.source == "response_body" else 2
    assert result.billing.billable_units == expected_units
    assert result.billing.actual_cost_usd == pytest.approx(
        price_parallel_extract(url_count=expected_units)
    )


async def test_parallel_search_ai_live() -> None:
    settings = LlmSettings()
    client = _build_parallel_client(settings)
    try:
        request = SearchAiSearchRequest(
            provider="parallel",
            prompt="Find the official Python documentation homepage",
            count=10,
        )
        billing_response = await client.search_ai(request)
        assert isinstance(billing_response.response.data, list)
        assert billing_response.billing is not None
        assert billing_response.billing.billable_units == len(billing_response.response.data)
        assert billing_response.billing.provider_request_id is not None
        assert billing_response.billing.source == "response_results"
        assert billing_response.billing.actual_cost_usd == pytest.approx(
            price_parallel_search(billable_results=billing_response.billing.billable_units)
        )
        assert price_parallel_search(billable_results=billing_response.billing.billable_units) >= 0.005
    finally:
        await client.aclose()


async def test_parallel_fetch_page_live() -> None:
    settings = LlmSettings()
    client = _build_parallel_client(settings)
    try:
        billing_response = await client.fetch_page(
            FetchPageRequest.model_validate(
                {
                    "provider": "parallel",
                    "url": "https://example.com",
                    "provider_extra": {"max_chars_total": 20_000},
                }
            )
        )
        response = billing_response.response
        assert len(response.data) == 1
        assert response.data[0].url.rstrip("/") == "https://example.com"
        assert response.data[0].content
        assert billing_response.billing is not None
        assert billing_response.billing.billable_units == len(billing_response.response.data)
        assert billing_response.billing.provider_request_id is not None
        assert billing_response.billing.source in {"response_body", "request_body"}
        assert billing_response.billing.actual_cost_usd == pytest.approx(
            price_parallel_extract(url_count=billing_response.billing.billable_units)
        )
        assert price_parallel_extract(url_count=billing_response.billing.billable_units) == pytest.approx(0.001)
    finally:
        await client.aclose()


@pytest.mark.expensive
async def test_miner_paid_parallel_helper_search_ai_live() -> None:
    settings = LlmSettings()
    assert settings.parallel_api_key_value, "PARALLEL_API_KEY must be set"
    client = build_miner_paid_web_search_provider(
        provider="parallel",
        api_key=settings.parallel_api_key,
        llm_settings=settings,
    )
    try:
        request = SearchAiSearchRequest(
            provider="parallel",
            prompt="Find the official Python documentation homepage",
            count=10,
        )
        result = await client.search_ai(request)
        response = result.response
        assert isinstance(response.data, list)
        assert result.billing.actual_cost_usd is not None
    finally:
        await client.aclose()
