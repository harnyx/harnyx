from __future__ import annotations

import pytest

from harnyx_commons.clients import PARALLEL
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.pricing import price_parallel_extract, price_parallel_search
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
    request = SearchWebSearchRequest(provider="parallel", search_queries=("python", "documentation"), num=3)
    try:
        billing_response = await client.search_web(request)
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
            FetchPageRequest(provider="parallel", url="https://example.com")
        )
        response = billing_response.response
        assert len(response.data) == 1
        assert response.data[0].url == "https://example.com"
        assert response.data[0].content
        assert billing_response.billing is not None
        assert billing_response.billing.billable_units == len(billing_response.response.data)
        assert billing_response.billing.provider_request_id is not None
        assert billing_response.billing.source == "response_results"
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
