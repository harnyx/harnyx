from __future__ import annotations

import pytest

from harnyx_commons.clients import PARALLEL
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.pricing import price_parallel_extract, price_parallel_search
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
    request = SearchWebSearchRequest(search_queries=("python", "documentation"), num=3)
    try:
        response = await client.search_web(request)
        assert isinstance(response.data, list)
        assert price_parallel_search(requested_results=request.num) == pytest.approx(0.005)
    finally:
        await client.aclose()


async def test_parallel_search_ai_live() -> None:
    settings = LlmSettings()
    client = _build_parallel_client(settings)
    try:
        request = SearchAiSearchRequest(
            prompt="Find the official Python documentation homepage",
            count=10,
        )
        response = await client.search_ai(request)
        assert isinstance(response.data, list)
        assert price_parallel_search(requested_results=request.count) == pytest.approx(0.005)
    finally:
        await client.aclose()


async def test_parallel_fetch_page_live() -> None:
    settings = LlmSettings()
    client = _build_parallel_client(settings)
    try:
        response = await client.fetch_page(FetchPageRequest(url="https://example.com"))
        assert len(response.data) == 1
        assert response.data[0].url == "https://example.com"
        assert response.data[0].content
        assert price_parallel_extract(url_count=1) == pytest.approx(0.001)
    finally:
        await client.aclose()
