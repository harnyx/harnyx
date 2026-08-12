from __future__ import annotations

import pytest

from harnyx_commons.clients import FIRECRAWL
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.firecrawl import FirecrawlClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchWebSearchRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


def _api_key() -> str:
    value = LlmSettings().firecrawl_api_key_value.strip()
    assert value, "FIRECRAWL_API_KEY must be set"
    return value


async def test_firecrawl_live_search_and_scrape_contract() -> None:
    client = FirecrawlClient(
        base_url=FIRECRAWL.base_url,
        api_key=_api_key(),
        timeout=FIRECRAWL.timeout_seconds,
        include_payloads_in_logs=False,
    )
    try:
        search = await client.search_web(
            SearchWebSearchRequest(
                provider="firecrawl",
                search_queries=("Example Domain IANA",),
                num=3,
            )
        )
        scrape = await client.fetch_page(
            FetchPageRequest.model_validate(
                {
                    "provider": "firecrawl",
                    "url": "https://example.com",
                    "provider_extra": {"formats": ["markdown", "rawHtml"]},
                }
            )
        )
    finally:
        await client.aclose()

    assert search.response.data
    assert all(result.link.strip() for result in search.response.data)
    assert search.billing.actual_cost_provider == "firecrawl"
    assert len(scrape.response.data) == 2
    assert all(result.url.strip() for result in scrape.response.data)
    assert scrape.response.data[0].content.strip()
    assert "<" in scrape.response.data[1].content
    assert scrape.billing.actual_cost_provider == "firecrawl"
