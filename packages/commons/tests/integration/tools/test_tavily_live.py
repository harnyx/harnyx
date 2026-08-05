from __future__ import annotations

import pytest

from harnyx_commons.clients import TAVILY
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.search_models import FetchPageRequest, SearchWebSearchRequest
from harnyx_commons.tools.tavily import TavilyClient

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_tavily_live_search_and_extract_contract() -> None:
    key = LlmSettings().tavily_api_key_value.strip()
    assert key, "TAVILY_API_KEY must be set"
    client = TavilyClient(
        base_url=TAVILY.base_url,
        api_key=key,
        timeout=TAVILY.timeout_seconds,
        include_payloads_in_logs=False,
    )
    try:
        search = await client.search_web(
            SearchWebSearchRequest.model_validate(
                {
                    "provider": "tavily",
                    "search_queries": ["Example Domain IANA"],
                    "num": 3,
                    "provider_extra": {"search_depth": "ultra-fast"},
                }
            )
        )
        fetch = await client.fetch_page(
            FetchPageRequest(provider="tavily", url="https://example.com")
        )
    finally:
        await client.aclose()

    assert search.response.data
    assert search.billing.provider_request_id
    assert search.billing.usage_count is not None
    assert fetch.response.data[0].content.strip()
    assert fetch.billing.provider_request_id
    assert fetch.billing.usage_count is not None
