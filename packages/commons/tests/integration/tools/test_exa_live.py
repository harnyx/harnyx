from __future__ import annotations

import pytest

from harnyx_commons.clients import EXA
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.exa import ExaClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchWebSearchRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_exa_live_search_and_contents_contract() -> None:
    key = LlmSettings().exa_api_key_value.strip()
    assert key, "EXA_API_KEY must be set"
    client = ExaClient(
        base_url=EXA.base_url,
        api_key=key,
        timeout=EXA.timeout_seconds,
        include_payloads_in_logs=False,
    )
    try:
        search = await client.search_web(
            SearchWebSearchRequest(provider="exa", search_queries=("Example Domain IANA",), num=3)
        )
        fetch = await client.fetch_page(
            FetchPageRequest(provider="exa", url="https://example.com")
        )
    finally:
        await client.aclose()

    assert search.response.data
    assert search.billing.actual_cost_usd is not None
    assert search.billing.provider_request_id
    assert fetch.response.data[0].content.strip()
    assert fetch.billing.actual_cost_usd is not None
