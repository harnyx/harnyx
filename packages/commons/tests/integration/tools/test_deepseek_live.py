"""Verify the standalone DeepSeek APIs against their actual response schema."""

import httpx
import pytest

from harnyx_commons.clients import DEEPSEEK
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.tools.deepseek import DeepSeekClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchWebSearchRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_deepseek_live_search_and_fetch_contract() -> None:
    key = LlmSettings().deepseek_api_key_value.strip()
    assert key, "DEEPSEEK_API_KEY must be set"
    client = DeepSeekClient(base_url=DEEPSEEK.base_url, api_key=key, include_payloads_in_logs=False)
    try:
        search = await client.search_web(SearchWebSearchRequest(
            provider="deepseek", search_queries=("Example Domain IANA",), num=3,
        ))
        fetch = await client.fetch_page(FetchPageRequest(provider="deepseek", url="https://example.com"))
    finally:
        await client.aclose()
    assert search.response.data
    assert search.billing.billable_units == 1
    assert fetch.response.data[0].content.strip()
    assert fetch.billing.billable_units == 1


async def test_deepseek_live_rejection_is_confirmed_unbilled() -> None:
    """Exercise a real unauthorized rejection without depending on account balance."""
    async with httpx.AsyncClient(base_url=DEEPSEEK.base_url) as http:
        client = DeepSeekClient(
            base_url=DEEPSEEK.base_url, api_key="invalid-live-contract-key", client=http,
            retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0), include_payloads_in_logs=False,
        )
        with pytest.raises(ToolProviderError) as error:
            await client.search_web(SearchWebSearchRequest(provider="deepseek", search_queries=("Example Domain",)))
    assert error.value.http_status == 401
    assert error.value.billing is not None
    assert error.value.billing.actual_cost_usd == 0.0
