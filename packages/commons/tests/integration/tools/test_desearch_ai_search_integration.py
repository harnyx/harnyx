from __future__ import annotations

import pytest

from harnyx_commons.clients import DESEARCH
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.desearch import DeSearchAiDateFilter, DeSearchClient
from harnyx_commons.tools.desearch_ai_protocol import DeSearchAiDocsResponse
from harnyx_commons.tools.invocation_clients import build_miner_paid_web_search_provider
from harnyx_commons.tools.search_models import SearchAiSearchRequest

pytestmark = [pytest.mark.anyio("asyncio"), pytest.mark.integration]


async def test_desearch_ai_search_live() -> None:
    settings = LlmSettings()
    assert settings.desearch_api_key_value, "DESEARCH_API_KEY must be set"

    desearch = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=settings.llm_timeout_seconds,
        max_concurrent=1,
    )
    try:
        response = await desearch.ai_search_twitter_posts(
            prompt="Bittensor",
            count=10,
            date_filter=DeSearchAiDateFilter.PAST_WEEK,
        )
        assert isinstance(response, DeSearchAiDocsResponse)
    finally:
        await desearch.aclose()


async def test_desearch_search_ai_live() -> None:
    settings = LlmSettings()
    assert settings.desearch_api_key_value, "DESEARCH_API_KEY must be set"

    desearch = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=settings.llm_timeout_seconds,
        max_concurrent=1,
    )
    try:
        billing_response = await desearch.search_ai(
            SearchAiSearchRequest(
                provider="desearch",
                prompt="Find the official Python documentation homepage",
                count=10,
            )
        )
        response = billing_response.response
        assert isinstance(response.data, list)
        assert billing_response.billing is not None
        if billing_response.billing.actual_cost_usd is None:
            assert response.data
        else:
            assert billing_response.billing.source in {"response_body", "response_headers"}
    finally:
        await desearch.aclose()


@pytest.mark.expensive
async def test_miner_paid_desearch_helper_search_ai_live() -> None:
    settings = LlmSettings()
    assert settings.desearch_api_key_value, "DESEARCH_API_KEY must be set"

    desearch = build_miner_paid_web_search_provider(
        provider="desearch",
        api_key=settings.desearch_api_key,
        llm_settings=settings,
    )
    try:
        result = await desearch.search_ai(
            SearchAiSearchRequest(
                provider="desearch",
                prompt="Find the official Python documentation homepage",
                count=10,
            )
        )
        response = result.response
        assert isinstance(response.data, list)
        if result.billing.actual_cost_usd is None:
            assert response.data
    finally:
        await desearch.aclose()
