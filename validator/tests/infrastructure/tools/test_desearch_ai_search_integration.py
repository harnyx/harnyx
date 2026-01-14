from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from caster_commons.config.llm import LlmSettings
from caster_commons.tools.desearch import DeSearchClient

pytestmark = [pytest.mark.anyio("asyncio"), pytest.mark.integration]


async def test_desearch_ai_search_live() -> None:
    settings = LlmSettings()
    assert settings.desearch_api_key_value, "DESEARCH_API_KEY must be set"

    desearch = DeSearchClient(
        base_url=settings.desearch_base_url,
        api_key=settings.desearch_api_key_value,
        timeout=settings.llm_timeout_seconds,
        max_concurrent=1,
    )
    try:
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=7)
        tweets = await desearch.ai_search_twitter_posts(
            prompt="Bittensor",
            count=10,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        assert isinstance(tweets, list)
    finally:
        await desearch.aclose()

