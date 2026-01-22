from __future__ import annotations

import logging

import httpx
import pytest

from caster_commons.tools.desearch import DeSearchClient
from caster_commons.tools.search_models import SearchXSearchRequest

pytestmark = pytest.mark.anyio("asyncio")


def _tweet(*, tweet_id: int, created_at: str) -> dict[str, object]:
    return {
        "id": str(tweet_id),
        "created_at": created_at,
        "text": f"tweet {tweet_id}",
        "url": f"https://x.com/user/status/{tweet_id}",
        "like_count": 123,
        "user": {"username": "user"},
    }


async def test_search_links_twitter_emits_summary_log(caplog: pytest.LogCaptureFixture) -> None:
    tweets = [
        _tweet(tweet_id=100 + idx, created_at=f"2026-01-14T00:00:{idx:02d}Z") for idx in range(25)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/twitter"
        return httpx.Response(200, json=tweets, request=request)

    caplog.set_level(logging.INFO, logger="caster_commons.tools.desearch.calls")
    client = DeSearchClient(
        base_url="https://desearch.example",
        api_key="token-123",
        client=httpx.AsyncClient(
            base_url="https://desearch.example",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        await client.search_links_twitter(SearchXSearchRequest(query="test", count=100))
    finally:
        await client.aclose()

    summary_records = [
        record
        for record in caplog.records
        if record.name == "caster_commons.tools.desearch.calls"
        and record.msg == "desearch.search_links_twitter.summary"
    ]
    assert len(summary_records) == 1
    record = summary_records[0]

    assert record.data["return_count"] == 25
    assert record.data["min_id"] == 100
    assert record.data["max_id"] == 124
    assert record.data["min_created_at"] == "2026-01-14T00:00:00Z"
    assert record.data["max_created_at"] == "2026-01-14T00:00:24Z"
    assert record.json_fields["sample_tweets"] == tweets[:20]

