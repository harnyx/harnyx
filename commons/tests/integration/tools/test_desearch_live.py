from __future__ import annotations

import asyncio

import pytest

from caster_commons.clients import DESEARCH
from caster_commons.config.llm import LlmSettings
from caster_commons.tools.desearch import DeSearchClient
from caster_commons.tools.search_models import SearchWebSearchRequest, SearchXSearchRequest

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


async def test_search_web_live() -> None:
    settings = LlmSettings()

    client = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=DESEARCH.timeout_seconds,
    )
    request = SearchWebSearchRequest(query="United States latest news", num=5)

    result = await client.search_links_web(request)
    await client.aclose()

    assert isinstance(result.data, list)


async def test_search_twitter_live() -> None:
    settings = LlmSettings()

    client = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=DESEARCH.timeout_seconds,
    )
    request = SearchXSearchRequest(query="US", count=3)

    result = await client.search_links_twitter(request)
    await client.aclose()

    assert isinstance(result.data, list)


async def test_fetch_twitter_post_live() -> None:
    settings = LlmSettings()

    client = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=DESEARCH.timeout_seconds,
    )
    try:
        search = await client.search_links_twitter(SearchXSearchRequest(query="US", count=3))
        post_id = next((post.id for post in search.data if post.id), None)
        assert post_id, "desearch twitter search returned no ids for fetch_twitter_post test"
        post = await client.fetch_twitter_post(post_id=post_id)
        assert post is not None
        assert post.id == post_id
    finally:
        await client.aclose()


async def test_search_twitter_live_paginates_with_max_id() -> None:
    settings = LlmSettings()

    client = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=DESEARCH.timeout_seconds,
    )

    try:
        query = "US"
        first_attempts = 3
        first = None
        first_ids: list[int] = []
        for attempt in range(first_attempts):
            try:
                first = await client.search_links_twitter(
                    SearchXSearchRequest(
                        query=query,
                        count=100,
                        sort="Latest",
                    ),
                )
            except RuntimeError:
                if attempt + 1 >= first_attempts:
                    raise
                await asyncio.sleep(1)
                continue
            first_ids = [int(post.id) for post in first.data if post.id and post.id.isdigit()]
            if first_ids:
                break
            if attempt + 1 < first_attempts:
                await asyncio.sleep(1)
        if not first_ids:
            raise AssertionError("desearch twitter search returned no tweet ids to paginate")
        cursor = min(first_ids) - 1

        second_query = f"{query} max_id:{cursor}"
        attempts = 3
        min_under_cursor = 10
        last_second_ids: list[int] = []
        for attempt in range(attempts):
            second = await client.search_links_twitter(
                SearchXSearchRequest(
                    query=second_query,
                    count=100,
                    sort="Latest",
                ),
            )
            second_ids = [int(post.id) for post in second.data if post.id and post.id.isdigit()]
            last_second_ids = second_ids
            under_cursor = [tweet_id for tweet_id in second_ids if tweet_id <= cursor]
            if len(under_cursor) >= min_under_cursor:
                return
            if attempt + 1 < attempts:
                await asyncio.sleep(1)
    finally:
        await client.aclose()

    assert last_second_ids
    under_cursor = [tweet_id for tweet_id in last_second_ids if tweet_id <= cursor]
    assert (
        len(under_cursor) >= min_under_cursor
    ), (
        "max_id pagination yielded too few under-cursor tweets "
        f"(cursor={cursor}, under_cursor={len(under_cursor)}, sample={last_second_ids[:5]})"
    )
