from __future__ import annotations

import logging

import httpx
import pytest

from caster_commons.tools.desearch import DeSearchClient
from caster_commons.tools.search_models import SearchWebSearchRequest, SearchXSearchRequest

pytestmark = pytest.mark.anyio("asyncio")


def _tweet(*, tweet_id: int) -> dict[str, object]:
    return {
        "id": str(tweet_id),
        "created_at": "2026-01-14T00:00:00Z",
        "text": f"tweet {tweet_id}",
        "url": f"https://x.com/user/status/{tweet_id}",
        "like_count": 1,
        "user": {"username": "user"},
    }


def _web_result(*, link: str) -> dict[str, object]:
    return {"link": link, "title": "t", "snippet": "s"}


async def test_iter_search_links_twitter_pages_adds_max_id_and_stops_on_empty() -> None:
    queries: list[str] = []

    tweets_page_1 = [_tweet(tweet_id=10), _tweet(tweet_id=8)]
    tweets_page_2 = [_tweet(tweet_id=7)]
    tweets_page_3: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/twitter"
        query = request.url.params.get("query")
        assert query is not None
        queries.append(query)

        if len(queries) == 1:
            assert query == "hello"
            return httpx.Response(200, json=tweets_page_1, request=request)
        if len(queries) == 2:
            assert query == "hello max_id:7"
            return httpx.Response(200, json=tweets_page_2, request=request)
        if len(queries) == 3:
            assert query == "hello max_id:6"
            return httpx.Response(200, json=tweets_page_3, request=request)
        raise AssertionError(f"unexpected request #{len(queries)}: {request.url!s}")

    client = DeSearchClient(
        base_url="https://desearch.example",
        api_key="token-123",
        client=httpx.AsyncClient(
            base_url="https://desearch.example",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        pages = []
        async for page in client.iter_search_links_twitter_pages(
            SearchXSearchRequest(
                query="hello",
                count=100,
                sort="Latest",
            )
        ):
            pages.append(page)
        assert [p.cursor for p in pages] == [None, 7]
        assert [len(p.posts) for p in pages] == [2, 1]
        assert queries == ["hello", "hello max_id:7", "hello max_id:6"]
    finally:
        await client.aclose()


async def test_iter_search_links_twitter_pages_stops_if_max_id_ignored(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="caster_commons.tools.desearch.calls")

    queries: list[str] = []
    tweets_page_1 = [_tweet(tweet_id=10), _tweet(tweet_id=8)]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/twitter"
        query = request.url.params.get("query")
        assert query is not None
        queries.append(query)

        if len(queries) == 1:
            return httpx.Response(200, json=tweets_page_1, request=request)
        if len(queries) == 2:
            # Simulate DeSearch ignoring max_id by returning the same page again.
            return httpx.Response(200, json=tweets_page_1, request=request)
        raise AssertionError(f"unexpected request #{len(queries)}: {request.url!s}")

    client = DeSearchClient(
        base_url="https://desearch.example",
        api_key="token-123",
        client=httpx.AsyncClient(
            base_url="https://desearch.example",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        pages = []
        async for page in client.iter_search_links_twitter_pages(
            SearchXSearchRequest(query="hello", count=100, sort="Latest")
        ):
            pages.append(page)

        assert len(pages) == 1
        assert queries == ["hello", "hello max_id:7"]

        stop_records = [
            record
            for record in caplog.records
            if record.msg == "desearch.search_links_twitter.pagination.stopped_max_id_ignored"
        ]
        assert len(stop_records) == 1
    finally:
        await client.aclose()


async def test_iter_search_links_web_pages_increments_start_and_stops_on_repeat_first_link() -> None:
    starts: list[int] = []

    page_1 = [
        _web_result(link="https://a.example"),
        _web_result(link="https://b.example"),
        _web_result(link="https://c.example"),
    ]
    page_2 = [_web_result(link="https://a.example")]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/web"
        start = request.url.params.get("start")
        assert start is not None
        starts.append(int(start))

        if len(starts) == 1:
            assert int(start) == 0
            return httpx.Response(200, json=page_1, request=request)
        if len(starts) == 2:
            assert int(start) == 3
            return httpx.Response(200, json=page_2, request=request)
        raise AssertionError(f"unexpected request #{len(starts)}: {request.url!s}")

    client = DeSearchClient(
        base_url="https://desearch.example",
        api_key="token-123",
        client=httpx.AsyncClient(
            base_url="https://desearch.example",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        pages = []
        async for page in client.iter_search_links_web_pages(SearchWebSearchRequest(query="hello", num=100)):
            pages.append(page)
        assert len(pages) == 1
        assert starts == [0, 3]
    finally:
        await client.aclose()
