from __future__ import annotations

import logging

import httpx
import pytest

import harnyx_commons.tools.desearch as desearch_module
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.tools.desearch import DeSearchClient
from harnyx_commons.tools.search_models import SearchWebSearchRequest, SearchXSearchRequest

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


async def test_desearch_client_can_suppress_request_and_raw_response_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="harnyx_commons.tools.desearch.calls")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"link": "https://example.com/raw-provider-envelope"}]},
            request=request,
        )

    client = DeSearchClient(
        base_url="https://desearch.example",
        api_key="token-123",
        client=httpx.AsyncClient(
            base_url="https://desearch.example",
            transport=httpx.MockTransport(handler),
        ),
        include_payloads_in_logs=False,
    )

    await client._post("/search", {"prompt": "private-query"})

    record = next(record for record in caplog.records if record.msg == "desearch.request.complete")
    assert not hasattr(record, "json_fields")
    assert "private-query" not in str(record.__dict__)
    assert "raw-provider-envelope" not in str(record.__dict__)


async def test_desearch_payload_suppression_disables_automatic_otel_exception_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_kwargs: list[dict[str, object]] = []

    class Span:
        def set_attributes(self, attributes: dict[str, object]) -> None:
            del attributes

    class SpanScope:
        def __enter__(self) -> Span:
            return Span()

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

    class Tracer:
        def start_as_current_span(self, name: str, **kwargs: object) -> SpanScope:
            start_kwargs.append({"name": name, **kwargs})
            return SpanScope()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "raw-provider-envelope"}, request=request)

    monkeypatch.setattr(desearch_module.trace, "get_tracer", lambda _: Tracer())
    client = DeSearchClient(
        base_url="https://desearch.example",
        api_key="token-123",
        client=httpx.AsyncClient(
            base_url="https://desearch.example",
            transport=httpx.MockTransport(handler),
        ),
        include_payloads_in_logs=False,
    )

    with pytest.raises(ToolProviderError):
        await client._post("/search", {"prompt": "private-query"})

    assert start_kwargs[0]["record_exception"] is False
    assert start_kwargs[0]["set_status_on_exception"] is False


async def test_desearch_403_is_typed_as_authentication_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "raw-provider-envelope"}, request=request)

    client = DeSearchClient(
        base_url="https://desearch.example",
        api_key="token-123",
        client=httpx.AsyncClient(
            base_url="https://desearch.example",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await client._post("/search", {"prompt": "private-query"})

    assert exc_info.value.failure_code is ToolProviderFailureCode.AUTHENTICATION_FAILED
    assert exc_info.value.http_status == 403


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
    caplog.set_level(logging.INFO, logger="harnyx_commons.tools.desearch.calls")

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
    queries: list[str] = []

    page_1 = [
        _web_result(link="https://a.example"),
        _web_result(link="https://b.example"),
        _web_result(link="https://c.example"),
    ]
    page_2 = [_web_result(link="https://a.example")]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/web"
        start = request.url.params.get("start")
        query = request.url.params.get("query")
        assert start is not None
        assert query is not None
        starts.append(int(start))
        queries.append(query)

        if len(starts) == 1:
            assert int(start) == 0
            assert query == "(alpha phrase) OR (beta)"
            return httpx.Response(200, json=page_1, request=request)
        if len(starts) == 2:
            assert int(start) == 3
            assert query == "(alpha phrase) OR (beta)"
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
        async for page in client.iter_search_links_web_pages(
            SearchWebSearchRequest(provider="desearch", search_queries=("alpha phrase", "beta"), num=100)
        ):
            pages.append(page)
        assert len(pages) == 1
        assert starts == [0, 3]
        assert queries == ["(alpha phrase) OR (beta)", "(alpha phrase) OR (beta)"]
    finally:
        await client.aclose()
