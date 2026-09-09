"""Assignment download bounds must close provider streams without retrying paid work."""

import asyncio
import gzip
import json

import httpx
import pytest

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.tools.desearch import DeSearchClient
from harnyx_commons.tools.exa import ExaClient
from harnyx_commons.tools.firecrawl import FirecrawlClient
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.search_http import read_search_response
from harnyx_commons.tools.search_models import FetchPageRequest
from harnyx_commons.tools.tavily import TavilyClient


class ResponseStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider,client_type",
    [
        ("desearch", DeSearchClient),
        ("parallel", ParallelClient),
        ("firecrawl", FirecrawlClient),
        ("exa", ExaClient),
        ("tavily", TavilyClient),
    ],
)
@pytest.mark.parametrize("status", [200, 500])
@pytest.mark.parametrize("compressed", [False, True])
async def test_oversized_download_closes_without_retry(provider, client_type, status, compressed):
    chunks = [b"x" * 65, b"must not consume the rest"]
    if compressed:
        chunks[0] = gzip.compress(chunks[0])
    stream = ResponseStream(chunks)
    attempts = 0

    async def handle(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            status,
            headers={
                **({"Content-Encoding": "gzip"} if compressed else {}),
                "x-desearch-cost-usd": "0.125",
            },
            stream=stream,
        )

    async with httpx.AsyncClient(base_url="https://provider.example", transport=httpx.MockTransport(handle)) as http:
        client = client_type(
            base_url="https://provider.example", api_key="test-key", client=http, max_response_bytes=64
        )
        with pytest.raises(ToolProviderError, match="response exceeds") as raised:
            await client.fetch_page(FetchPageRequest(provider=provider, url="https://source.example"))
        assert attempts == 1
        assert stream.reads == 1
        assert stream.closed
        if provider == "desearch":
            assert raised.value.billing.actual_cost_usd == 0.125
        else:
            assert raised.value.billing is None


@pytest.mark.anyio
@pytest.mark.parametrize("length", [2 * 1024 * 1024 - 1, 2 * 1024 * 1024, 2 * 1024 * 1024 + 1])
async def test_decoded_download_exact_boundary(length):
    limit = 2 * 1024 * 1024
    body = b"x" * length
    stream = ResponseStream([body[:limit], body[limit:]])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    ) as client:
        async with client.stream("GET", "https://provider.example") as response:
            if length > limit:
                with pytest.raises(ToolProviderError, match="response exceeds"):
                    await read_search_response(response, provider="parallel", max_response_bytes=limit)
            else:
                assert await read_search_response(response, provider="parallel", max_response_bytes=limit) == body
    assert stream.closed


@pytest.mark.anyio
async def test_search_stream_cancellation_releases_connection():
    waiting = asyncio.Event()
    closed = asyncio.Event()

    class BlockedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            waiting.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async with httpx.AsyncClient(
        base_url="https://provider.example",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=BlockedStream())),
    ) as http:
        client = ParallelClient(base_url="https://provider.example", api_key="key", client=http, max_response_bytes=64)
        pending = asyncio.create_task(
            client.fetch_page(FetchPageRequest(provider="parallel", url="https://source.example"))
        )
        try:
            async with asyncio.timeout(5):
                await waiting.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        assert closed.is_set()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider,client_type,payload",
    [
        ("desearch", DeSearchClient, "evidence"),
        (
            "parallel",
            ParallelClient,
            {
                "extract_id": "extract-1",
                "session_id": "session-1",
                "errors": [],
                "results": [{"url": "https://source.example", "full_content": "evidence"}],
            },
        ),
        ("firecrawl", FirecrawlClient, {"success": True, "data": {"markdown": "evidence", "metadata": {}}}),
        ("exa", ExaClient, {"results": [{"url": "https://source.example", "text": "evidence"}]}),
        ("tavily", TavilyClient, {"results": [{"url": "https://source.example", "raw_content": "evidence"}]}),
    ],
)
async def test_valid_stream_at_limit_keeps_provider_evidence(provider, client_type, payload):
    body = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
    stream = ResponseStream([body[:3], body[3:]])
    async with httpx.AsyncClient(
        base_url="https://provider.example",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)),
    ) as http:
        client = client_type(
            base_url="https://provider.example", api_key="key", client=http, max_response_bytes=len(body)
        )
        request = {"provider": provider, "url": "https://source.example"}
        if provider == "desearch":
            request["provider_extra"] = {"format": "html"}
        result = await client.fetch_page(FetchPageRequest.model_validate(request))
        assert result.response.data[0].content == "evidence"
        assert stream.closed
