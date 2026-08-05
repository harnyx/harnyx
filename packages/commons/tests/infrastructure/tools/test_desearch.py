from __future__ import annotations

from typing import Any

import httpx
import pytest

from harnyx_commons.tools.desearch import DeSearchClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchWebSearchRequest

pytestmark = pytest.mark.anyio("asyncio")


async def test_desearch_client_search_web_applies_request_timeout_to_provider_call() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions["timeout"]
        return httpx.Response(
            200,
            json={"data": []},
            headers={
                "x-harnyx-cost-usd": "0",
                "x-harnyx-billable-units": "0",
            },
        )

    adapter = DeSearchClient(
        base_url="https://api.desearch.ai",
        api_key="desearch-key",
        timeout=60.0,
        client=httpx.AsyncClient(
            base_url="https://api.desearch.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    await adapter.search_web(
        SearchWebSearchRequest(
            provider="desearch",
            search_queries=("timeout parity",),
            timeout=180.0,
        )
    )

    assert captured["timeout"] == {
        "connect": 190.0,
        "read": 190.0,
        "write": 190.0,
        "pool": 190.0,
    }


async def test_desearch_client_fetch_page_accepts_raw_html_response() -> None:
    html = "<html><body>Example</body></html>"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "html"
        return httpx.Response(
            200,
            text=html,
            headers={
                "content-type": "text/html",
                "x-harnyx-cost-usd": "0",
                "x-harnyx-billable-units": "1",
            },
        )

    adapter = DeSearchClient(
        base_url="https://api.desearch.ai",
        api_key="desearch-key",
        client=httpx.AsyncClient(
            base_url="https://api.desearch.ai",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = await adapter.fetch_page(
        FetchPageRequest.model_validate(
            {
                "provider": "desearch",
                "url": "https://example.com",
                "provider_extra": {"format": "html"},
            }
        )
    )

    assert result.response.data[0].content == html
