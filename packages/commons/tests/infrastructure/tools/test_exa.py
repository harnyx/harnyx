from __future__ import annotations

import json

import httpx
import pytest

from harnyx_commons.tools.exa import ExaClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchWebSearchRequest

pytestmark = pytest.mark.anyio("asyncio")


async def test_exa_search_maps_retrieval_controls_and_cost() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "requestId": "exa-1",
                "costDollars": {"total": 0.007},
                "results": [{"url": "https://example.com", "title": "Example"}],
            },
        )

    http_client = httpx.AsyncClient(base_url="https://api.exa.ai", transport=httpx.MockTransport(handler))
    client = ExaClient(base_url="https://api.exa.ai", api_key="exa-key", client=http_client)
    result = await client.search_web(
        SearchWebSearchRequest.model_validate(
            {
                "provider": "exa",
                "search_queries": ["one", "two"],
                "num": 3,
                "provider_extra": {"type": "instant", "moderation": True},
            }
        )
    )

    assert requests[0].headers["x-api-key"] == "exa-key"
    assert json.loads(requests[0].content) == {
        "query": "(one) OR (two)",
        "type": "instant",
        "moderation": True,
        "numResults": 3,
    }
    assert result.response.data[0].link == "https://example.com"
    assert result.billing.actual_cost_usd == pytest.approx(0.007)
    assert result.billing.provider_request_id == "exa-1"
    await http_client.aclose()


async def test_exa_fetch_uses_contents_without_generated_outputs() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [{"url": "https://example.com", "title": "Example", "text": "body"}],
                "costDollars": {"total": 0.003},
            },
        )

    http_client = httpx.AsyncClient(base_url="https://api.exa.ai", transport=httpx.MockTransport(handler))
    client = ExaClient(base_url="https://api.exa.ai", api_key="exa-key", client=http_client)
    result = await client.fetch_page(
        FetchPageRequest.model_validate(
            {"provider": "exa", "url": "https://example.com", "provider_extra": {"max_age_hours": 2}}
        )
    )

    assert payloads == [{"urls": ["https://example.com"], "text": True, "maxAgeHours": 2}]
    assert result.response.data[0].content == "body"
    await http_client.aclose()
