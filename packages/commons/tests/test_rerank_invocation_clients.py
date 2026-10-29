"""Native contract failures must retain the same billing as OpenRouter chat."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.errors import ToolInvocationCancelledError, ToolProviderError
from harnyx_commons.llm.providers.openrouter import OpenRouterRerankClient
from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingCatalog
from harnyx_commons.tools.invocation_clients import OpenRouterRerankProvider
from harnyx_commons.tools.rerank_models import RerankRequest

pytestmark = pytest.mark.anyio("asyncio")
MODEL = "cohere/rerank-v3.5"


def _request() -> RerankRequest:
    return RerankRequest(
        provider="openrouter", model=MODEL, query="capital of France?", documents=["Berlin", "Paris"], top_n=1
    )


def _raw(**updates: Any) -> dict[str, Any]:
    return {
        "model": "rerank-v3.5",
        "provider": "Cohere",
        "id": "gen-rerank-test",
        "results": [{"index": 1, "relevance_score": 0.9, "document": {"text": "Paris"}}],
        "usage": {"cost": 0.001, "search_units": 1},
        **updates,
    }


@asynccontextmanager
async def _provider(
    raw: dict[str, Any], *, discount: float = 0, status: int = 200, cancel: bool = False
) -> AsyncIterator[tuple[OpenRouterRerankProvider, list[httpx.Request]]]:
    paid_requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": MODEL,
                            "canonical_slug": MODEL + "-versioned",
                            "architecture": {"input_modalities": ["text"], "output_modalities": ["rerank"]},
                        }
                    ]
                },
            )
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": MODEL,
                        "endpoints": [
                            {
                                "provider_name": "Cohere",
                                "tag": "cohere",
                                "model_id": MODEL,
                                "pricing": {"prompt": "0", "completion": "0", "discount": discount},
                            }
                        ],
                    }
                },
            )
        assert request.url.path == "/api/v1/rerank"
        paid_requests.append(request)
        if cancel:
            raise asyncio.CancelledError()
        return httpx.Response(status, json=raw)

    transport = httpx.MockTransport(handle)
    catalog = OpenRouterPricingCatalog(transport=transport)
    async with httpx.AsyncClient(base_url="https://openrouter.ai/api/v1/", transport=transport) as client:
        provider = OpenRouterRerankProvider(api_key=SecretStr("test-key"), timeout_seconds=5, pricing_catalog=catalog)
        provider._clients[MODEL] = OpenRouterRerankClient(model=MODEL, api_key=SecretStr("test-key"), client=client)
        try:
            yield provider, paid_requests
        finally:
            await provider.aclose()


@pytest.mark.parametrize(
    "discount,byok,actual,reference",
    [
        (0, False, 0.001, 0.001),
        (0.5, False, 0.001, 0.002),
        (0.5, True, 0.005, 0.009),
        (0, True, 0.005, 0.005),
    ],
)
async def test_rerank_reuses_reported_charge_byok_and_discount_rules(
    discount: float,
    byok: bool,
    actual: float,
    reference: float,
) -> None:
    raw = _raw(
        usage={"cost": 0.001, "is_byok": byok, "cost_details": {"upstream_inference_cost": 0.004}, "search_units": 1}
    )
    async with _provider(raw, discount=discount) as (provider, requests):
        result = await provider.rerank(_request())
    assert len(requests) == 1
    assert result.response.model == MODEL
    assert result.response.results[0].index == 1
    assert result.response.usage is not None and result.response.usage.search_units == 1
    assert result.response.usage.total_tokens is None
    assert result.billing.actual_cost_usd == pytest.approx(actual)
    assert result.billing.reference_cost_usd == pytest.approx(reference)
    assert result.billing.provider_usage["native_model"] == "rerank-v3.5"
    assert result.billing.provider_usage["provider_request_id"] == "gen-rerank-test"


async def test_rerank_missing_byok_flag_uses_reported_charge_without_token_pricing() -> None:
    async with _provider(_raw()) as (provider, _):
        result = await provider.rerank(_request())
    assert result.billing.actual_cost_usd == result.billing.reference_cost_usd == 0.001


@pytest.mark.parametrize(
    "updates",
    [
        {"model": "unrelated-model"},
        {"provider": "Different Provider"},
        {"results": [{"index": 50, "relevance_score": 0.9, "document": {"text": "Paris"}}]},
        {"results": [{"index": 1, "relevance_score": 0.9, "document": {"text": "altered"}}]},
        {"results": []},
    ],
)
async def test_charged_invalid_rerank_retains_actual_cost(updates: dict[str, Any]) -> None:
    async with _provider(_raw(**updates)) as (provider, _):
        with pytest.raises(ToolProviderError) as error:
            await provider.rerank(_request())
    assert error.value.billing is not None
    assert error.value.billing.actual_cost_usd == 0.001
    assert error.value.billing.provider_usage["inference"] == "attempted"


@pytest.mark.parametrize(
    "usage", [{}, {"cost": "bad"}, {"cost": True}, {"cost": -0.001}, {"cost": 0.001, "is_byok": True}]
)
async def test_rerank_does_not_invent_missing_or_malformed_charges(usage: dict[str, Any]) -> None:
    async with _provider(_raw(usage=usage)) as (provider, _):
        with pytest.raises(ToolProviderError) as error:
            await provider.rerank(_request())
    assert error.value.billing is not None
    assert error.value.billing.actual_cost_usd is None
    assert error.value.billing.reference_cost_unavailable


async def test_complete_discount_is_rejected_before_paid_rerank_dispatch() -> None:
    async with _provider(_raw(), discount=1) as (provider, paid):
        with pytest.raises(ToolProviderError) as error:
            await provider.rerank(_request())
    assert paid == []
    assert error.value.billing is not None
    assert error.value.billing.actual_cost_usd == error.value.billing.reference_cost_usd == 0
    assert error.value.billing.provider_usage["inference"] == "not_dispatched"


async def test_cancelled_paid_rerank_is_unknown_not_free() -> None:
    async with _provider(_raw(), cancel=True) as (provider, paid):
        with pytest.raises(ToolInvocationCancelledError) as error:
            await provider.rerank(_request())
    assert len(paid) == 1
    assert error.value.billing.actual_cost_usd is None
    assert error.value.billing.provider_usage["inference"] == "attempted"


async def test_error_http_response_retains_observed_rerank_charge() -> None:
    async with _provider(_raw(), status=500) as (provider, _):
        with pytest.raises(ToolProviderError) as error:
            await provider.rerank(_request())
    assert error.value.http_status == 500
    assert error.value.billing.actual_cost_usd == 0.001
