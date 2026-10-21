from __future__ import annotations

import copy
from decimal import Decimal
from typing import cast

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.errors import ProviderCredentialUnavailableError, ToolProviderError
from harnyx_commons.json_types import JsonObject
from harnyx_commons.tools.decision_models import DecisionQueryRequest
from harnyx_commons.tools.invocation_clients import CachedDecisionProviderRegistry, _decision_result

pytestmark = pytest.mark.anyio("asyncio")


def _snapshot():
    from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingSnapshot, PricingEndpoint

    return OpenRouterPricingSnapshot(
        "cloudflare/clef-flash",
        (
            PricingEndpoint.model_validate(
                {
                    "provider_name": "Cloudflare",
                    "tag": "cloudflare",
                    "model_id": "cloudflare/clef-flash-versioned",
                    "pricing": {"prompt": "0.00000009", "completion": "0", "discount": 0},
                }
            ),
        ),
        "now",
    )


def _request() -> DecisionQueryRequest:
    return DecisionQueryRequest.model_validate(
        {
            "provider": "openrouter",
            "model": "cloudflare/clef-flash",
            "state": "Paris is in France",
            "questions": {
                "b": {"type": "boolean", "instructions": "Correct?"},
                "c": {"type": "choice", "instructions": "Country?", "criteria": {"fr": "France", "de": "Germany"}},
                "s": {"type": "score", "instructions": "Rate?", "criteria": ["Low", "High"]},
            },
        }
    )


def _raw() -> JsonObject:
    return {
        "model": "cloudflare/clef-flash-versioned",
        "id": "native-id",
        "provider": "Cloudflare",
        "answers": {
            "b": {"type": "noul", "noul": 0.9},
            "c": {"type": "choice", "choice": "fr", "probabilities": {"fr": 0.9, "de": 0.1}},
            "s": {"type": "score", "score": 0.8},
        },
        "usage": {"input_tokens": 1000, "output_tokens": 10, "cost": 0.002},
    }


def test_reported_cost_wins_and_native_answers_and_usage_normalize() -> None:
    result = _decision_result(_request(), _raw(), snapshot=_snapshot())
    assert result.actual_cost_usd == 0.002
    assert result.response.answers["b"].model_dump() == {"type": "boolean", "probability": 0.9}
    assert result.response.usage is not None and result.response.usage.input_tokens == 1000
    assert result.response.model.endswith("versioned")


def test_missing_original_charge_never_uses_static_estimate() -> None:
    raw = _raw()
    cast(JsonObject, raw["usage"]).pop("cost")
    with pytest.raises(ToolProviderError) as caught:
        _decision_result(_request(), raw, snapshot=_snapshot())
    assert caught.value.billing is not None
    assert caught.value.billing.actual_cost_usd is None
    assert caught.value.billing.reference_cost_unavailable


def test_promotions_use_the_maximum_provider_discount_and_keep_actual_charge() -> None:
    snapshot = _snapshot()
    endpoint = snapshot.endpoints[0]
    snapshot = snapshot.__class__(
        snapshot.model,
        (
            endpoint.model_copy(update={"pricing": endpoint.pricing.model_copy(update={"discount": Decimal("0.25")})}),
            endpoint.model_copy(update={"pricing": endpoint.pricing.model_copy(update={"discount": Decimal("0.5")})}),
        ),
        snapshot.fetched_at,
    )
    result = _decision_result(_request(), _raw(), snapshot=snapshot)
    assert result.actual_cost_usd == 0.002
    assert result.reference_cost_usd == 0.004
    assert result.actual_cost_evidence["maximum_discount"] == 0.5
    assert result.billing.usage == result.response.usage


def test_missing_serving_provider_preserves_actual_charge_and_blocks_reference_comparison() -> None:
    raw = _raw()
    raw.pop("provider")
    with pytest.raises(ToolProviderError) as caught:
        _decision_result(_request(), raw, snapshot=_snapshot())
    assert caught.value.billing is not None
    assert caught.value.billing.actual_cost_usd == 0.002
    assert caught.value.billing.reference_cost_unavailable


def test_byok_separate_router_charge_is_not_discount_normalized() -> None:
    raw = _raw()
    raw["usage"] = {
        "input_tokens": 10,
        "output_tokens": 0,
        "cost": 0.001,
        "is_byok": True,
        "cost_details": {"upstream_inference_cost": 0.004},
    }
    snapshot = _snapshot()
    endpoint = snapshot.endpoints[0]
    snapshot = snapshot.__class__(
        snapshot.model,
        (endpoint.model_copy(update={"pricing": endpoint.pricing.model_copy(update={"discount": Decimal("0.5")})}),),
        snapshot.fetched_at,
    )
    result = _decision_result(_request(), raw, snapshot=snapshot)
    assert result.actual_cost_usd == 0.005
    assert result.reference_cost_usd == 0.009


@pytest.mark.parametrize("unpriceable", [False, True])
@pytest.mark.parametrize(
    "discount,byok,charge,expected_actual,expected_reference",
    [
        (0.5, False, 0.002, 0.002, 0.004),
        (0, False, 0.014, 0.014, 0.014),
        (0, True, 0, 0.004, 0.004),
        (0, True, 0.001, 0.005, 0.005),
    ],
)
async def test_dynamic_decision_route_binds_filtered_catalog_before_one_paid_dispatch(
    unpriceable: bool, discount: float, byok: bool, charge: float, expected_actual: float, expected_reference: float
) -> None:
    from harnyx_commons.llm.providers.openrouter import OpenRouterDecisionClient
    from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingCatalog
    from harnyx_commons.tools.invocation_clients import OpenRouterDecisionProvider

    posts = 0
    requested_model = "new/decision-20261021"

    def serve(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "POST":
            posts += 1
            raw = _raw()
            cast(JsonObject, raw["usage"]).update(
                cost=charge, is_byok=byok, cost_details={"upstream_inference_cost": 0.004}
            )
            return httpx.Response(200, json={**raw, "model": requested_model})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "new/decision",
                        "endpoints": [
                            {
                                "provider_name": "Cloudflare",
                                "tag": "cloudflare",
                                "model_id": "new/decision",
                                "pricing": {
                                    "prompt": "0.00001",
                                    "completion": "0",
                                    "discount": 1 if unpriceable else discount,
                                    **(
                                        {
                                            "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.00002"}],
                                            "request": "0.001",
                                            "web_search": "0.01",
                                        }
                                        if discount == 0
                                        else {}
                                    ),
                                },
                            }
                        ],
                    }
                },
            )
        assert request.url.params["output_modalities"] == "decisions"
        return httpx.Response(200, json={"data": [{"id": "new/decision", "canonical_slug": requested_model}]})

    transport = httpx.MockTransport(serve)
    provider = OpenRouterDecisionProvider(
        api_key=SecretStr("test"), timeout_seconds=1, pricing_catalog=OpenRouterPricingCatalog(transport=transport)
    )
    async with httpx.AsyncClient(transport=transport, base_url="https://openrouter.ai/api/alpha/") as client:
        provider._clients[requested_model] = OpenRouterDecisionClient(
            model=requested_model, api_key=SecretStr("test"), client=client
        )
        try:
            request = _request().model_copy(update={"model": requested_model})
            if unpriceable:
                with pytest.raises(ToolProviderError) as caught:
                    await provider.query(request)
                assert caught.value.billing is not None
                assert caught.value.billing.actual_cost_usd == caught.value.billing.reference_cost_usd == 0
                assert caught.value.billing.provider_usage["inference"] == "not_dispatched"
                assert posts == 0
            else:
                result = await provider.query(request)
                assert posts == 1
                assert result.actual_cost_usd == pytest.approx(expected_actual)
                assert result.reference_cost_usd == pytest.approx(expected_reference)
        finally:
            await provider.aclose()


@pytest.mark.parametrize(
    "bad_answer",
    [
        {"type": "unknown"},
        {"type": "choice"},
        {"type": "noul", "noul": 2.0},
    ],
)
def test_schema_invalid_answers_keep_independently_valid_charge(bad_answer: JsonObject) -> None:
    raw = _raw()
    cast(JsonObject, raw["answers"])["b"] = bad_answer
    with pytest.raises(ToolProviderError) as caught:
        _decision_result(_request(), raw, snapshot=_snapshot())
    assert caught.value.billing is not None
    assert caught.value.billing.actual_cost_usd == 0.002
    assert caught.value.billing.usage is not None
    assert caught.value.billing.usage.model_dump() == {"input_tokens": 1000, "output_tokens": 10}


@pytest.mark.parametrize("bad_cost", [-1, float("nan"), True, "not a cost"])
def test_invalid_cost_never_becomes_free_or_static_success(bad_cost: object) -> None:
    raw = _raw()
    cast(dict, raw["usage"])["cost"] = bad_cost
    with pytest.raises(ToolProviderError):
        _decision_result(_request(), raw, snapshot=_snapshot())


def test_gateway_normalizes_camel_case_usage_string_cost_and_metadata() -> None:
    request = _request().model_copy(update={"provider": "ai_gateway"})
    raw = copy.deepcopy(_raw())
    raw["usage"] = {"inputTokens": 12, "outputTokens": 3}
    raw["providerMetadata"] = {"gateway": {"cost": "0.002"}}
    cast(JsonObject, raw["answers"])["b"] = {"type": "boolean", "probability": 0.9}
    result = _decision_result(request, raw)
    assert result.actual_cost_usd == 0.002
    assert result.response.usage is not None and result.response.usage.input_tokens == 12
    assert result.response.provider_metadata == {"gateway": {"cost": "0.002"}}


async def test_registry_reuses_adapters_and_requires_matching_key() -> None:
    registry = CachedDecisionProviderRegistry(llm_settings=LlmSettings(openrouter_api_key=SecretStr("test-key")))
    assert registry.resolve("openrouter") is registry.resolve("openrouter")
    with pytest.raises(ProviderCredentialUnavailableError):
        registry.resolve("ai_gateway")
    await registry.aclose()


@pytest.mark.parametrize("provider", ["openrouter", "ai_gateway"])
@pytest.mark.parametrize("tokens", [100, -1, True, "100"])
def test_billed_invalid_answers_retain_only_valid_native_usage(provider, tokens) -> None:
    request = _request().model_copy(update={"provider": provider})
    raw = _raw()
    raw["answers"] = {}
    if provider == "openrouter":
        raw["usage"] = {"input_tokens": tokens, "output_tokens": 0, "cost": 0.002}
    else:
        raw["usage"] = {"inputTokens": tokens, "outputTokens": 0}
        raw["providerMetadata"] = {"gateway": {"cost": "0.002"}}
    with pytest.raises(ToolProviderError) as caught:
        _decision_result(request, raw)
    assert caught.value.billing is not None
    assert caught.value.billing.actual_cost_usd == 0.002
    usage = caught.value.billing.usage
    if type(tokens) is int and tokens >= 0:
        assert usage is not None
        assert usage.model_dump() == {"input_tokens": 100, "output_tokens": 0}
    else:
        assert usage is None
