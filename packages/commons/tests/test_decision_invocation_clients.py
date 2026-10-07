from __future__ import annotations

import copy
from typing import cast

import pytest
from pydantic import SecretStr

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.errors import ProviderCredentialUnavailableError, ToolProviderError
from harnyx_commons.json_types import JsonObject
from harnyx_commons.tools.decision_models import DecisionQueryRequest
from harnyx_commons.tools.invocation_clients import CachedDecisionProviderRegistry, _decision_result

pytestmark = pytest.mark.anyio("asyncio")


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
    result = _decision_result(_request(), _raw())
    assert result.actual_cost_usd == 0.002
    assert result.response.answers["b"].model_dump() == {"type": "boolean", "probability": 0.9}
    assert result.response.usage is not None and result.response.usage.input_tokens == 1000
    assert result.response.model.endswith("versioned")


def test_absent_cost_requires_known_card_and_complete_token_evidence() -> None:
    raw = _raw()
    usage = cast(JsonObject, raw["usage"])
    usage.pop("cost")
    assert _decision_result(_request(), raw).actual_cost_usd == pytest.approx(0.00009)
    usage.pop("input_tokens")
    with pytest.raises(ToolProviderError):
        _decision_result(_request(), raw)


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
        _decision_result(_request(), raw)
    assert caught.value.billing is not None
    assert caught.value.billing.actual_cost_usd == 0.002
    assert caught.value.billing.usage is not None
    assert caught.value.billing.usage.model_dump() == {"input_tokens": 1000, "output_tokens": 10}


@pytest.mark.parametrize("bad_cost", [-1, float("nan"), True, "not a cost"])
def test_invalid_cost_never_becomes_free_or_static_success(bad_cost: object) -> None:
    raw = _raw()
    cast(dict, raw["usage"])["cost"] = bad_cost
    with pytest.raises(ToolProviderError):
        _decision_result(_request(), raw)


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
