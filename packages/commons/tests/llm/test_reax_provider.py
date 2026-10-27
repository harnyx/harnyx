"""Protect key-default routing and financial evidence independent of answer validity."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.errors import ProviderCredentialUnavailableError, ToolProviderError
from harnyx_commons.llm.providers.reax import ReaxDecisionProvider
from harnyx_commons.tools import invocation_clients
from harnyx_commons.tools.decision_models import DecisionQueryRequest

pytestmark = pytest.mark.anyio("asyncio")


def _request(model: str = "s1-pro-3") -> DecisionQueryRequest:
    questions = {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"red": None, "blue": None}},
        "red": {"type": "boolean", "instructions": "Is it red?"},
        "score": {"type": "score", "instructions": "Support?", "criteria": ["Low", "High"]},
    }
    return DecisionQueryRequest.model_validate(
        {
            "provider": "reax",
            "model": model,
            "state": {"color": "red"},
            "timeout": 5.0,
            "questions": questions if model == "s1-pro-3" else {"color": questions["color"]},
        }
    )


def _response(model: str = "s1-pro-3", tier: str = "eu") -> dict:
    answers = {
        "color": {"type": "choice", "choice": "red", "confidence": 0.8, "probabilities": {"red": 0.8, "blue": 0.2}},
        "red": {"type": "noul", "noul": 0.8},
        "score": {
            "type": "score",
            "score": 0.8,
            "confidence": 0.8,
            "legend": {"0": "Low", "1": "High"},
            "probabilities": {"0": 0.2, "1": 0.8},
        },
    }
    return {
        "id": "dec_test",
        "model": model,
        "model_version": model,
        "tier": tier,
        "answers": answers if model == "s1-pro-3" else {"color": answers["color"]},
        "usage": {"input_tokens": 1000, "output_tokens": 0, "decisions": 3 if model == "s1-pro-3" else 1},
    }


def _headers(currency: str = "EUR", charge: str = "1000", tier: str = "eu") -> dict[str, str]:
    return {
        "S1-Charge-Nanos": charge,
        "S1-Currency": currency,
        "S1-Region": tier,
        "S1-Execution-Region": "eu-fin",
        "S1-Request-Id": "dec_test",
    }


@pytest.mark.parametrize("model", ["s1-fast-2", "s1-pro-3"])
@pytest.mark.parametrize("tier", ["eu", "global"])
async def test_native_decisions_use_key_default_and_effective_tier_for_eur_accounting(model, tier) -> None:
    """EUR invoice amounts must not be misrepresented as USD or priced for another tier."""

    def respond(request: httpx.Request) -> httpx.Response:
        assert "S1-Region" not in request.headers
        assert request.extensions["timeout"]["read"] == 5.0
        payload = json.loads(request.content)
        assert payload["model"] == model and payload["state"] == {"color": "red"}
        if model == "s1-pro-3":
            assert payload["questions"]["red"]["type"] == "noul"
        return httpx.Response(200, json=_response(model, tier), headers=_headers(tier=tier))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        result = await ReaxDecisionProvider(api_key=SecretStr("key"), client=http).query(_request(model))
        expected = {
            ("s1-fast-2", "eu"): 0.021,
            ("s1-fast-2", "global"): 0.019,
            ("s1-pro-3", "eu"): 0.084,
            ("s1-pro-3", "global"): 0.080,
        }[model, tier] / 1000
        assert result.actual_cost_usd == result.reference_cost_usd == pytest.approx(expected)
        assert result.actual_cost_evidence["settlement_source"] == "published_usd_usage_rate"
        assert result.actual_cost_evidence["currency"] == "EUR"
        assert result.actual_cost_evidence["charge_nanos"] == 1000
        assert result.response.provider_metadata["tier"] == tier
        if model == "s1-pro-3":
            assert result.response.answers["red"].probability == 0.8


@pytest.mark.parametrize("currency,charge,actual", [("USD", "1900", 0.0000019), ("EUR", "0", 0.0)])
@pytest.mark.parametrize("invalid_usage", [False, True])
async def test_valid_receipt_survives_invalid_answers_and_usage(currency, charge, actual, invalid_usage) -> None:
    """A malformed answer or token counter cannot erase independently known charges."""
    body = _response()
    body["answers"]["color"]["choice"] = "green"
    if invalid_usage:
        body["usage"]["input_tokens"] = True
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body, headers=_headers(currency, charge)))
    ) as http:
        with pytest.raises(ToolProviderError) as caught:
            await ReaxDecisionProvider(api_key=SecretStr("key"), client=http).query(_request())
    assert caught.value.billing.actual_cost_usd == pytest.approx(actual)
    assert (caught.value.billing.usage is None) == invalid_usage


@pytest.mark.parametrize("failure", ["missing", "negative", "invalid", "unsupported", "tier_conflict", "usage"])
async def test_unsettled_response_is_rejected_with_original_evidence(failure) -> None:
    """A usable-looking answer must not bypass the settled-cost gate."""
    headers = _headers()
    body = _response()
    if failure == "missing":
        headers.pop("S1-Charge-Nanos")
    if failure == "negative":
        headers["S1-Charge-Nanos"] = "-1"
    if failure == "invalid":
        headers["S1-Charge-Nanos"] = "1.2"
    if failure == "unsupported":
        headers["S1-Currency"] = "GBP"
    if failure == "tier_conflict":
        headers["S1-Region"] = "global"
    if failure == "usage":
        body["usage"]["input_tokens"] = True
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body, headers=headers))
    ) as http:
        with pytest.raises(ToolProviderError) as caught:
            await ReaxDecisionProvider(api_key=SecretStr("key"), client=http).query(_request())
    assert caught.value.billing.actual_cost_usd is None
    assert caught.value.billing.provider_usage["headers"]


@pytest.mark.parametrize("status", [402, 403, 429, 503])
async def test_upstream_failure_never_retries_or_switches_tier(status) -> None:
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "private input"}}, headers=_headers("USD", "1900"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(ToolProviderError) as caught:
            await ReaxDecisionProvider(api_key=SecretStr("key"), client=http).query(_request())
    assert len(calls) == 1 and "S1-Region" not in calls[0].headers
    assert "private input" not in str(caught.value)
    assert caught.value.billing.actual_cost_usd == pytest.approx(0.0000019)


@pytest.mark.parametrize("currency,actual", [("USD", 0.0000019), ("EUR", None)])
async def test_unrepresentable_reference_price_does_not_discard_independent_receipt(currency, actual) -> None:
    """Malformed usage must not erase USD charges or fabricate EUR accounting."""
    body = _response()
    body["usage"]["input_tokens"] = 10**309
    body["answers"]["color"]["choice"] = "green"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body, headers=_headers(currency, "1900")))
    ) as http:
        with pytest.raises(ToolProviderError) as caught:
            await ReaxDecisionProvider(api_key=SecretStr("key"), client=http).query(_request())
    assert caught.value.billing.actual_cost_usd == actual
    assert caught.value.billing.reference_cost_usd is None
    assert caught.value.billing.reference_cost_unavailable
    assert caught.value.billing.provider_usage["charge_nanos"] == 1900
    assert caught.value.billing.provider_usage["currency"] == currency


@pytest.mark.parametrize("model,tokens", [("s1-fast-2", "8,000"), ("s1-pro-3", "8,192")])
@pytest.mark.parametrize(
    "status,code", [(400, "invalid_request"), (400, "request_limit_exceeded"), (413, "payload_too_large")]
)
async def test_limit_errors_explain_model_limits_without_exposing_vendor_text(model, tokens, status, code) -> None:
    """Miners can reduce rejected inputs without leaking echoed content or retrying paid calls."""
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={"error": {
                "code": code, "message": "private input and secret-key", "request_id": "private-id",
                "retryable": False, "details": [{"input": "private input"}],
            }},
            headers=_headers("USD", "1900"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(ToolProviderError) as caught:
            await ReaxDecisionProvider(api_key=SecretStr("secret-key"), client=http).query(_request(model))
    message = str(caught.value)
    assert code in message and tokens in message
    assert "private" not in message and "secret-key" not in message
    if status == 413:
        assert "serialized" in message
    assert len(calls) == 1 and "S1-Region" not in calls[0].headers
    assert caught.value.http_status == status
    assert caught.value.billing.actual_cost_usd == pytest.approx(0.0000019)


@pytest.mark.parametrize("error", [
    {"code": "private input", "message": "private input", "request_id": "private-id", "retryable": False},
    {"code": "invalid_request", "message": "private input"},
])
async def test_unknown_or_malformed_vendor_errors_remain_generic(error) -> None:
    """Only complete typed envelopes with allowlisted codes may influence public diagnostics."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(400, json={"error": error}))
    ) as http:
        with pytest.raises(ToolProviderError) as caught:
            await ReaxDecisionProvider(api_key=SecretStr("key"), client=http).query(_request())
    assert str(caught.value) == "REAX HTTP 400"


def test_invalid_selection_is_rejected_before_dispatch() -> None:
    payload = _request().model_dump()
    payload["model"] = "s1-fast-2"  # Fast cannot accept this bundle.
    with pytest.raises(ValidationError, match="question"):
        DecisionQueryRequest.model_validate(payload)


async def test_registry_and_miner_keys_are_isolated(monkeypatch) -> None:
    keys = []

    def respond(request):
        keys.append(request.headers["Authorization"])
        return httpx.Response(200, json=_response(), headers=_headers())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(
            invocation_clients, "ReaxDecisionProvider", lambda **kwargs: ReaxDecisionProvider(**kwargs, client=http)
        )
        settings = LlmSettings(_env_file=None, REAX_API_KEY="platform-key")
        registry = invocation_clients.CachedDecisionProviderRegistry(llm_settings=settings)
        miner = invocation_clients.build_miner_paid_decision_provider(
            provider="reax", api_key="miner-key", llm_settings=settings
        )
        await registry.resolve("reax").query(_request())
        await miner.query(_request())
        await registry.aclose()
        await miner.aclose()
    assert keys == ["Bearer platform-key", "Bearer miner-key"]
    with pytest.raises(ProviderCredentialUnavailableError):
        invocation_clients.CachedDecisionProviderRegistry(
            llm_settings=LlmSettings(_env_file=None, REAX_API_KEY="")
        ).resolve("reax")
