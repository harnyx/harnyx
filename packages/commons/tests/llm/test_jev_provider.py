"""Protect shared JEV routing, native decision contracts and independent billing."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.errors import ProviderCredentialUnavailableError
from harnyx_commons.llm.providers.jev import JevDecisionError, JevDecisionProvider
from harnyx_commons.tools import invocation_clients
from harnyx_commons.tools.decision_models import DecisionQueryRequest

pytestmark = pytest.mark.anyio("asyncio")


def _request() -> DecisionQueryRequest:
    return DecisionQueryRequest.model_validate(
        {
            "provider": "jev",
            "model": "jev-1.13.0",
            "state": {"color": "blue"},
            "timeout": 5,
            "questions": {
                "color": {"type": "choice", "instructions": "Color?", "criteria": {"blue": "Blue", "red": "Red"}},
                "blue": {"type": "boolean", "instructions": "Is the color blue?"},
                "score": {"type": "score", "instructions": "Rate", "criteria": ["Low", "High"]},
            },
        }
    )


def _response() -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "color": {
                "type": "choice",
                "choice": "blue",
                "probabilities": {"blue": 0.8, "red": 0.2},
                "confidence": 0.8,
            },
            "blue": {"type": "noul", "noul": 0.8},
            "score": {
                "type": "score",
                "score": 0.8,
                "probabilities": {"0": 0.2, "1": 0.8},
                "confidence": 0.8,
                "legend": {"0": "Low", "1": "High"},
            },
        },
        "usage": {"input_tokens": 1000, "output_tokens": 250},
    }


async def test_registry_and_miner_builder_isolate_keys_and_preserve_native_decisions(monkeypatch) -> None:
    """Miner calls must never borrow the platform key, and all typed answers must survive normalization."""
    keys = []

    def respond(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers["Authorization"])
        assert request.url == "https://api.typesafe.ai/v1/systemone"
        payload = json.loads(request.content)
        assert set(payload) == {"model", "state", "questions"}
        assert payload["questions"]["blue"]["type"] == "noul"
        assert payload["state"] == {"color": "blue"}
        assert request.extensions["timeout"]["read"] == 5
        return httpx.Response(200, json=_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(
            invocation_clients,
            "JevDecisionProvider",
            lambda **kwargs: JevDecisionProvider(**kwargs, client=http),
        )
        settings = LlmSettings(_env_file=None, TYPESAFE_API_KEY="platform-key", OPENROUTER_API_KEY="unrelated")
        registry = invocation_clients.CachedDecisionProviderRegistry(llm_settings=settings)
        platform = registry.resolve("jev")
        assert registry.resolve("jev") is platform
        miner = invocation_clients.build_miner_paid_decision_provider(
            provider="jev",
            api_key="miner-key",
            llm_settings=settings,
        )
        try:
            for provider in (platform, miner):
                result = await provider.query(_request())
                assert result.response.answers["blue"].model_dump() == {"type": "boolean", "probability": 0.8}
                assert result.response.answers["color"].choice == "blue"
                assert result.response.answers["score"].score == 0.8
                assert result.actual_cost_usd == result.reference_cost_usd == pytest.approx(0.000042)
                assert result.actual_cost_evidence["pricing_origin"] == "typesafe_published_rates"
                assert result.native_response.status_code == 200
                assert json.loads(result.native_response.body) == _response()
        finally:
            await registry.aclose()
            await miner.aclose()
    assert keys == ["Bearer platform-key", "Bearer miner-key"]
    missing = invocation_clients.CachedDecisionProviderRegistry(
        llm_settings=LlmSettings(_env_file=None, TYPESAFE_API_KEY="", OPENROUTER_API_KEY="unrelated")
    )
    with pytest.raises(ProviderCredentialUnavailableError):
        missing.resolve("jev")


@pytest.mark.parametrize(
    "failure", ["invalid_choice", "missing_confidence", "invalid_usage", "missing_question", "http"]
)
async def test_invalid_receipts_preserve_only_valid_billing_and_never_retry(failure: str) -> None:
    """A provider error cannot hide known billable usage or publish an incomplete decision."""
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = _response()
        if failure == "invalid_choice":
            body["answers"]["color"]["choice"] = "green"
        elif failure == "missing_confidence":
            body["answers"]["color"].pop("confidence")
        elif failure == "invalid_usage":
            body["usage"]["input_tokens"] = True
        elif failure == "missing_question":
            body["answers"].pop("score")
        return httpx.Response(429 if failure == "http" else 200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = JevDecisionProvider(api_key=SecretStr("key"), client=http)
        with pytest.raises(JevDecisionError) as caught:
            await provider.query(_request())
    assert calls == 1
    assert caught.value.native_response is not None
    billing = caught.value.billing
    if failure == "invalid_usage":
        assert billing is None
    else:
        assert billing.actual_cost_usd == billing.reference_cost_usd == pytest.approx(0.000042)
        assert billing.usage.input_tokens == 1000
