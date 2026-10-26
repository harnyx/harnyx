"""Paid native contract acceptance for Workers AI chat, decisions and extraction."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.providers.cloudflare import CloudflareDecisionProvider, CloudflareLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest
from harnyx_miner_sdk.tools.cloudflare import CloudflareCredential
from harnyx_miner_sdk.tools.decision_models import DecisionQueryRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


def _credential() -> CloudflareCredential:
    settings = LlmSettings()
    assert settings.cloudflare_account_id and settings.cloudflare_api_token.get_secret_value(), (
        "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be configured"
    )
    return CloudflareCredential.model_validate({
        "account_id": settings.cloudflare_account_id, "api_token": settings.cloudflare_api_token,
    })


async def test_cloudflare_chat_native_contract() -> None:
    provider = CloudflareLlmProvider(credential=_credential())
    try:
        response = await provider.invoke(LlmRequest(
            provider="cloudflare", model="@cf/openai/gpt-oss-20b", temperature=None, max_output_tokens=None,
            messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Reply with only OK."),)),),
        ))
        assert response.raw_text.strip()
        assert response.metadata["actual_cost_provider"] == "cloudflare"
        assert response.metadata["actual_cost_usd"] >= 0
        assert response.metadata["reference_cost_usd"] == response.metadata["actual_cost_usd_total"]
    finally:
        await provider.aclose()


@pytest.mark.parametrize("capability", ["decision", "extraction"])
async def test_cloudflare_native_typed_contract(capability: str) -> None:
    questions = (
        {
            "invoice": {"type": "text", "instructions": "Invoice ID"},
            "total": {"type": "number", "instructions": "Invoice total", "decimals": 2},
            "due": {"type": "date", "instructions": "Due date"},
            "time": {"type": "time", "instructions": "Time of delivery"},
            "delivery": {"type": "datetime", "instructions": "Delivery date and time"},
            "duration": {"type": "duration", "unit": "days", "instructions": "Payment term length"},
            "contacts": {"type": "list", "items": "text", "instructions": "Contact names"},
            "estimate": {"type": "estimate", "min": 0, "max": 60, "step": 1, "instructions": "Days until paid"},
            "payment": {"type": "estimate", "min": "2026-11-01", "max": "2026-12-31", "step": "month",
                        "instructions": "Payment date"},
            "forecast": {"type": "forecast", "min": 0, "max": 120, "step": 1, "horizon": 2,
                         "interval": "week", "instructions": "Outstanding balance"},
        }
        if capability == "extraction" else {
            "paid": {"type": "boolean", "instructions": "Has the invoice been paid?"},
            "kind": {"type": "choice", "instructions": "Document kind", "criteria": {"invoice": None, "letter": None}},
            "amount": {"type": "score", "instructions": "Invoice size", "criteria": ["Small", "Large"]},
        }
    )
    model = "clairic" if capability == "extraction" else "clef-flash"
    provider = CloudflareDecisionProvider(credential=_credential())
    try:
        result = await provider.query(DecisionQueryRequest.model_validate({
            "provider": "cloudflare", "model": f"@cf/cloudflare/{model}",
            "state": "😀 Café. Invoice INV-42. Total USD 120.00. Not paid. Due 2026-11-01. Net 30 days. "
                     "Delivery 2026-10-22T09:00:00. Contacts: Sam, Alex.", "questions": questions,
        }))
        if capability == "extraction":
            answer = result.response.answers["invoice"]
            assert answer.value == "INV-42"
            assert answer.span == [16, 22]
        assert set(result.response.answers) == set(questions)
        assert result.response.usage.input_tokens > 0
        assert result.response.usage.output_tokens == 0
        assert result.actual_cost_usd >= 0
        assert result.reference_cost_usd == result.actual_cost_usd
    finally:
        await provider.aclose()
