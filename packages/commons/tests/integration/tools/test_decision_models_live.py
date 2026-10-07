"""Paid acceptance for the exact requested OpenRouter decision models."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.decision_models import DecisionQueryRequest
from harnyx_commons.tools.invocation_clients import OpenRouterDecisionProvider

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


@pytest.mark.parametrize("model", ["cloudflare/clef-flash", "cloudflare/clef", "jaredpalmer/kev-4b"])
async def test_openrouter_decision_model_live(model: str) -> None:
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"
    provider = OpenRouterDecisionProvider(api_key=settings.openrouter_api_key, timeout_seconds=120)
    try:
        result = await provider.query(
            DecisionQueryRequest(
                provider="openrouter",
                model=model,
                state={"color": "red", "number": 2},
                questions={
                    "color": {
                        "type": "choice",
                        "instructions": "Choose the color",
                        "criteria": {"red": "Red", "blue": "Blue"},
                    },
                    "number": {
                        "type": "score",
                        "instructions": "Rate the number",
                        "criteria": ["Small", "Medium", "Large"],
                    },
                    "red": {"type": "boolean", "instructions": "Is the color red?"},
                },
            )
        )
        assert set(result.response.answers) == {"color", "number", "red"}
        assert result.actual_cost_usd is not None and result.actual_cost_usd >= 0
        assert result.actual_cost_provider == "openrouter"
        assert result.response.usage is not None
    finally:
        await provider.aclose()
