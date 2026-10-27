"""Paid acceptance of direct JEV choice, score and boolean provider contracts."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.decision_models import DecisionQueryRequest
from harnyx_commons.tools.invocation_clients import CachedDecisionProviderRegistry

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_jev_native_decisions_through_shared_registry() -> None:
    settings = LlmSettings()
    assert settings.typesafe_api_key_value, "TYPESAFE_API_KEY must be configured"
    registry = CachedDecisionProviderRegistry(llm_settings=settings)
    try:
        result = await registry.resolve("jev").query(
            DecisionQueryRequest.model_validate(
                {
                    "provider": "jev",
                    "model": "jev-1.13.0",
                    "state": {"color": "blue", "quantity": 2},
                    "questions": {
                        "color": {
                            "type": "choice",
                            "instructions": "Choose the color",
                            "criteria": {"blue": "Blue", "red": "Red"},
                        },
                        "quantity": {
                            "type": "score",
                            "instructions": "Rate quantity",
                            "criteria": ["Small", "Medium", "Large"],
                        },
                        "blue": {"type": "boolean", "instructions": "Is the color blue?"},
                    },
                }
            )
        )
        assert set(result.response.answers) == {"color", "quantity", "blue"}
        assert result.response.model == "jev-1.13.0"
        assert result.actual_cost_provider == "jev"
        assert result.actual_cost_usd == result.reference_cost_usd
        assert result.actual_cost_usd is not None and result.actual_cost_usd >= 0
        assert result.response.usage is not None
        assert result.native_response is not None
    finally:
        await registry.aclose()
