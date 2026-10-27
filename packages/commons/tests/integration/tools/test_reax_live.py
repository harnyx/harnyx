"""Paid acceptance of REAX key-default choice, score and boolean provider contracts."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.decision_models import DecisionQueryRequest
from harnyx_commons.tools.invocation_clients import CachedDecisionProviderRegistry

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_reax_native_decisions_through_shared_registry() -> None:
    settings = LlmSettings()
    assert settings.reax_api_key.get_secret_value().strip(), "REAX_API_KEY must be configured"
    registry = CachedDecisionProviderRegistry(llm_settings=settings)
    try:
        result = await registry.resolve("reax").query(
            DecisionQueryRequest.model_validate(
                {
                    "provider": "reax",
                    "model": "s1-pro-3",
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
        assert result.response.model == "s1-pro-3"
        assert result.actual_cost_provider == "reax"
        assert result.response.provider_metadata["tier"] in {"eu", "global"}
        assert result.actual_cost_evidence["currency"] in {"EUR", "USD"}
        assert result.actual_cost_evidence["settlement_source"] in {"provider_returned", "published_usd_usage_rate"}
        if result.actual_cost_evidence["charge_nanos"] == 0:
            assert result.actual_cost_usd == 0
        assert result.reference_cost_usd is not None and result.reference_cost_usd > 0
        assert result.actual_cost_usd is not None and result.actual_cost_usd >= 0
        assert result.response.usage is not None
        assert result.native_response is not None
    finally:
        await registry.aclose()
