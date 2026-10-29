"""One representative paid rerank boundary, including returned-charge settlement."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.invocation_clients import OpenRouterRerankProvider
from harnyx_commons.tools.rerank_models import RerankRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_openrouter_rerank_native_contract_live() -> None:
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"
    provider = OpenRouterRerankProvider(api_key=settings.openrouter_api_key, timeout_seconds=120)
    try:
        result = await provider.rerank(
            RerankRequest(
                provider="openrouter",
                model="cohere/rerank-v3.5",
                query="What is the capital of France?",
                documents=[
                    "Berlin is the capital of Germany.",
                    "Paris is the capital of France.",
                    "Bananas are a fruit.",
                ],
                top_n=2,
            )
        )
        assert len(result.response.results) == 2
        assert len({item.index for item in result.response.results}) == 2
        assert result.billing.actual_cost_usd is not None
        assert result.billing.reference_cost_usd is not None
        assert result.billing.actual_cost_provider == "openrouter"
        assert result.billing.provider_usage["provider_request_id"]
    finally:
        await provider.aclose()
