"""Check openrouter liveness with its cheapest listed chat model."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_openrouter_liveness_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="openrouter",
        api_key=settings.openrouter_api_key,
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(
            LlmRequest(
                provider="openrouter",
                model="openai/gpt-oss-20b",
                messages=(
                    LlmMessage(
                        role="user",
                        content=(LlmMessageContentPart.input_text("Say hello."),),
                    ),
                ),
                temperature=None,
                max_output_tokens=128,
                timeout=60.0,
            )
        )
    finally:
        await provider.aclose()

    assert response.raw_text.strip()
