"""Check ai_gateway liveness with its cheapest listed chat model."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmThinkingConfig

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_ai_gateway_liveness_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="ai_gateway",
        api_key=settings.ai_gateway_api_key,
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(
            LlmRequest(
                provider="ai_gateway",
                model="openai/gpt-oss-20b",
                messages=(
                    LlmMessage(
                        role="user",
                        content=(LlmMessageContentPart.input_text("Say hello."),),
                    ),
                ),
                temperature=None,
                max_output_tokens=64,
                thinking=LlmThinkingConfig(enabled=False),
                timeout=60.0,
            )
        )
    finally:
        await provider.aclose()

    assert response.raw_text.strip()
