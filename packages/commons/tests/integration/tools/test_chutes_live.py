"""Check chutes liveness with its cheapest listed chat model."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmThinkingConfig

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_chutes_liveness_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="chutes",
        api_key=settings.chutes_api_key,
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(
            LlmRequest(
                provider="chutes",
                model="google/gemma-4-31B-turbo-TEE",
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
