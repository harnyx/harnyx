from __future__ import annotations

import pytest

from caster_commons.clients import CHUTES
from caster_commons.config.llm import LlmSettings
from caster_commons.llm.providers.chutes import ChutesLlmProvider
from caster_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

settings = LlmSettings()

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]

async def test_chutes_utilization_live() -> None:
    api_key = settings.chutes_api_key_value
    assert api_key, "CHUTES_API_KEY must be configured"
    base_url = CHUTES.base_url
    timeout = float(CHUTES.timeout_seconds)

    provider = ChutesLlmProvider(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    request = LlmRequest(
        provider="chutes",
        model="Qwen/Qwen3Guard-Gen-0.6B",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=0.7,
        max_output_tokens=256,
    )

    response = await provider.invoke(request)
    await provider.aclose()

    assert response.raw_text, "Expected completion text from Chutes"
    assert response.usage.total_tokens is not None
