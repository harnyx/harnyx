from __future__ import annotations

import pytest

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.providers.bedrock import BedrockLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


def _provider() -> LlmProviderAdapter:
    settings = BedrockSettings()
    return LlmProviderAdapter(
        provider_name="bedrock",
        delegate=BedrockLlmProvider(
            region=settings.region_value,
            connect_timeout_seconds=settings.connect_timeout_seconds,
            read_timeout_seconds=settings.read_timeout_seconds,
        ),
    )


async def test_bedrock_openai_tee_alias_live() -> None:
    provider = _provider()
    request = LlmRequest(
        provider="bedrock",
        model="openai/gpt-oss-20b-TEE",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('What is 7 times 8? Reply with only "56".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text, "Bedrock response should include text output"
    assert "56" in response.raw_text
    assert response.metadata is not None
    assert dict(response.metadata)["raw_response"]["events"]


async def test_bedrock_kimi_alias_live() -> None:
    provider = _provider()
    request = LlmRequest(
        provider="bedrock",
        model="moonshotai/Kimi-K2.5-TEE",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('What is 7 times 8? Reply with only "56".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text, "Bedrock response should include text output"
    assert "56" in response.raw_text
    assert response.metadata is not None
    assert dict(response.metadata)["raw_response"]["events"]
