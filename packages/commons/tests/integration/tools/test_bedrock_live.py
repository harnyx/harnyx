from __future__ import annotations

from collections.abc import Mapping

import pytest

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.providers.bedrock import BedrockLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]

_LIVE_READ_TIMEOUT_SECONDS = 600.0


def _provider() -> LlmProviderAdapter:
    settings = BedrockSettings()
    return LlmProviderAdapter(
        provider_name="bedrock",
        delegate=BedrockLlmProvider(
            region=settings.region_value,
            connect_timeout_seconds=settings.connect_timeout_seconds,
            read_timeout_seconds=_LIVE_READ_TIMEOUT_SECONDS,
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


async def test_bedrock_openai_tee_reasoning_effort_live() -> None:
    provider = _provider()
    request = LlmRequest(
        provider="bedrock",
        model="openai/gpt-oss-20b-TEE",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Think briefly, then reply with only "56".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=256,
        reasoning_effort="high",
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text, "Bedrock reasoning response should include text output"
    assert "56" in response.raw_text
    assert response.metadata is not None
    raw_events = dict(response.metadata)["raw_response"]["events"]
    assert raw_events
    assert response.choices[0].message.reasoning or _has_bedrock_reasoning_event(raw_events)


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


def _has_bedrock_reasoning_event(events: object) -> bool:
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, Mapping):
            continue
        content_block_delta = event.get("contentBlockDelta")
        if not isinstance(content_block_delta, Mapping):
            continue
        delta = content_block_delta.get("delta")
        if isinstance(delta, Mapping) and "reasoningContent" in delta:
            return True
    return False


async def test_bedrock_glm5_alias_live() -> None:
    provider = _provider()
    request = LlmRequest(
        provider="bedrock",
        model="zai-org/GLM-5-TEE",
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
