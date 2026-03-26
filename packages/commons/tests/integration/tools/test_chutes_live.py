from __future__ import annotations

import pytest
from pydantic import BaseModel

from harnyx_commons.clients import CHUTES
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.json_utils import pydantic_postprocessor
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]

REASONING_MODEL = "tngtech/DeepSeek-TNG-R1T2-Chimera-TEE"


class JsonObjectAnswer(BaseModel):
    ping: str
    count: int


class ThrowawayStructuredAnswer(BaseModel):
    animal: str
    count: int
    approved: bool


def _provider_settings() -> tuple[str, str, float]:
    settings = LlmSettings()
    api_key = settings.chutes_api_key_value
    assert api_key, "CHUTES_API_KEY must be configured"
    model = "openai/gpt-oss-120b-TEE"
    timeout = float(CHUTES.timeout_seconds)
    return api_key, model, timeout


async def test_chutes_json_object_live() -> None:
    api_key, model, timeout = _provider_settings()
    provider = ChutesLlmProvider(
        base_url=CHUTES.base_url,
        api_key=api_key,
        timeout=timeout,
    )
    request = LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(
                    LlmMessageContentPart.input_text(
                        'Return JSON only with {"ping":"pong","count":2}.'
                    ),
                ),
            ),
        ),
        temperature=0.0,
        max_output_tokens=128,
        output_mode="json_object",
        postprocessor=pydantic_postprocessor(JsonObjectAnswer),
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    parsed = response.postprocessed
    assert isinstance(parsed, JsonObjectAnswer)
    assert parsed.ping == "pong"
    assert parsed.count == 2


async def test_chutes_json_schema_live_with_throwaway_schema() -> None:
    api_key, model, timeout = _provider_settings()
    provider = ChutesLlmProvider(
        base_url=CHUTES.base_url,
        api_key=api_key,
        timeout=timeout,
    )
    request = LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(
                    LlmMessageContentPart.input_text(
                        'Return JSON only with {"animal":"otter","count":3,"approved":true}.'
                    ),
                ),
            ),
        ),
        temperature=0.0,
        max_output_tokens=128,
        output_mode="structured",
        output_schema=ThrowawayStructuredAnswer,
        postprocessor=pydantic_postprocessor(ThrowawayStructuredAnswer),
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    parsed = response.postprocessed
    assert isinstance(parsed, ThrowawayStructuredAnswer)
    assert parsed.animal == "otter"
    assert parsed.count == 3
    assert parsed.approved is True


async def test_chutes_reasoning_live_normalizes_string_reasoning_payload() -> None:
    api_key, _, timeout = _provider_settings()
    provider = ChutesLlmProvider(
        base_url=CHUTES.base_url,
        api_key=api_key,
        timeout=timeout,
    )
    request = LlmRequest(
        provider="chutes",
        model=REASONING_MODEL,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('What is 7 times 8? Reply with only "56".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=32,
        timeout_seconds=180.0,
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    reasoning = response.choices[0].message.reasoning
    assert isinstance(reasoning, str)
    assert reasoning.strip()
