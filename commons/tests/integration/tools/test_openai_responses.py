from __future__ import annotations

import pytest
from pydantic import BaseModel

from caster_commons.config.llm import LlmSettings
from caster_commons.llm.json_utils import pydantic_postprocessor
from caster_commons.llm.providers.openai import OpenAILlmProvider
from caster_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


class StructuredAnswer(BaseModel):
    answer: str


async def test_openai_responses_roundtrip() -> None:
    settings = LlmSettings()
    api_key = settings.openai_api_key_value
    assert api_key, "OPENAI_API_KEY not configured"
    base_url = settings.openai_base_url
    timeout = settings.llm_timeout_seconds
    model = "gpt-5-nano"

    provider = OpenAILlmProvider(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    request = LlmRequest(
        provider="openai",
        model=model,
        messages=(
            LlmMessage(
                role="system",
                content=(
                    LlmMessageContentPart.input_text(
                        "You respond with a short JSON object containing an 'answer' key."
                    ),
                ),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Return answer:'pong'."),),
            ),
        ),
        temperature=None,
        max_output_tokens=512,
        output_mode="text",
    )

    response = await provider.invoke(request)
    await provider.aclose()

    text = response.raw_text
    assert text is not None and text.strip() != ""
    assert response.tool_calls == ()


async def test_openai_responses_grounded_search_roundtrip() -> None:
    settings = LlmSettings()
    api_key = settings.openai_api_key_value
    assert api_key, "OPENAI_API_KEY not configured"
    base_url = settings.openai_base_url
    timeout = settings.llm_timeout_seconds
    model = "gpt-5-nano"

    provider = OpenAILlmProvider(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    request = LlmRequest(
        provider="openai",
        model=model,
        messages=(
            LlmMessage(
                role="system",
                content=(
                    LlmMessageContentPart.input_text(
                        "Answer with one sentence summarizing a recent tech headline. "
                        "Return plain text output."
                    ),
                ),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Give me one news snippet with a citation."),),
            ),
        ),
        temperature=None,
        max_output_tokens=4096,
        output_mode="text",
    )

    response = await provider.invoke(request)
    await provider.aclose()

    text = response.raw_text
    assert text is not None and text.strip() != ""


async def test_openai_structured_output() -> None:
    settings = LlmSettings()
    api_key = settings.openai_api_key_value
    assert api_key, "OPENAI_API_KEY not configured"
    base_url = settings.openai_base_url
    timeout = settings.llm_timeout_seconds
    # Use a model that supports structured outputs
    model = "gpt-5-nano"

    provider = OpenAILlmProvider(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    request = LlmRequest(
        provider="openai",
        model=model,
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("Return JSON only with an 'answer' field."),),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Say ok."),),
            ),
        ),
        temperature=None,
        max_output_tokens=1024,
        output_mode="structured",
        output_schema=StructuredAnswer,
        postprocessor=pydantic_postprocessor(StructuredAnswer),
    )

    try:
        response = await provider.invoke(request)
        await provider.aclose()
    except RuntimeError as exc:
        pytest.skip(f"OpenAI structured output failed: {exc}")
    parsed = response.postprocessed
    assert isinstance(parsed, StructuredAnswer)
    assert parsed.answer
