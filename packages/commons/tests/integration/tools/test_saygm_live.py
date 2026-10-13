"""Exercise SayGM streaming and provider-returned settlement over the live API."""

import json
import os
from dataclasses import replace

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.schema import (
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmThinkingConfig,
    LlmTool,
)

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_saygm_chat_completions_stream_and_settle_live() -> None:
    key = os.environ.get("SAYGM_API_KEY", "").strip()
    assert key, "SAYGM_API_KEY must be configured for the SayGM live contract"
    provider = build_miner_paid_llm_provider(provider="saygm", api_key=key, llm_settings=LlmSettings())
    try:
        response = await provider.invoke(
            LlmRequest(
                provider="saygm",
                model="gpt-oss-20b",
                temperature=None,
                max_output_tokens=None,
                messages=(
                    LlmMessage(
                        role="user", content=(LlmMessageContentPart.input_text("Reply with one short greeting."),)
                    ),
                ),
                timeout=120.0,
            )
        )
    finally:
        await provider.aclose()
    assert response.raw_text
    assert response.metadata["actual_cost_provider"] == "saygm"
    assert response.metadata["actual_cost_usd"] >= 0
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "provider_returned"


async def test_saygm_thinking_and_function_tool_replay_live() -> None:
    key = os.environ.get("SAYGM_API_KEY", "").strip()
    assert key, "SAYGM_API_KEY must be configured for the SayGM live contract"
    provider = build_miner_paid_llm_provider(provider="saygm", api_key=key, llm_settings=LlmSettings())
    request = LlmRequest(
        provider="saygm",
        model="gpt-oss-20b",
        temperature=None,
        max_output_tokens=None,
        thinking=LlmThinkingConfig(enabled=True, effort="low"),
        messages=(
            LlmMessage(
                role="user",
                content=(
                    LlmMessageContentPart.input_text(
                        "Look up the temperature in Seoul using lookup_weather, then answer in one short sentence."
                    ),
                ),
            ),
        ),
        tools=(
            LlmTool(
                type="function",
                function={
                    "name": "lookup_weather",
                    "description": "Look up a city's current temperature.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            ),
        ),
        extra={
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "parallel_tool_calls": False,
        },
        timeout=120.0,
    )
    try:
        first = await provider.invoke(request)
        assistant = first.choices[0].message
        assert assistant.tool_calls is not None
        assert len(assistant.tool_calls) == 1
        call = assistant.tool_calls[0]
        assert call.name == "lookup_weather"
        assert isinstance(json.loads(call.arguments)["city"], str)
        final = await provider.invoke(
            replace(
                request,
                messages=(
                    *request.messages,
                    assistant.to_input_message(),
                    LlmMessage(
                        role="tool",
                        content=(
                            LlmInputToolResultPart(
                                tool_call_id=call.id, name=call.name, output_json='{"temperature_c":19}'
                            ),
                        ),
                    ),
                ),
                extra={"tool_choice": "none", "parallel_tool_calls": False},
            )
        )
        assert final.raw_text
        for response in (first, final):
            assert response.metadata["actual_cost_provider"] == "saygm"
            assert response.metadata["actual_cost_usd"] >= 0
    finally:
        await provider.aclose()
