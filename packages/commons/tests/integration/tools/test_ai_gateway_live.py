from __future__ import annotations

import os

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


def _api_key() -> str:
    api_key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    assert api_key, "AI_GATEWAY_API_KEY must be configured"
    return api_key


def _request() -> LlmRequest:
    return LlmRequest(
        provider="ai_gateway",
        model="openai/gpt-oss-20b",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
        thinking=LlmThinkingConfig(enabled=False),
        timeout_seconds=180.0,
        extra={"providerOptions": {"gateway": {"only": ["groq"]}}},
    )


async def test_miner_paid_ai_gateway_groq_selection_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="ai_gateway",
        api_key=_api_key(),
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["actual_cost_provider"] == "ai_gateway"
    assert response.metadata["actual_cost_usd"] >= 0.0
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "provider_returned"
    raw_response = response.metadata["raw_response"]
    assert raw_response["providerMetadata"]["gateway"]["cost"]


async def test_miner_paid_ai_gateway_inkling_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="ai_gateway",
        api_key=_api_key(),
        llm_settings=settings,
    )
    request = LlmRequest(
        provider="ai_gateway",
        model="thinkingmachines/inkling",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=None,
        max_output_tokens=None,
        thinking=LlmThinkingConfig(enabled=True, effort="low"),
        timeout_seconds=180.0,
        extra=None,
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["actual_cost_provider"] == "ai_gateway"
    assert response.metadata["actual_cost_usd"] >= 0.0
    message = response.choices[0].message
    assert message.reasoning, "Inkling must expose a reasoning trace through AI Gateway"
    assert message.reasoning_details, "Inkling must expose reasoning details through AI Gateway"
    assert response.usage.reasoning_tokens is not None
    assert response.usage.reasoning_tokens > 0


async def test_ai_gateway_cerebras_gemma_reasoning_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="ai_gateway",
        api_key=_api_key(),
        llm_settings=settings,
    )
    request = LlmRequest(
        provider="ai_gateway",
        model="google/gemma-4-31b-it",
        messages=(
            LlmMessage(
                role="user",
                content=(
                    LlmMessageContentPart.input_text(
                        "Solve 37 * 48. Briefly explain the calculation, then state the final integer. "
                        "Do not use tools or external information, and do not omit the final integer."
                    ),
                ),
            ),
        ),
        temperature=None,
        max_output_tokens=None,
        thinking=LlmThinkingConfig(enabled=True, effort="medium"),
        timeout_seconds=180.0,
        extra={"providerOptions": {"gateway": {"only": ["cerebras"]}}},
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.metadata is not None
    routing = response.metadata["raw_response"]["providerMetadata"]["gateway"]["routing"]
    assert routing["resolvedProvider"] == "cerebras"
    assert routing["finalProvider"] == "cerebras"
    message = response.choices[0].message
    reasoning = message.reasoning
    assert reasoning, "Gemma 4 must expose a reasoning trace through AI Gateway on Cerebras"
    assert message.reasoning_details, "Gemma 4 must expose reasoning details through AI Gateway on Cerebras"
    assert response.usage.reasoning_tokens is not None
    assert response.usage.reasoning_tokens > 0


async def test_ai_gateway_two_turn_function_tool_loop_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="ai_gateway",
        api_key=_api_key(),
        llm_settings=settings,
    )
    tool = LlmTool(
        type="function",
        function={
            "name": "lookup_weather",
            "description": "Return weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    )
    user_message = LlmMessage(
        role="user",
        content=(LlmMessageContentPart.input_text("Use lookup_weather for Paris."),),
    )
    common = {
        "provider": "ai_gateway",
        "model": "openai/gpt-oss-20b",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "tools": (tool,),
        "thinking": LlmThinkingConfig(enabled=False),
        "timeout_seconds": 180.0,
        "extra": {"providerOptions": {"gateway": {"only": ["groq"]}}},
    }

    try:
        first = await provider.invoke(
            LlmRequest(
                messages=(user_message,),
                tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
                **common,
            )
        )
        calls = first.choices[0].message.tool_calls
        assert calls and calls[0].id
        tool_result_messages = tuple(
            LlmMessage(
                role="tool",
                content=(
                    LlmInputToolResultPart(
                        tool_call_id=call.id,
                        name=None,
                        output_json='{"temperature_c":19}',
                    ),
                ),
            )
            for call in calls
        )
        second = await provider.invoke(
            LlmRequest(
                messages=(
                    user_message,
                    first.choices[0].message.to_input_message(),
                    *tool_result_messages,
                ),
                tool_choice="none",
                **common,
            )
        )
    finally:
        await provider.aclose()

    assert second.raw_text
    assert second.metadata is not None
    assert second.metadata["actual_cost_provider"] == "ai_gateway"
