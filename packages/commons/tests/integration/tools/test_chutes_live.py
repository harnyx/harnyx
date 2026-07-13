from __future__ import annotations

import pytest

from harnyx_commons.clients import CHUTES
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.schema import (
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmThinkingConfig,
    LlmTool,
)
from harnyx_commons.llm.tool_models import MINER_SELECTED_LLM_PROVIDER_MODELS

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]

CHUTES_TOOL_MODELS = MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]


def _provider_settings() -> tuple[str, float]:
    settings = LlmSettings()
    api_key = settings.chutes_api_key_value
    assert api_key, "CHUTES_API_KEY must be configured"
    timeout = float(CHUTES.timeout_seconds)
    return api_key, timeout


def _completion_request(*, model: str) -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=32,
        timeout_seconds=180.0,
    )


def _thinking_request(*, model: str, enabled: bool) -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Think briefly, then reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=96,
        timeout_seconds=180.0,
        thinking=LlmThinkingConfig(enabled=enabled),
    )


@pytest.mark.parametrize("model", CHUTES_TOOL_MODELS)
async def test_chutes_tool_model_completion_live(model: str) -> None:
    api_key, timeout = _provider_settings()
    provider = ChutesLlmProvider(
        base_url=CHUTES.base_url,
        api_key=api_key,
        timeout=timeout,
    )
    request = _completion_request(model=model)

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["actual_cost_usd"] >= 0.0
    assert response.metadata["actual_cost_provider"] == "chutes"
    assert response.metadata["actual_cost_evidence"]["settlement_source"] in {
        "cached_provider_pricing",
        "static_pricing",
    }
    assert response.metadata["actual_cost_evidence"]["pricing_origin"] in {
        "chutes_live_snapshot",
        "chutes_repo_rates",
    }


@pytest.mark.parametrize("model", CHUTES_TOOL_MODELS)
async def test_chutes_supported_model_reasoning_usage_live(model: str) -> None:
    api_key, timeout = _provider_settings()
    provider = ChutesLlmProvider(
        base_url=CHUTES.base_url,
        api_key=api_key,
        timeout=timeout,
    )
    request = _thinking_request(model=model, enabled=True)

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.choices[0].message.reasoning
    assert response.usage.completion_tokens > 0
    if response.usage.reasoning_tokens is None:
        assert response.usage.completion_tokens > 1
    else:
        assert response.usage.reasoning_tokens > 1


async def test_miner_paid_chutes_helper_completion_live() -> None:
    settings = LlmSettings()
    assert settings.chutes_api_key_value, "CHUTES_API_KEY must be configured"
    provider = build_miner_paid_llm_provider(
        provider="chutes",
        api_key=settings.chutes_api_key,
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(_completion_request(model="zai-org/GLM-5-TEE"))
    finally:
        await provider.aclose()

    assert response.raw_text


async def test_chutes_two_turn_function_tool_loop_live() -> None:
    api_key, timeout = _provider_settings()
    provider = ChutesLlmProvider(base_url=CHUTES.base_url, api_key=api_key, timeout=timeout)
    model = CHUTES_TOOL_MODELS[0]
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
    first_request = LlmRequest(
        provider="chutes",
        model=model,
        messages=(user_message,),
        temperature=0.0,
        max_output_tokens=128,
        tools=(tool,),
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
        timeout_seconds=180.0,
    )

    try:
        first = await provider.invoke(first_request)
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
                provider="chutes",
                model=model,
                messages=(
                    user_message,
                    first.choices[0].message.to_input_message(),
                    *tool_result_messages,
                ),
                temperature=0.0,
                max_output_tokens=128,
                tools=(tool,),
                tool_choice="none",
                timeout_seconds=180.0,
            )
        )
    finally:
        await provider.aclose()

    assert second.raw_text
