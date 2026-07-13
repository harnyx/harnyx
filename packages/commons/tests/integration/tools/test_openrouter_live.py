from __future__ import annotations

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.providers.openrouter import (
    OPENROUTER_INTERNAL_TO_NATIVE_MODEL,
    OpenRouterEmbeddingClient,
    OpenRouterLlmProvider,
)
from harnyx_commons.llm.schema import (
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmThinkingConfig,
    LlmTool,
)
from harnyx_commons.tools.embedding_models import QWEN3_OPENROUTER_EMBEDDING_MODEL

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]

OPENROUTER_LIVE_CHAT_MODEL = "openai/gpt-oss-20b"
OPENROUTER_REASONING_PROVIDER_BY_NATIVE_MODEL = {
    "openai/gpt-oss-20b": "wandb",
    "openai/gpt-oss-120b": "wandb",
    "deepseek/deepseek-v3.2": "deepinfra",
    "z-ai/glm-5": "streamlake",
    "qwen/qwen3.6-27b": "wandb",
    "google/gemma-4-31b-it": "wandb",
}


def _openrouter_request(*, model: str) -> LlmRequest:
    return LlmRequest(
        provider="openrouter",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=256,
        extra=_openrouter_reasoning_provider_extra(model=model),
        thinking=LlmThinkingConfig(enabled=True, effort="low"),
        timeout_seconds=180.0,
    )


def _openrouter_reasoning_request(*, model: str) -> LlmRequest:
    return LlmRequest(
        provider="openrouter",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Think briefly, then reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=256,
        extra=_openrouter_reasoning_provider_extra(model=model),
        thinking=LlmThinkingConfig(enabled=True, effort="low"),
        timeout_seconds=180.0,
    )


def _openrouter_reasoning_provider_extra(*, model: str) -> dict[str, object]:
    native_model = OPENROUTER_INTERNAL_TO_NATIVE_MODEL.get(model, model)
    provider = OPENROUTER_REASONING_PROVIDER_BY_NATIVE_MODEL.get(native_model)
    if provider is None:
        raise AssertionError(f"No OpenRouter reasoning provider pin configured for {model!r}")
    return {
        "provider": {
            "only": [provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    }


async def test_openrouter_provider_invokes_cheapest_chat_model_live() -> None:
    model = OPENROUTER_LIVE_CHAT_MODEL
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    provider = OpenRouterLlmProvider(
        openrouter_api_key=settings.openrouter_api_key,
    )
    request = _openrouter_request(model=model)

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "openrouter"
    assert response.metadata["effective_model"] == model
    assert response.usage.reasoning_tokens is not None
    assert response.usage.reasoning_tokens > 0


async def test_openrouter_provider_reasoning_live() -> None:
    model = OPENROUTER_LIVE_CHAT_MODEL
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    provider = OpenRouterLlmProvider(
        openrouter_api_key=settings.openrouter_api_key,
    )
    try:
        response = await provider.invoke(_openrouter_reasoning_request(model=model))
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "openrouter"
    assert response.metadata["effective_model"] == model
    assert response.choices[0].message.reasoning or (
        response.usage.reasoning_tokens is not None and response.usage.reasoning_tokens > 0
    )
    assert response.usage.reasoning_tokens is not None
    assert response.usage.reasoning_tokens > 0


async def test_miner_paid_openrouter_helper_completion_live() -> None:
    model = OPENROUTER_LIVE_CHAT_MODEL
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    provider = build_miner_paid_llm_provider(
        provider="openrouter",
        api_key=settings.openrouter_api_key,
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(_openrouter_request(model=model))
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "openrouter"
    assert response.metadata["effective_model"] == model


async def test_openrouter_embedding_client_invokes_qwen3_8b_live() -> None:
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    client = OpenRouterEmbeddingClient(
        model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
        api_key=settings.openrouter_api_key_value,
        dimensions=8,
        timeout_seconds=180.0,
    )
    try:
        response = await client.embed_many(("hello",))
    finally:
        await client.aclose()

    assert len(response.vectors) == 1
    assert len(response.vectors[0]) == 8
    assert response.usage is not None
    assert response.usage.prompt_tokens is not None
    assert response.usage.prompt_tokens > 0


async def test_openrouter_two_turn_function_tool_loop_with_reasoning_replay_live() -> None:
    model = OPENROUTER_LIVE_CHAT_MODEL
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"
    provider = OpenRouterLlmProvider(openrouter_api_key=settings.openrouter_api_key)
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

    try:
        first = await provider.invoke(
            LlmRequest(
                provider="openrouter",
                model=model,
                messages=(user_message,),
                temperature=0.0,
                max_output_tokens=256,
                tools=(tool,),
                tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
                thinking=LlmThinkingConfig(enabled=True, effort="low"),
                extra=_openrouter_reasoning_provider_extra(model=model),
                timeout_seconds=180.0,
            )
        )
        first_message = first.choices[0].message
        calls = first_message.tool_calls
        assert calls and calls[0].id
        assert first_message.reasoning_details
        replayed_details = tuple(first_message.reasoning_details)
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
                provider="openrouter",
                model=model,
                messages=(
                    user_message,
                    first_message.to_input_message(),
                    *tool_result_messages,
                ),
                temperature=0.0,
                max_output_tokens=256,
                tools=(tool,),
                tool_choice="none",
                thinking=LlmThinkingConfig(enabled=True, effort="low"),
                extra=_openrouter_reasoning_provider_extra(model=model),
                timeout_seconds=180.0,
            )
        )
    finally:
        await provider.aclose()

    assert first_message.to_input_message().reasoning_details == replayed_details
    assert second.raw_text
