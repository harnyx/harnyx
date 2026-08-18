from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.providers.openrouter import (
    OPENROUTER_INTERNAL_TO_NATIVE_MODEL,
    OpenRouterEmbeddingClient,
    OpenRouterLlmProvider,
)
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmThinkingConfig,
    LlmTool,
)
from harnyx_commons.tools.embedding_models import QWEN3_OPENROUTER_EMBEDDING_MODEL

pytestmark = [
    pytest.mark.integration,
    pytest.mark.expensive,
    pytest.mark.anyio("asyncio"),
    pytest.mark.flaky(reruns=1),
]

OPENROUTER_LIVE_CHAT_MODEL = "openai/gpt-oss-20b"
OPENROUTER_BYOK_LIVE_CHAT_MODEL = "openai/gpt-oss-120b"
NEW_OPENROUTER_MODELS = (
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-pro",
    "z-ai/glm-5.2",
    "thinkingmachines/inkling",
    "qwen/qwen3.5-397b-a17b",
    "meta/muse-glimmer-30b",
    "qwen/qwen3.8-27b",
)
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
                content=(
                    LlmMessageContentPart.input_text(
                        f'Reply with only "ok". Request nonce: {uuid4().hex}.'
                    ),
                ),
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


def _openrouter_byok_request() -> LlmRequest:
    return LlmRequest(
        provider="openrouter",
        model=OPENROUTER_BYOK_LIVE_CHAT_MODEL,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=128,
        extra={"provider": {"only": ["cerebras"], "allow_fallbacks": False}},
        timeout_seconds=180.0,
        retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0.0),
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
    assert response.metadata["actual_cost_evidence"]["upstream_provider"]
    assert response.metadata["actual_cost_evidence"]["upstream_model"]
    assert response.metadata["actual_cost_evidence"]["provider_request_id"] == response.id
    assert response.usage.reasoning_tokens is not None
    assert response.usage.reasoning_tokens > 0


async def test_openrouter_byok_completion_settles_original_response_cost_live() -> None:
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    provider = OpenRouterLlmProvider(openrouter_api_key=settings.openrouter_api_key)
    try:
        response = await provider.invoke(_openrouter_byok_request())
    finally:
        await provider.aclose()

    assert response.metadata is not None
    raw_response = response.metadata.get("raw_response")
    assert isinstance(raw_response, Mapping)
    usage = raw_response.get("usage")
    assert isinstance(usage, Mapping)
    assert usage.get("is_byok") is True
    openrouter_cost = usage.get("cost")
    assert isinstance(openrouter_cost, int | float) and not isinstance(openrouter_cost, bool)
    cost_details = usage.get("cost_details")
    assert isinstance(cost_details, Mapping)
    upstream_inference_cost = cost_details.get("upstream_inference_cost")
    assert isinstance(upstream_inference_cost, int | float) and not isinstance(
        upstream_inference_cost, bool
    )
    assert upstream_inference_cost >= 0.0
    assert response.metadata["actual_cost_usd"] == pytest.approx(
        float(openrouter_cost) + float(upstream_inference_cost)
    )
    evidence = response.metadata["actual_cost_evidence"]
    assert isinstance(evidence, Mapping)
    assert evidence["usage"] == {
        "is_byok": True,
        "cost": float(openrouter_cost),
        "cost_details": {"upstream_inference_cost": float(upstream_inference_cost)},
    }


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
        api_key=settings.openrouter_api_key,
        dimensions=32,
        timeout_seconds=180.0,
    )
    try:
        response = await client.embed_many(("hello",))
    finally:
        await client.aclose()

    assert len(response.vectors) == 1
    assert len(response.vectors[0]) == 32
    assert response.usage is not None
    assert response.usage.prompt_tokens is not None
    assert response.usage.prompt_tokens > 0
    assert response.model
    if response.usage.cost is not None:
        assert isinstance(response.usage.cost, int | float)
        assert response.usage.cost >= 0.0
    if response.id is not None:
        assert response.id


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


@pytest.mark.parametrize("model", NEW_OPENROUTER_MODELS)
async def test_new_openrouter_model_exact_route_contract_live(model: str) -> None:
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
    direct_messages = (
        LlmMessage(
            role="user",
            content=(LlmMessageContentPart.input_text('Think briefly, then reply with only "ok".'),),
        ),
    )
    user_message = LlmMessage(
        role="user",
        content=(LlmMessageContentPart.input_text("Use lookup_weather for Paris."),),
    )
    common = {
        "provider": "openrouter",
        "model": model,
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "thinking": LlmThinkingConfig(enabled=True),
        "timeout_seconds": 180.0,
    }

    try:
        direct = await provider.invoke(LlmRequest(messages=direct_messages, **common))
        first = await provider.invoke(
            LlmRequest(
                messages=(user_message,),
                tools=(tool,),
                tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
                **common,
            )
        )
        calls = first.choices[0].message.tool_calls
        assert calls and calls[0].id
        second = await provider.invoke(
            LlmRequest(
                messages=(
                    user_message,
                    first.choices[0].message.to_input_message(),
                    *(
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
                    ),
                ),
                tools=(tool,),
                tool_choice="none",
                **common,
            )
        )
    finally:
        await provider.aclose()

    assert direct.raw_text
    assert direct.choices[0].message.reasoning or (
        direct.usage.reasoning_tokens is not None and direct.usage.reasoning_tokens > 0
    )
    for response in (direct, first, second):
        assert response.metadata is not None
        assert response.metadata["effective_provider"] == "openrouter"
        assert response.metadata["effective_model"] == model
        assert response.metadata["actual_cost_provider"] == "openrouter"
        assert response.metadata["actual_cost_usd"] >= 0.0
        assert response.usage.total_tokens is not None
        assert response.usage.total_tokens > 0
    assert second.raw_text
