from __future__ import annotations

import pytest

from harnyx_commons.config.llm import LlmSettings, OpenRouterModelProviderOptions
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.providers.openrouter import OPENROUTER_SUPPORTED_MODELS, OpenRouterLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmThinkingConfig

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


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
        thinking=LlmThinkingConfig(enabled=True, effort="low"),
        timeout_seconds=180.0,
    )


@pytest.mark.parametrize("model", OPENROUTER_SUPPORTED_MODELS)
async def test_openrouter_provider_invokes_supported_model_live(model: str) -> None:
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    provider = OpenRouterLlmProvider(
        openrouter_api_key=settings.openrouter_api_key,
        model_provider_options={
            model: OpenRouterModelProviderOptions(require_parameters=True),
        },
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
    raw_usage = response.metadata["raw_response"]["usage"]
    assert isinstance(raw_usage["cost"], (int, float))
    assert raw_usage["cost"] >= 0.0
    assert response.usage.reasoning_tokens is None or response.usage.reasoning_tokens >= 0


async def test_openrouter_provider_reasoning_live() -> None:
    model = "openai/gpt-oss-20b"
    settings = LlmSettings(
        OPENROUTER_MODEL_PROVIDER_OPTIONS_JSON='{"openai/gpt-oss-20b":{"require_parameters":true}}',
    )
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    provider = OpenRouterLlmProvider(
        openrouter_api_key=settings.openrouter_api_key,
        model_provider_options={
            model: OpenRouterModelProviderOptions(require_parameters=True),
        },
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


async def test_miner_paid_openrouter_helper_completion_live() -> None:
    model = "openai/gpt-oss-20b"
    settings = LlmSettings(
        OPENROUTER_MODEL_PROVIDER_OPTIONS_JSON='{"openai/gpt-oss-20b":{"require_parameters":true}}',
    )
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
