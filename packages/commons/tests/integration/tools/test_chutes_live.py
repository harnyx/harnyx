"""Check Chutes text and structured-output contracts with a representative model."""

import pytest
from pydantic import BaseModel, ConfigDict

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmThinkingConfig

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


class _StructuredReply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    message: str
    number: int


async def test_chutes_liveness_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="chutes",
        api_key=settings.chutes_api_key,
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(
            LlmRequest(
                provider="chutes",
                model="google/gemma-4-31B-turbo-TEE",
                messages=(
                    LlmMessage(
                        role="user",
                        content=(LlmMessageContentPart.input_text("Say hello."),),
                    ),
                ),
                temperature=None,
                max_output_tokens=1024,
                thinking=LlmThinkingConfig(enabled=False),
                timeout=60.0,
            )
        )
    finally:
        await provider.aclose()

    assert response.raw_text.strip()


async def test_chutes_structured_output_live() -> None:
    """Catch a broken JSON-schema request or a response that violates the schema."""
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="chutes",
        api_key=settings.chutes_api_key,
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(
            LlmRequest(
                provider="chutes",
                model="google/gemma-4-31B-turbo-TEE",
                messages=(
                    LlmMessage(
                        role="user",
                        content=(LlmMessageContentPart.input_text(
                            'Return a JSON object with message "hello" and number 7.',
                        ),),
                    ),
                ),
                temperature=None,
                max_output_tokens=1024,
                thinking=LlmThinkingConfig(enabled=False),
                output_mode="structured",
                output_schema=_StructuredReply,
                timeout=60.0,
            )
        )
    finally:
        await provider.aclose()

    result = _StructuredReply.model_validate_json(response.raw_text)
    assert result.message == "hello"
    assert result.number == 7
