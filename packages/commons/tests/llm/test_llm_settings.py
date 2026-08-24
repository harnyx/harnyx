from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_commons.config.llm import LlmSettings


def test_openrouter_api_key_value_strips_secret() -> None:
    settings = LlmSettings(OPENROUTER_API_KEY=" test-openrouter-key ")

    assert settings.openrouter_api_key_value == "test-openrouter-key"


def test_tool_embedding_provider_defaults_to_chutes_for_direct_runtime_compatibility() -> None:
    settings = LlmSettings()

    assert settings.tool_embedding_provider == "chutes"


def test_tool_embedding_provider_can_be_openrouter() -> None:
    settings = LlmSettings(TOOL_EMBEDDING_PROVIDER="openrouter")

    assert settings.tool_embedding_provider == "openrouter"


def test_ai_gateway_api_key_value_strips_secret() -> None:
    settings = LlmSettings(AI_GATEWAY_API_KEY=" test-ai-gateway-key ")

    assert settings.ai_gateway_api_key_value == "test-ai-gateway-key"


def test_llm_settings_reject_empty_provider_name() -> None:
    with pytest.raises(ValidationError):
        LlmSettings(DIGEST_LLM_PROVIDER="")


def test_llm_settings_preserve_empty_freeform_model() -> None:
    settings = LlmSettings(DIGEST_LLM_MODEL="")

    assert settings.digest_llm_model == ""
