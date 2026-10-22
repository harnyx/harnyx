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


def test_scoring_and_similarity_output_budgets_have_independent_defaults() -> None:
    settings = LlmSettings(_env_file=None)

    assert settings.scoring_llm_max_output_tokens == 32768
    assert settings.similarity_llm_max_output_tokens == 20480


def test_ai_gateway_api_key_value_strips_secret() -> None:
    settings = LlmSettings(AI_GATEWAY_API_KEY=" test-ai-gateway-key ")

    assert settings.ai_gateway_api_key_value == "test-ai-gateway-key"


def test_llm_settings_reject_empty_provider_name() -> None:
    with pytest.raises(ValidationError):
        LlmSettings(DIGEST_LLM_PROVIDER="")


def test_llm_settings_preserve_empty_freeform_model() -> None:
    settings = LlmSettings(DIGEST_LLM_MODEL="")

    assert settings.digest_llm_model == ""


@pytest.mark.parametrize("prefix", ["SCORING", "SIMILARITY"])
def test_fallback_models_parse_env_order_and_empty_list(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """Future failure: env overrides silently retain the expensive default chain."""
    key = f"{prefix}_LLM_FALLBACK_MODELS_JSON"
    attr = f"{prefix.lower()}_llm_fallback_models"
    monkeypatch.delenv(key, raising=False)
    assert getattr(LlmSettings(_env_file=None), attr) is None
    monkeypatch.setenv(key, '[" vendor/model-b ", "vendor/model-a"]')
    assert getattr(LlmSettings(_env_file=None), attr) == ("vendor/model-b", "vendor/model-a")
    monkeypatch.setenv(key, '[]')
    assert getattr(LlmSettings(_env_file=None), attr) == ()
    monkeypatch.setenv(key, "null")
    assert getattr(LlmSettings(_env_file=None), attr) is None


@pytest.mark.parametrize("prefix", ["SCORING", "SIMILARITY"])
@pytest.mark.parametrize("raw", ['[""]', '["   "]', '[1]', '["model", " model "]', '{}', '"model"'])
def test_fallback_models_reject_invalid_env(monkeypatch: pytest.MonkeyPatch, prefix: str, raw: str) -> None:
    """Future failure: malformed chains start a validator with unintended or repeated calls."""
    monkeypatch.setenv(f"{prefix}_LLM_FALLBACK_MODELS_JSON", raw)
    with pytest.raises(ValidationError):
        LlmSettings(_env_file=None)
