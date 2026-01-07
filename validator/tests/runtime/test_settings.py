from __future__ import annotations

from caster_validator.runtime.settings import Settings


def test_settings_loads_from_environment(monkeypatch) -> None:
    """Settings loads from environment variables."""
    monkeypatch.setenv("TOOL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CASTER_SANDBOX_IMAGE", "test-sandbox:latest")

    settings = Settings.load()

    assert settings.llm.tool_llm_provider == "openai"
    assert settings.sandbox.sandbox_image == "test-sandbox:latest"
