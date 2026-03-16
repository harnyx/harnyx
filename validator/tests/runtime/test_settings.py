from __future__ import annotations

from caster_validator.runtime.settings import Settings


def test_settings_defaults_sandbox_image_when_unset(monkeypatch) -> None:
    """Settings default the validator sandbox image when no override is supplied."""
    monkeypatch.setenv("TOOL_LLM_PROVIDER", "chutes")
    monkeypatch.delenv("CASTER_SANDBOX_IMAGE", raising=False)

    settings = Settings.load()

    assert settings.llm.tool_llm_provider == "chutes"
    assert settings.sandbox.sandbox_image == "castersubnet/caster-subnet-sandbox:finney"


def test_settings_honor_sandbox_image_override(monkeypatch) -> None:
    """Settings still honor explicit validator sandbox image overrides."""
    monkeypatch.setenv("TOOL_LLM_PROVIDER", "chutes")
    monkeypatch.setenv("CASTER_SANDBOX_IMAGE", "test-sandbox:latest")

    settings = Settings.load()

    assert settings.llm.tool_llm_provider == "chutes"
    assert settings.sandbox.sandbox_image == "test-sandbox:latest"


def test_settings_honor_sandbox_image_override_from_dotenv(tmp_path, monkeypatch) -> None:
    """Settings honor validator sandbox overrides from a local .env file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOOL_LLM_PROVIDER", "chutes")
    monkeypatch.delenv("CASTER_SANDBOX_IMAGE", raising=False)
    (tmp_path / ".env").write_text("CASTER_SANDBOX_IMAGE=dotenv-sandbox:latest\n", encoding="utf-8")

    settings = Settings.load()

    assert settings.llm.tool_llm_provider == "chutes"
    assert settings.sandbox.sandbox_image == "dotenv-sandbox:latest"
