from __future__ import annotations

from harnyx_validator.runtime.settings import Settings


def test_settings_defaults_sandbox_image_when_unset(monkeypatch) -> None:
    """Settings default the validator sandbox image when no override is supplied."""
    monkeypatch.setenv("TOOL_LLM_PROVIDER", "chutes")
    monkeypatch.delenv("SANDBOX_IMAGE", raising=False)

    settings = Settings.load()

    assert settings.llm.tool_llm_provider == "chutes"
    assert settings.sandbox.sandbox_image == "harnyx/harnyx-subnet-sandbox:finney"


def test_settings_honor_sandbox_image_override(monkeypatch) -> None:
    """Settings still honor explicit validator sandbox image overrides."""
    monkeypatch.setenv("TOOL_LLM_PROVIDER", "chutes")
    monkeypatch.setenv("SANDBOX_IMAGE", "test-sandbox:latest")

    settings = Settings.load()

    assert settings.llm.tool_llm_provider == "chutes"
    assert settings.sandbox.sandbox_image == "test-sandbox:latest"


def test_settings_accepts_validator_env_names(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_HOST", "127.0.0.1")
    monkeypatch.setenv("VALIDATOR_PORT", "9001")

    settings = Settings.load()

    assert settings.rpc_listen_host == "127.0.0.1"
    assert settings.rpc_port == 9001


def test_settings_accepts_validator_env_names_when_empty_values_set_first(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_HOST", "")
    monkeypatch.setenv("VALIDATOR_PORT", "")
    monkeypatch.setenv("VALIDATOR_HOST", "127.0.0.1")
    monkeypatch.setenv("VALIDATOR_PORT", "9001")

    settings = Settings.load()

    assert settings.rpc_listen_host == "127.0.0.1"
    assert settings.rpc_port == 9001


def test_settings_honor_sandbox_image_override_from_dotenv(tmp_path, monkeypatch) -> None:
    """Settings honor validator sandbox overrides from a local .env file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOOL_LLM_PROVIDER", "chutes")
    monkeypatch.delenv("SANDBOX_IMAGE", raising=False)
    (tmp_path / ".env").write_text("SANDBOX_IMAGE=dotenv-sandbox:latest\n", encoding="utf-8")

    settings = Settings.load()

    assert settings.llm.tool_llm_provider == "chutes"
    assert settings.sandbox.sandbox_image == "dotenv-sandbox:latest"
