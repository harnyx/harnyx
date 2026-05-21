from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_settings_defaults_artifact_task_parallelism_to_external_default(monkeypatch) -> None:
    monkeypatch.delenv("VALIDATOR_TASK_PARALLELISM", raising=False)

    settings = Settings.load()

    assert settings.artifact_task_parallelism == 20


def test_settings_defaults_artifact_parallelism_to_external_default(monkeypatch) -> None:
    monkeypatch.delenv("VALIDATOR_ARTIFACT_PARALLELISM", raising=False)

    settings = Settings.load()

    assert settings.artifact_parallelism == 4


def test_settings_accepts_artifact_parallelism_override(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_ARTIFACT_PARALLELISM", "2")

    settings = Settings.load()

    assert settings.artifact_parallelism == 2


def test_settings_rejects_non_positive_artifact_parallelism(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_ARTIFACT_PARALLELISM", "0")

    with pytest.raises(ValidationError):
        Settings.load()


def test_settings_accepts_artifact_task_parallelism_override(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_TASK_PARALLELISM", "5")

    settings = Settings.load()

    assert settings.artifact_task_parallelism == 5


def test_settings_defaults_validator_state_dir(monkeypatch) -> None:
    monkeypatch.delenv("VALIDATOR_STATE_DIR", raising=False)

    settings = Settings.load()

    assert settings.validator_state_dir == Path("/workspace/.harnyx_state")


def test_settings_accepts_validator_state_dir_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VALIDATOR_STATE_DIR", str(tmp_path / "validator-state"))

    settings = Settings.load()

    assert settings.validator_state_dir == tmp_path / "validator-state"


def test_settings_defaults_run_progress_retention(monkeypatch) -> None:
    monkeypatch.delenv("VALIDATOR_RUN_PROGRESS_RETENTION_SECONDS", raising=False)
    monkeypatch.delenv("VALIDATOR_RUN_PROGRESS_CLEANUP_INTERVAL_SECONDS", raising=False)

    settings = Settings.load()

    assert settings.run_progress_retention_seconds == 24 * 60 * 60
    assert settings.run_progress_cleanup_interval_seconds == 10 * 60


def test_settings_accepts_run_progress_retention_overrides(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_RUN_PROGRESS_RETENTION_SECONDS", "3600")
    monkeypatch.setenv("VALIDATOR_RUN_PROGRESS_CLEANUP_INTERVAL_SECONDS", "30")

    settings = Settings.load()

    assert settings.run_progress_retention_seconds == 3600
    assert settings.run_progress_cleanup_interval_seconds == 30


def test_settings_rejects_non_positive_run_progress_retention(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_RUN_PROGRESS_RETENTION_SECONDS", "0")

    with pytest.raises(ValidationError):
        Settings.load()


def test_settings_rejects_non_positive_run_progress_cleanup_interval(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_RUN_PROGRESS_CLEANUP_INTERVAL_SECONDS", "0")

    with pytest.raises(ValidationError):
        Settings.load()


def test_settings_rejects_non_positive_artifact_task_parallelism(monkeypatch) -> None:
    monkeypatch.setenv("VALIDATOR_TASK_PARALLELISM", "0")

    with pytest.raises(ValidationError):
        Settings.load()


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
