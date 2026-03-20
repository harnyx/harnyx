from __future__ import annotations

from sentry_sdk.integrations.logging import LoggingIntegration

import harnyx_commons.observability.sentry as sentry_mod
from harnyx_commons.config.observability import ObservabilitySettings


def _observability(
    *,
    environment: str | None = "staging",
    release: str | None = None,
    traces_sample_rate: float | None = 0.5,
    send_default_pii: bool = False,
    enable_logs: bool = False,
    debug: bool = False,
) -> ObservabilitySettings:
    return ObservabilitySettings(
        SENTRY_ENVIRONMENT=environment,
        SENTRY_RELEASE=release,
        SENTRY_TRACES_SAMPLE_RATE=traces_sample_rate,
        SENTRY_SEND_DEFAULT_PII=send_default_pii,
        SENTRY_ENABLE_LOGS=enable_logs,
        SENTRY_DEBUG=debug,
    )


def test_configure_sentry_sdk_passes_shared_observability_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_init(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sentry_mod, "sentry_init", _fake_init)

    sentry_mod.configure_sentry_sdk(
        dsn="https://shared@example.invalid/1",
        observability=_observability(
            traces_sample_rate=1.0,
            send_default_pii=True,
            enable_logs=True,
            debug=True,
        ),
    )

    assert captured["dsn"] == "https://shared@example.invalid/1"
    assert captured["environment"] == "staging"
    assert captured["release"] is None
    assert captured["traces_sample_rate"] == 1.0
    assert captured["send_default_pii"] is True
    assert captured["enable_logs"] is True
    assert captured["debug"] is True
    assert captured["disabled_integrations"] == [LoggingIntegration]


def test_configure_sentry_sdk_noops_when_dsn_blank(monkeypatch) -> None:
    called = False

    def _fake_init(**_: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(sentry_mod, "sentry_init", _fake_init)

    sentry_mod.configure_sentry_sdk(dsn="", observability=_observability())

    assert called is False


def test_configure_sentry_sdk_from_env_passes_parseable_shared_options(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_init(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sentry_mod, "sentry_init", _fake_init)
    monkeypatch.setenv("SENTRY_DSN", "https://shared@example.invalid/1")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", " staging ")
    monkeypatch.setenv("SENTRY_RELEASE", " release-123 ")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.75")
    monkeypatch.setenv("SENTRY_SEND_DEFAULT_PII", "true")
    monkeypatch.setenv("SENTRY_ENABLE_LOGS", "1")
    monkeypatch.setenv("SENTRY_DEBUG", "yes")

    sentry_mod.configure_sentry_sdk_from_env(dsn_env_var="SENTRY_DSN")

    assert captured["dsn"] == "https://shared@example.invalid/1"
    assert captured["environment"] == "staging"
    assert captured["release"] == "release-123"
    assert captured["traces_sample_rate"] == 0.75
    assert captured["send_default_pii"] is True
    assert captured["enable_logs"] is True
    assert captured["debug"] is True
    assert captured["disabled_integrations"] == [LoggingIntegration]


def test_configure_sentry_sdk_from_env_ignores_malformed_shared_options(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_init(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sentry_mod, "sentry_init", _fake_init)
    monkeypatch.setenv("SENTRY_DSN", "https://shared@example.invalid/1")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "2")
    monkeypatch.setenv("SENTRY_SEND_DEFAULT_PII", "not-a-bool")
    monkeypatch.setenv("SENTRY_ENABLE_LOGS", "also-bad")
    monkeypatch.setenv("SENTRY_DEBUG", "wrong")

    sentry_mod.configure_sentry_sdk_from_env(dsn_env_var="SENTRY_DSN")

    assert captured["dsn"] == "https://shared@example.invalid/1"
    assert captured["traces_sample_rate"] is None
    assert captured["send_default_pii"] is False
    assert captured["enable_logs"] is False
    assert captured["debug"] is False


def test_configure_sentry_sdk_from_env_reads_dotenv_when_process_env_absent(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_init(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sentry_mod, "sentry_init", _fake_init)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SENTRY_DSN=https://shared-from-dotenv.example.invalid/1\nSENTRY_RELEASE=dotenv-release\n",
        encoding="utf-8",
    )

    sentry_mod.configure_sentry_sdk_from_env(dsn_env_var="SENTRY_DSN")

    assert captured["dsn"] == "https://shared-from-dotenv.example.invalid/1"
    assert captured["release"] == "dotenv-release"


def test_configure_sentry_sdk_from_env_prefers_process_env_over_dotenv(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_init(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sentry_mod, "sentry_init", _fake_init)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SENTRY_DSN=https://shared-from-dotenv.example.invalid/1\nSENTRY_RELEASE=dotenv-release\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SENTRY_DSN", "https://shared-from-env.example.invalid/1")
    monkeypatch.setenv("SENTRY_RELEASE", "env-release")

    sentry_mod.configure_sentry_sdk_from_env(dsn_env_var="SENTRY_DSN")

    assert captured["dsn"] == "https://shared-from-env.example.invalid/1"
    assert captured["release"] == "env-release"


def test_configure_sentry_sdk_from_env_reads_case_insensitive_process_env_keys(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_init(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sentry_mod, "sentry_init", _fake_init)
    monkeypatch.setenv("sentry_dsn", "https://shared-lowercase.example.invalid/1")
    monkeypatch.setenv("SeNtRy_EnViRoNmEnT", "mixed-env")
    monkeypatch.setenv("sentry_release", "lowercase-release")
    monkeypatch.setenv("SeNtRy_TrAcEs_SaMpLe_RaTe", "0.25")
    monkeypatch.setenv("sentry_send_default_pii", "true")
    monkeypatch.setenv("SeNtRy_EnAbLe_LoGs", "yes")
    monkeypatch.setenv("sentry_debug", "1")

    sentry_mod.configure_sentry_sdk_from_env(dsn_env_var="SENTRY_DSN")

    assert captured["dsn"] == "https://shared-lowercase.example.invalid/1"
    assert captured["environment"] == "mixed-env"
    assert captured["release"] == "lowercase-release"
    assert captured["traces_sample_rate"] == 0.25
    assert captured["send_default_pii"] is True
    assert captured["enable_logs"] is True
    assert captured["debug"] is True


def test_configure_sentry_sdk_from_env_matches_case_insensitive_process_env_precedence(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_init(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sentry_mod, "sentry_init", _fake_init)
    monkeypatch.setenv("SENTRY_DSN", "https://shared-uppercase.example.invalid/1")
    monkeypatch.setenv("SENTRY_RELEASE", "uppercase-release")
    monkeypatch.setenv("sentry_dsn", "https://shared-lowercase.example.invalid/2")
    monkeypatch.setenv("sentry_release", "lowercase-release")

    sentry_mod.configure_sentry_sdk_from_env(dsn_env_var="SENTRY_DSN")

    assert captured["dsn"] == "https://shared-lowercase.example.invalid/2"
    assert captured["release"] == "lowercase-release"


def test_capture_exception_forwards_to_sdk(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(sentry_mod, "_capture_exception", captured.append)

    sentry_mod.capture_exception(RuntimeError("boom"))

    assert [str(exc) for exc in captured] == ["boom"]


def test_capture_exception_for_status_only_sends_5xx(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(sentry_mod, "_capture_exception", captured.append)

    sentry_mod.capture_exception_for_status(RuntimeError("nope"), status_code=404)
    sentry_mod.capture_exception_for_status(RuntimeError("boom"), status_code=500)

    assert [str(exc) for exc in captured] == ["boom"]
