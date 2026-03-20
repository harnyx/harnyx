from __future__ import annotations

from types import SimpleNamespace

import harnyx_validator.infrastructure.observability.sentry as sentry_mod
from harnyx_commons.config.observability import ObservabilitySettings


def _settings(
    *,
    dsn: str = "https://validator@example.invalid/1",
    environment: str | None = "staging",
    release: str | None = None,
    traces_sample_rate: float | None = 0.5,
    send_default_pii: bool = False,
    enable_logs: bool = False,
    debug: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        observability=ObservabilitySettings(
            SENTRY_DSN=dsn,
            SENTRY_ENVIRONMENT=environment,
            SENTRY_RELEASE=release,
            SENTRY_TRACES_SAMPLE_RATE=traces_sample_rate,
            SENTRY_SEND_DEFAULT_PII=send_default_pii,
            SENTRY_ENABLE_LOGS=enable_logs,
            SENTRY_DEBUG=debug,
        )
    )


def test_configure_sentry_delegates_validator_dsn_to_commons(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_configure_sentry_sdk(*, dsn: str, observability: ObservabilitySettings) -> None:
        captured["dsn"] = dsn
        captured["observability"] = observability

    monkeypatch.setattr(sentry_mod, "configure_sentry_sdk", _fake_configure_sentry_sdk)

    settings = _settings(
        traces_sample_rate=1.0,
        send_default_pii=True,
        enable_logs=True,
        debug=True,
    )

    sentry_mod.configure_sentry_from_observability(observability=settings.observability)

    assert captured["dsn"] == "https://validator@example.invalid/1"
    assert captured["observability"] is settings.observability


def test_configure_sentry_passes_blank_dsn_to_commons(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_configure_sentry_sdk(*, dsn: str, observability: ObservabilitySettings) -> None:
        captured["dsn"] = dsn
        captured["observability"] = observability

    monkeypatch.setattr(sentry_mod, "configure_sentry_sdk", _fake_configure_sentry_sdk)

    settings = _settings(dsn="")
    sentry_mod.configure_sentry_from_observability(observability=settings.observability)

    assert captured["dsn"] == ""
    assert captured["observability"] is settings.observability


def test_configure_sentry_from_env_delegates_validator_dsn_to_commons(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_configure_sentry_sdk_from_env(*, dsn_env_var: str) -> None:
        captured["dsn_env_var"] = dsn_env_var

    monkeypatch.setattr(sentry_mod, "configure_sentry_sdk_from_env", _fake_configure_sentry_sdk_from_env)

    sentry_mod.configure_sentry_from_env()

    assert captured["dsn_env_var"] == "SENTRY_DSN"


def test_configure_sentry_delegates_settings_observability_to_early_bootstrap(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_configure_sentry_from_observability(*, observability: ObservabilitySettings) -> None:
        captured["observability"] = observability

    monkeypatch.setattr(
        sentry_mod,
        "configure_sentry_from_observability",
        _fake_configure_sentry_from_observability,
    )

    settings = _settings()
    sentry_mod.configure_sentry(settings=settings)

    assert captured["observability"] is settings.observability
