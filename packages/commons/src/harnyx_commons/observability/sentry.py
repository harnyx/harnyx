"""Shared Sentry SDK bootstrap and capture helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic_settings.sources import DotEnvSettingsSource
from sentry_sdk import capture_exception as _capture_exception
from sentry_sdk import init as sentry_init
from sentry_sdk.integrations.logging import LoggingIntegration

from harnyx_commons.config.observability import ObservabilitySettings


def configure_sentry_sdk(*, dsn: str, observability: ObservabilitySettings) -> None:
    _configure_sentry_sdk(
        dsn=dsn,
        environment=_clean_optional(observability.sentry_environment),
        release=_clean_optional(observability.sentry_release),
        traces_sample_rate=observability.sentry_traces_sample_rate,
        send_default_pii=observability.sentry_send_default_pii,
        enable_logs=observability.sentry_enable_logs,
        debug=observability.sentry_debug,
    )


def configure_sentry_sdk_from_env(*, dsn_env_var: str) -> None:
    dotenv_env = _dotenv_bootstrap_env()
    _configure_sentry_sdk(
        dsn=_clean_optional(_bootstrap_env_value(dsn_env_var, dotenv_env)) or "",
        environment=_clean_optional(_bootstrap_env_value("SENTRY_ENVIRONMENT", dotenv_env)),
        release=_clean_optional(_bootstrap_env_value("SENTRY_RELEASE", dotenv_env)),
        traces_sample_rate=_clean_sample_rate(_bootstrap_env_value("SENTRY_TRACES_SAMPLE_RATE", dotenv_env)),
        send_default_pii=_clean_bool(_bootstrap_env_value("SENTRY_SEND_DEFAULT_PII", dotenv_env)),
        enable_logs=_clean_bool(_bootstrap_env_value("SENTRY_ENABLE_LOGS", dotenv_env)),
        debug=_clean_bool(_bootstrap_env_value("SENTRY_DEBUG", dotenv_env)),
    )


def _configure_sentry_sdk(
    *,
    dsn: str,
    environment: str | None,
    release: str | None,
    traces_sample_rate: float | None,
    send_default_pii: bool,
    enable_logs: bool,
    debug: bool,
) -> None:
    if dsn == "":
        return

    sentry_init(
        dsn=dsn,
        environment=environment,
        release=release,
        traces_sample_rate=traces_sample_rate,
        send_default_pii=send_default_pii,
        enable_logs=enable_logs,
        debug=debug,
        disabled_integrations=[LoggingIntegration],
    )


def capture_exception(exc: BaseException) -> None:
    _capture_exception(exc)


def capture_exception_for_status(exc: Exception, *, status_code: int) -> None:
    if status_code < 500:
        return
    _capture_exception(exc)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _clean_sample_rate(value: str | None) -> float | None:
    cleaned = _clean_optional(value)
    if cleaned is None:
        return None
    try:
        sample_rate = float(cleaned)
    except ValueError:
        return None
    if 0.0 <= sample_rate <= 1.0:
        return sample_rate
    return None


def _clean_bool(value: str | None) -> bool:
    cleaned = _clean_optional(value)
    if cleaned is None:
        return False
    normalized = cleaned.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return False


def _dotenv_bootstrap_env() -> Mapping[str, str | None]:
    return DotEnvSettingsSource(ObservabilitySettings).env_vars


def _bootstrap_env_value(name: str, dotenv_env: Mapping[str, str | None]) -> str | None:
    process_value = _case_insensitive_process_env_value(name)
    if process_value is not None:
        return process_value
    return dotenv_env.get(name.lower())


def _case_insensitive_process_env_value(name: str) -> str | None:
    lowered_name = name.lower()
    matched_value: str | None = None
    for env_name, env_value in os.environ.items():
        if env_name.lower() == lowered_name:
            matched_value = env_value
    return matched_value


__all__ = [
    "capture_exception",
    "capture_exception_for_status",
    "configure_sentry_sdk",
    "configure_sentry_sdk_from_env",
]
