"""Validator-specific Sentry bootstrap and capture helpers."""

from __future__ import annotations

from harnyx_commons.config.observability import ObservabilitySettings
from harnyx_commons.observability.sentry import (
    capture_exception,
    configure_sentry_sdk,
    configure_sentry_sdk_from_env,
)
from harnyx_validator.runtime.settings import Settings


def configure_sentry_from_env() -> None:
    configure_sentry_sdk_from_env(dsn_env_var="SENTRY_DSN")


def configure_sentry_from_observability(*, observability: ObservabilitySettings) -> None:
    configure_sentry_sdk(
        dsn=observability.sentry_dsn_value,
        observability=observability,
    )


def configure_sentry(*, settings: Settings) -> None:
    configure_sentry_from_observability(observability=settings.observability)


__all__ = [
    "capture_exception",
    "configure_sentry",
    "configure_sentry_from_env",
    "configure_sentry_from_observability",
]
