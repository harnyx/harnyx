"""Observability configuration shared by services."""

from __future__ import annotations

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ObservabilitySettings(BaseSettings):
    """Flags controlling logging/export behavior."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    enable_cloud_logging: bool = Field(default=False, alias="ENABLE_CLOUD_LOGGING")
    gcp_project_id: str | None = Field(default=None, alias="GCP_PROJECT_ID")
    sentry_dsn_platform_api: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        alias="SENTRY_DSN_PLATFORM_API",
    )
    sentry_dsn_platform_worker: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        alias="SENTRY_DSN_PLATFORM_WORKER",
    )
    sentry_dsn: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        alias="SENTRY_DSN",
    )
    sentry_environment: str | None = Field(default=None, alias="SENTRY_ENVIRONMENT")
    sentry_release: str | None = Field(default=None, alias="SENTRY_RELEASE")
    sentry_traces_sample_rate: float | None = Field(
        default=None,
        alias="SENTRY_TRACES_SAMPLE_RATE",
        ge=0.0,
        le=1.0,
    )
    sentry_send_default_pii: bool = Field(default=False, alias="SENTRY_SEND_DEFAULT_PII")
    sentry_enable_logs: bool = Field(default=False, alias="SENTRY_ENABLE_LOGS")
    sentry_debug: bool = Field(default=False, alias="SENTRY_DEBUG")

    @field_validator("sentry_traces_sample_rate", mode="before")
    @classmethod
    def _empty_sentry_traces_sample_rate_is_none(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if stripped == "":
            return None
        return stripped

    @property
    def sentry_dsn_platform_api_value(self) -> str:
        return self.sentry_dsn_platform_api.get_secret_value().strip()

    @property
    def sentry_dsn_platform_worker_value(self) -> str:
        return self.sentry_dsn_platform_worker.get_secret_value().strip()

    @property
    def sentry_dsn_value(self) -> str:
        return self.sentry_dsn.get_secret_value().strip()


__all__ = ["ObservabilitySettings"]
