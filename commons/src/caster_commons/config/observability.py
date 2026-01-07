"""Observability configuration shared by services."""

from __future__ import annotations

from pydantic import Field
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


__all__ = ["ObservabilitySettings"]
