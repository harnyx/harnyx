"""Platform API connectivity settings shared with validators."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformApiSettings(BaseSettings):
    """Endpoints and signing configuration for platform interactions."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    platform_base_url: str | None = Field(default=None, alias="PLATFORM_BASE_URL")
    platform_hotkey_ss58: str = Field(default="", alias="PLATFORM_HOTKEY_SS58")
    validator_public_base_url: str | None = Field(default=None, alias="VALIDATOR_PUBLIC_BASE_URL")


__all__ = ["PlatformApiSettings"]
