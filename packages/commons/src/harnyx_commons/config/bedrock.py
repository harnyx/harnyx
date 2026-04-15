"""AWS Bedrock configuration shared across services."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BedrockSettings(BaseSettings):
    """Region and transport settings for Bedrock runtime calls."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    region: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BEDROCK_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
    )
    connect_timeout_seconds: float = Field(default=5.0, alias="BEDROCK_CONNECT_TIMEOUT_SECONDS")
    read_timeout_seconds: float = Field(default=60.0, alias="BEDROCK_READ_TIMEOUT_SECONDS")

    @property
    def region_value(self) -> str:
        region = (self.region or "").strip()
        if not region:
            raise ValueError("BEDROCK_AWS_REGION or AWS_REGION must be configured for Bedrock")
        return region


__all__ = ["BedrockSettings"]
