"""Sandbox configuration shared by platform and validator."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SandboxPullPolicy = Literal["always", "missing", "never"]


class SandboxSettings(BaseSettings):
    """Docker sandbox settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_ignore_empty=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    sandbox_image: str = Field(
        ...,
        validation_alias=AliasChoices("CASTER_SANDBOX_IMAGE", "SANDBOX_IMAGE"),
    )
    sandbox_network: str | None = Field(
        default="caster-sandbox-net",
        validation_alias=AliasChoices("CASTER_SANDBOX_NETWORK", "SANDBOX_NETWORK"),
    )
    sandbox_pull_policy: SandboxPullPolicy = Field(
        default="always",
        validation_alias=AliasChoices("CASTER_SANDBOX_PULL_POLICY", "SANDBOX_PULL_POLICY"),
    )


def load_sandbox_settings() -> SandboxSettings:
    # Pydantic settings load required values from env/.env at runtime.
    return SandboxSettings()


__all__ = ["SandboxSettings", "SandboxPullPolicy", "load_sandbox_settings"]
