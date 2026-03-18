"""Sandbox configuration shared by platform and validator."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
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
        alias="SANDBOX_IMAGE",
    )
    sandbox_network: str | None = Field(
        default="harnyx-sandbox-net",
        alias="SANDBOX_NETWORK",
    )
    sandbox_pull_policy: SandboxPullPolicy = Field(
        default="always",
        alias="SANDBOX_PULL_POLICY",
    )


def load_sandbox_settings() -> SandboxSettings:
    # Pydantic settings load required values from env/.env at runtime.
    return SandboxSettings()


__all__ = ["SandboxSettings", "SandboxPullPolicy", "load_sandbox_settings"]
