"""Sandbox configuration shared by platform and validator."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SandboxPullPolicy = Literal["always", "missing", "never"]


class SandboxSettings(BaseSettings):
    """Docker sandbox settings and budgets."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    sandbox_image: str = Field(
        default="local/caster-sandbox:0.1.0-dev", alias="CASTER_SANDBOX_IMAGE"
    )
    sandbox_network: str | None = Field(default="caster-sandbox-net", alias="CASTER_SANDBOX_NETWORK")
    sandbox_pull_policy: SandboxPullPolicy = Field(
        default="always", alias="CASTER_SANDBOX_PULL_POLICY"
    )
    sandbox_tool_base_url: str = Field(default="", alias="PLATFORM_SANDBOX_BASE_URL")

    max_session_budget_usd: float = Field(
        default=0.05,
        alias="MAX_SESSION_BUDGET_USD",
        description="Maximum allowed cost per miner session (USD).",
    )


__all__ = ["SandboxSettings", "SandboxPullPolicy"]
