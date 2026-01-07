"""Shared Subtensor connectivity settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SubtensorSettings(BaseSettings):
    """Configuration for subtensor connectivity and wallet selection."""

    model_config = SettingsConfigDict(
        frozen=True,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    network: str = Field(default="local", alias="SUBTENSOR_NETWORK")
    endpoint: str = Field(default="ws://127.0.0.1:9945", alias="SUBTENSOR_ENDPOINT")
    netuid: int = Field(default=1, alias="SUBTENSOR_NETUID")
    wallet_name: str = Field(default="caster-validator", alias="SUBTENSOR_WALLET_NAME")
    hotkey_name: str = Field(default="default", alias="SUBTENSOR_HOTKEY_NAME")
    wait_for_inclusion: bool = Field(default=True, alias="SUBTENSOR_WAIT_FOR_INCLUSION")
    wait_for_finalization: bool = Field(default=False, alias="SUBTENSOR_WAIT_FOR_FINALIZATION")
    transaction_mode: str = Field(default="immortal", alias="TRANSACTION_MODE")
    transaction_period: int | None = Field(default=None, alias="TRANSACTION_MODE_PERIOD")


__all__ = ["SubtensorSettings"]
