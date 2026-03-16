"""Shared Subtensor connectivity settings."""

from __future__ import annotations

from pydantic import Field, SecretStr, field_validator
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
    hotkey_mnemonic: SecretStr | None = Field(default=None, alias="SUBTENSOR_HOTKEY_MNEMONIC")
    wait_for_inclusion: bool = Field(default=True, alias="SUBTENSOR_WAIT_FOR_INCLUSION")
    wait_for_finalization: bool = Field(default=False, alias="SUBTENSOR_WAIT_FOR_FINALIZATION")
    transaction_mode: str = Field(default="immortal", alias="TRANSACTION_MODE")
    transaction_period: int | None = Field(default=None, alias="TRANSACTION_MODE_PERIOD")

    @field_validator("hotkey_mnemonic", mode="before")
    @classmethod
    def _normalize_hotkey_mnemonic(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("SUBTENSOR_HOTKEY_MNEMONIC must be a non-empty string when provided")
            return SecretStr(stripped)
        return value

    @property
    def hotkey_mnemonic_value(self) -> str | None:
        if self.hotkey_mnemonic is None:
            return None
        return self.hotkey_mnemonic.get_secret_value()


__all__ = ["SubtensorSettings"]
