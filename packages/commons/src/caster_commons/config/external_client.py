"""Configuration for external client retry behavior."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from caster_commons.llm.retry_utils import RetryPolicy

DEFAULT_EXTERNAL_CLIENT_RETRY_ATTEMPTS = 10
DEFAULT_EXTERNAL_CLIENT_RETRY_INITIAL_MS = 1000
DEFAULT_EXTERNAL_CLIENT_RETRY_MAX_MS = 30000
DEFAULT_EXTERNAL_CLIENT_RETRY_JITTER = 0.2


class ExternalClientRetrySettings(BaseSettings):
    """Retry/backoff policy shared across external HTTP clients."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    attempts: int = Field(
        default=DEFAULT_EXTERNAL_CLIENT_RETRY_ATTEMPTS,
        alias="EXTERNAL_CLIENT_RETRY_ATTEMPTS",
        ge=1,
    )
    initial_ms: int = Field(
        default=DEFAULT_EXTERNAL_CLIENT_RETRY_INITIAL_MS,
        alias="EXTERNAL_CLIENT_RETRY_INITIAL_MS",
        ge=0,
    )
    max_ms: int = Field(
        default=DEFAULT_EXTERNAL_CLIENT_RETRY_MAX_MS,
        alias="EXTERNAL_CLIENT_RETRY_MAX_MS",
        ge=0,
    )
    jitter: float = Field(
        default=DEFAULT_EXTERNAL_CLIENT_RETRY_JITTER,
        alias="EXTERNAL_CLIENT_RETRY_JITTER",
        ge=0.0,
        le=1.0,
    )

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            attempts=self.attempts,
            initial_ms=self.initial_ms,
            max_ms=self.max_ms,
            jitter=self.jitter,
        )


__all__ = ["ExternalClientRetrySettings"]
