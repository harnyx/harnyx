"""LLM-related configuration shared by platform and validator."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from caster_commons.llm.provider_types import LlmProviderName

DEFAULT_MAX_OUTPUT_TOKENS = 1024


class LlmSettings(BaseSettings):
    """Configuration for LLM providers and related API keys."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    # --- Tooling / search ---
    tool_llm_provider: LlmProviderName = Field(default="openai", alias="TOOL_LLM_PROVIDER")
    search_provider: Literal["desearch"] | None = Field(default=None, alias="SEARCH_PROVIDER")

    # --- Generation / reference / benchmark ---
    generator_llm_provider: LlmProviderName = Field(default="openai", alias="GENERATOR_LLM_PROVIDER")
    generator_llm_model: str = Field(default="", alias="GENERATOR_LLM_MODEL")
    generator_llm_reasoning_effort: str | None = Field(
        default=None, alias="GENERATOR_LLM_REASONING_EFFORT"
    )
    generator_llm_temperature: float | None = Field(default=None, alias="GENERATOR_TEMPERATURE")
    generator_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="GENERATOR_LLM_MAX_OUTPUT_TOKENS"
    )
    generator_harden_claims: bool = Field(default=False, alias="GENERATOR_HARDEN_CLAIMS")

    reference_llm_provider: LlmProviderName = Field(default="openai", alias="REFERENCE_LLM_PROVIDER")
    reference_llm_model: str = Field(default="", alias="REFERENCE_LLM_MODEL")
    reference_llm_reasoning_effort: str | None = Field(
        default=None, alias="REFERENCE_LLM_REASONING_EFFORT"
    )
    reference_llm_temperature: float | None = Field(default=None, alias="REFERENCE_TEMPERATURE")
    reference_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="REFERENCE_LLM_MAX_OUTPUT_TOKENS"
    )

    benchmark_llm_provider: LlmProviderName = Field(default="openai", alias="BENCHMARK_LLM_PROVIDER")
    benchmark_llm_model: str = Field(default="", alias="BENCHMARK_LLM_MODEL")
    benchmark_llm_reasoning_effort: str | None = Field(
        default=None, alias="BENCHMARK_LLM_REASONING_EFFORT"
    )
    benchmark_llm_temperature: float | None = Field(
        default=None, alias="BENCHMARK_LLM_TEMPERATURE"
    )
    benchmark_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="BENCHMARK_LLM_MAX_OUTPUT_TOKENS"
    )

    # --- Digest (platform-only; run-scoped daily summaries) ---
    digest_llm_provider: LlmProviderName = Field(default="openai", alias="DIGEST_LLM_PROVIDER")
    digest_llm_model: str = Field(default="", alias="DIGEST_LLM_MODEL")
    digest_llm_reasoning_effort: str | None = Field(default=None, alias="DIGEST_LLM_REASONING_EFFORT")
    digest_llm_temperature: float | None = Field(default=None, alias="DIGEST_LLM_TEMPERATURE")
    digest_llm_max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, alias="DIGEST_LLM_MAX_OUTPUT_TOKENS")

    # --- Timeouts ---
    llm_timeout_seconds: float = Field(default=60.0, alias="PLATFORM_LLM_TIMEOUT_SECONDS")
    generator_llm_timeout_seconds: float | None = Field(
        default=None, alias="GENERATOR_LLM_TIMEOUT_SECONDS"
    )
    reference_llm_timeout_seconds: float | None = Field(
        default=None, alias="REFERENCE_LLM_TIMEOUT_SECONDS"
    )
    benchmark_llm_timeout_seconds: float | None = Field(
        default=None, alias="BENCHMARK_LLM_TIMEOUT_SECONDS"
    )
    digest_llm_timeout_seconds: float | None = Field(default=None, alias="DIGEST_LLM_TIMEOUT_SECONDS")
    scoring_llm_timeout_seconds: float = Field(default=30.0, alias="SCORING_LLM_TIMEOUT_SECONDS")

    # --- Scoring (validator) ---
    scoring_llm_provider: LlmProviderName = Field(default="chutes", alias="SCORING_LLM_PROVIDER")
    scoring_llm_model: str = Field(default="", alias="SCORING_LLM_MODEL")
    scoring_llm_temperature: float | None = Field(default=None, alias="SCORING_LLM_TEMPERATURE")
    scoring_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="SCORING_LLM_MAX_OUTPUT_TOKENS"
    )
    scoring_llm_reasoning_effort: str | None = Field(
        default=None, alias="SCORING_LLM_REASONING_EFFORT"
    )

    # --- Content review (platform-only) ---
    content_review_llm_provider: LlmProviderName | None = Field(default=None, alias="CONTENT_REVIEW_LLM_PROVIDER")
    content_review_llm_model: str = Field(default="", alias="CONTENT_REVIEW_LLM_MODEL")
    content_review_llm_reasoning_effort: str | None = Field(
        default=None, alias="CONTENT_REVIEW_LLM_REASONING_EFFORT"
    )
    content_review_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="CONTENT_REVIEW_LLM_MAX_OUTPUT_TOKENS"
    )
    content_review_llm_timeout_seconds: float | None = Field(
        default=None, alias="CONTENT_REVIEW_LLM_TIMEOUT_SECONDS"
    )

    # --- OpenAI / Chutes / DeSearch ---
    openai_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    desearch_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""), alias="DESEARCH_API_KEY"
    )
    desearch_base_url: str = Field(
        default="https://api.desearch.ai", alias="DESEARCH_BASE_URL"
    )

    chutes_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="CHUTES_API_KEY")

    # --- Concurrency limits ---
    openai_max_concurrent: int = Field(default=10, alias="OPENAI_MAX_CONCURRENT")
    vertex_max_concurrent: int = Field(default=8, alias="VERTEX_MAX_CONCURRENT")
    chutes_max_concurrent: int = Field(default=5, alias="CHUTES_MAX_CONCURRENT")
    desearch_max_concurrent: int = Field(default=5, alias="DESEARCH_MAX_CONCURRENT")

    # --- Validators ---
    @field_validator("desearch_api_key", mode="before")
    @classmethod
    def _unescape_env_var(cls, value: object) -> object:
        if isinstance(value, str):
            return value.replace("$$", "$")
        return value

    # --- Convenience accessors ---
    @property
    def openai_api_key_value(self) -> str:
        return self.openai_api_key.get_secret_value()

    @property
    def desearch_api_key_value(self) -> str:
        return self.desearch_api_key.get_secret_value()

    @property
    def chutes_api_key_value(self) -> str:
        return self.chutes_api_key.get_secret_value()


__all__ = ["LlmSettings", "DEFAULT_MAX_OUTPUT_TOKENS"]
