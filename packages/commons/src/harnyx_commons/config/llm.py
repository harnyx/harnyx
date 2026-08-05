"""LLM-related configuration shared by platform and validator."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Annotated, Literal, cast

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from harnyx_commons.llm.provider_types import LlmProviderName
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.routing import LlmModelProviderOverrides, parse_llm_model_provider_overrides
from harnyx_commons.tools.embedding_models import EmbeddingProviderName

DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_SCORING_MAX_OUTPUT_TOKENS = 20480
DEFAULT_SCORING_LLM_RETRY_ATTEMPTS = 6
DEFAULT_SCORING_LLM_RETRY_INITIAL_MS = 30_000
DEFAULT_SCORING_LLM_RETRY_MAX_MS = 300_000
DEFAULT_SCORING_LLM_RETRY_JITTER = 0.2
DEFAULT_SIMILARITY_LLM_RETRY_ATTEMPTS = 1
DEFAULT_SIMILARITY_LLM_RETRY_INITIAL_MS = 0
DEFAULT_SIMILARITY_LLM_RETRY_MAX_MS = 0
DEFAULT_SIMILARITY_LLM_RETRY_JITTER = 0.0
_ENDPOINT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class OpenAiCompatibleNoAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["none"]


class OpenAiCompatibleBearerTokenEnvAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["bearer_token_env"]
    token_env: str = Field(min_length=1)

    @field_validator("token_env")
    @classmethod
    def _normalize_token_env(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("bearer token env name must be non-empty")
        return normalized


class OpenAiCompatibleGoogleIdTokenAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["google_id_token"]
    audience: str = Field(min_length=1)
    credential_source: Literal["adc", "service_account_json_b64_env"]
    credential_env: str | None = None

    @field_validator("audience", "credential_env")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("google ID token auth text fields must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_credential_env(self) -> OpenAiCompatibleGoogleIdTokenAuthConfig:
        if self.credential_source == "service_account_json_b64_env" and self.credential_env is None:
            raise ValueError("google_id_token auth requires credential_env for service_account_json_b64_env")
        if self.credential_source == "adc" and self.credential_env is not None:
            raise ValueError("google_id_token auth credential_env is only allowed for service_account_json_b64_env")
        return self


OpenAiCompatibleAuthConfig = Annotated[
    OpenAiCompatibleNoAuthConfig
    | OpenAiCompatibleBearerTokenEnvAuthConfig
    | OpenAiCompatibleGoogleIdTokenAuthConfig,
    Field(discriminator="type"),
]


class OpenAiCompatibleEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str = Field(min_length=1)
    base_url: AnyHttpUrl
    auth: OpenAiCompatibleAuthConfig
    timeout_seconds: float | None = None
    max_concurrent: int | None = None

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("OpenAI-compatible endpoint id must be non-empty")
        if not _ENDPOINT_ID_PATTERN.fullmatch(normalized):
            raise ValueError(
                "OpenAI-compatible endpoint id may contain only letters, numbers, underscore, dot, and dash"
            )
        return normalized

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("OpenAI-compatible endpoint timeout_seconds must be positive")
        return value

    @field_validator("max_concurrent")
    @classmethod
    def _validate_max_concurrent(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("OpenAI-compatible endpoint max_concurrent must be positive")
        return value


_OPENAI_COMPATIBLE_ENDPOINTS_ADAPTER = TypeAdapter(list[OpenAiCompatibleEndpointConfig])

SearchProviderName = Literal["desearch", "parallel", "firecrawl", "exa", "tavily"]
SEARCH_PROVIDER_NAMES: tuple[SearchProviderName, ...] = (
    "desearch", "parallel", "firecrawl", "exa", "tavily"
)
AiSearchProviderName = Literal["desearch", "parallel"]
AI_SEARCH_PROVIDER_NAMES: tuple[AiSearchProviderName, ...] = ("desearch", "parallel")


def parse_search_provider_name(raw: str | None) -> SearchProviderName:
    if raw is None:
        raise ValueError("search provider must be specified")
    value = raw.strip().lower()
    if value not in SEARCH_PROVIDER_NAMES:
        raise ValueError(f"search provider {value!r} is not supported")
    return cast(SearchProviderName, value)


def parse_ai_search_provider_name(raw: str | None) -> AiSearchProviderName:
    if raw is None:
        raise ValueError("AI search provider must be specified")
    value = raw.strip().lower()
    if value not in AI_SEARCH_PROVIDER_NAMES:
        raise ValueError(f"AI search provider {value!r} is not supported")
    return cast(AiSearchProviderName, value)


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
    tool_llm_provider: LlmProviderName = Field(default="chutes", alias="TOOL_LLM_PROVIDER")
    tool_embedding_provider: EmbeddingProviderName = Field(default="chutes", alias="TOOL_EMBEDDING_PROVIDER")
    search_provider: SearchProviderName | None = Field(default=None, alias="SEARCH_PROVIDER")

    # --- Generation / reference / benchmark ---
    generator_llm_provider: LlmProviderName = Field(default="chutes", alias="GENERATOR_LLM_PROVIDER")
    generator_llm_model: str = Field(default="", alias="GENERATOR_LLM_MODEL")
    generator_llm_reasoning_effort: str | None = Field(default=None, alias="GENERATOR_LLM_REASONING_EFFORT")
    generator_llm_temperature: float | None = Field(default=None, alias="GENERATOR_TEMPERATURE")
    generator_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="GENERATOR_LLM_MAX_OUTPUT_TOKENS"
    )

    reference_llm_provider: LlmProviderName = Field(default="chutes", alias="REFERENCE_LLM_PROVIDER")
    reference_llm_model: str = Field(default="", alias="REFERENCE_LLM_MODEL")
    reference_llm_reasoning_effort: str | None = Field(default=None, alias="REFERENCE_LLM_REASONING_EFFORT")
    reference_llm_temperature: float | None = Field(default=None, alias="REFERENCE_TEMPERATURE")
    reference_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="REFERENCE_LLM_MAX_OUTPUT_TOKENS"
    )

    benchmark_llm_provider: LlmProviderName = Field(default="chutes", alias="BENCHMARK_LLM_PROVIDER")
    benchmark_llm_model: str = Field(default="", alias="BENCHMARK_LLM_MODEL")
    benchmark_llm_reasoning_effort: str | None = Field(default=None, alias="BENCHMARK_LLM_REASONING_EFFORT")
    benchmark_llm_temperature: float | None = Field(default=None, alias="BENCHMARK_LLM_TEMPERATURE")
    benchmark_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="BENCHMARK_LLM_MAX_OUTPUT_TOKENS"
    )

    # --- Digest (platform-only; run-scoped daily summaries) ---
    digest_llm_provider: LlmProviderName = Field(default="chutes", alias="DIGEST_LLM_PROVIDER")
    digest_llm_model: str = Field(default="", alias="DIGEST_LLM_MODEL")
    digest_llm_reasoning_effort: str | None = Field(default=None, alias="DIGEST_LLM_REASONING_EFFORT")
    digest_llm_temperature: float | None = Field(default=None, alias="DIGEST_LLM_TEMPERATURE")
    digest_llm_max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, alias="DIGEST_LLM_MAX_OUTPUT_TOKENS")
    llm_model_provider_overrides_json: str | None = Field(default=None, alias="LLM_MODEL_PROVIDER_OVERRIDES_JSON")
    openai_compatible_endpoints_json: str | None = Field(default=None, alias="LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON")
    openrouter_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="OPENROUTER_API_KEY")
    ai_gateway_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="AI_GATEWAY_API_KEY")

    # --- Timeouts ---
    llm_timeout_seconds: float = Field(default=300.0, alias="PLATFORM_LLM_TIMEOUT_SECONDS")
    generator_llm_timeout_seconds: float | None = Field(default=None, alias="GENERATOR_LLM_TIMEOUT_SECONDS")
    reference_llm_timeout_seconds: float | None = Field(default=None, alias="REFERENCE_LLM_TIMEOUT_SECONDS")
    benchmark_llm_timeout_seconds: float | None = Field(default=None, alias="BENCHMARK_LLM_TIMEOUT_SECONDS")
    digest_llm_timeout_seconds: float | None = Field(default=None, alias="DIGEST_LLM_TIMEOUT_SECONDS")
    scoring_llm_timeout_seconds: float = Field(default=300.0, alias="SCORING_LLM_TIMEOUT_SECONDS")
    similarity_llm_timeout_seconds: float = Field(default=300.0, alias="SIMILARITY_LLM_TIMEOUT_SECONDS")

    # --- Scoring (validator) ---
    scoring_llm_provider: LlmProviderName = Field(default="chutes", alias="SCORING_LLM_PROVIDER")
    scoring_llm_temperature: float | None = Field(default=None, alias="SCORING_LLM_TEMPERATURE")
    scoring_llm_max_output_tokens: int = Field(
        default=DEFAULT_SCORING_MAX_OUTPUT_TOKENS, alias="SCORING_LLM_MAX_OUTPUT_TOKENS"
    )
    scoring_llm_retry_attempts: int = Field(
        default=DEFAULT_SCORING_LLM_RETRY_ATTEMPTS,
        alias="SCORING_LLM_RETRY_ATTEMPTS",
        ge=1,
    )
    scoring_llm_retry_initial_ms: int = Field(
        default=DEFAULT_SCORING_LLM_RETRY_INITIAL_MS,
        alias="SCORING_LLM_RETRY_INITIAL_MS",
        ge=0,
    )
    scoring_llm_retry_max_ms: int = Field(
        default=DEFAULT_SCORING_LLM_RETRY_MAX_MS,
        alias="SCORING_LLM_RETRY_MAX_MS",
        ge=0,
    )
    scoring_llm_retry_jitter: float = Field(
        default=DEFAULT_SCORING_LLM_RETRY_JITTER,
        alias="SCORING_LLM_RETRY_JITTER",
        ge=0.0,
        le=1.0,
    )

    # --- Similarity / duplicate preflight (validator) ---
    similarity_llm_provider: LlmProviderName = Field(default="chutes", alias="SIMILARITY_LLM_PROVIDER")
    similarity_llm_temperature: float | None = Field(default=None, alias="SIMILARITY_LLM_TEMPERATURE")
    similarity_llm_max_output_tokens: int = Field(
        default=DEFAULT_SCORING_MAX_OUTPUT_TOKENS, alias="SIMILARITY_LLM_MAX_OUTPUT_TOKENS"
    )
    similarity_llm_model_override: str | None = Field(default=None, alias="SIMILARITY_LLM_MODEL_OVERRIDE")
    similarity_llm_retry_attempts: int = Field(
        default=DEFAULT_SIMILARITY_LLM_RETRY_ATTEMPTS,
        alias="SIMILARITY_LLM_RETRY_ATTEMPTS",
        ge=1,
    )
    similarity_llm_retry_initial_ms: int = Field(
        default=DEFAULT_SIMILARITY_LLM_RETRY_INITIAL_MS,
        alias="SIMILARITY_LLM_RETRY_INITIAL_MS",
        ge=0,
    )
    similarity_llm_retry_max_ms: int = Field(
        default=DEFAULT_SIMILARITY_LLM_RETRY_MAX_MS,
        alias="SIMILARITY_LLM_RETRY_MAX_MS",
        ge=0,
    )
    similarity_llm_retry_jitter: float = Field(
        default=DEFAULT_SIMILARITY_LLM_RETRY_JITTER,
        alias="SIMILARITY_LLM_RETRY_JITTER",
        ge=0.0,
        le=1.0,
    )

    # --- Content review (platform-only) ---
    content_review_llm_provider: LlmProviderName | None = Field(default=None, alias="CONTENT_REVIEW_LLM_PROVIDER")
    content_review_llm_model: str = Field(default="", alias="CONTENT_REVIEW_LLM_MODEL")
    content_review_llm_reasoning_effort: str | None = Field(default=None, alias="CONTENT_REVIEW_LLM_REASONING_EFFORT")
    content_review_llm_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS, alias="CONTENT_REVIEW_LLM_MAX_OUTPUT_TOKENS"
    )
    content_review_llm_timeout_seconds: float | None = Field(default=None, alias="CONTENT_REVIEW_LLM_TIMEOUT_SECONDS")

    # --- Chutes / web retrieval providers ---
    desearch_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="DESEARCH_API_KEY")
    parallel_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="PARALLEL_API_KEY")
    firecrawl_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="FIRECRAWL_API_KEY")
    exa_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="EXA_API_KEY")
    tavily_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="TAVILY_API_KEY")
    parallel_base_url: str = Field(default="https://api.parallel.ai", alias="PARALLEL_BASE_URL")

    chutes_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="CHUTES_API_KEY")

    # --- Concurrency limits ---
    vertex_max_concurrent: int = Field(default=100, alias="VERTEX_MAX_CONCURRENT")
    bedrock_max_concurrent: int = Field(default=100, alias="BEDROCK_MAX_CONCURRENT")
    chutes_max_concurrent: int = Field(default=100, alias="CHUTES_MAX_CONCURRENT")
    desearch_max_concurrent: int = Field(default=100, alias="DESEARCH_MAX_CONCURRENT")
    parallel_max_concurrent: int = Field(default=100, alias="PARALLEL_MAX_CONCURRENT")
    firecrawl_max_concurrent: int = Field(default=100, alias="FIRECRAWL_MAX_CONCURRENT")
    exa_max_concurrent: int = Field(default=100, alias="EXA_MAX_CONCURRENT")
    tavily_max_concurrent: int = Field(default=100, alias="TAVILY_MAX_CONCURRENT")

    # --- Validators ---
    @field_validator("desearch_api_key", mode="before")
    @classmethod
    def _unescape_env_var(cls, value: object) -> object:
        if isinstance(value, str):
            return value.replace("$$", "$")
        return value

    # --- Convenience accessors ---
    @property
    def desearch_api_key_value(self) -> str:
        return self.desearch_api_key.get_secret_value()

    @property
    def parallel_api_key_value(self) -> str:
        return self.parallel_api_key.get_secret_value()

    @property
    def firecrawl_api_key_value(self) -> str:
        return self.firecrawl_api_key.get_secret_value()

    @property
    def exa_api_key_value(self) -> str:
        return self.exa_api_key.get_secret_value()

    @property
    def tavily_api_key_value(self) -> str:
        return self.tavily_api_key.get_secret_value()

    @property
    def chutes_api_key_value(self) -> str:
        return self.chutes_api_key.get_secret_value()

    @property
    def openrouter_api_key_value(self) -> str:
        return self.openrouter_api_key.get_secret_value().strip()

    @property
    def ai_gateway_api_key_value(self) -> str:
        return self.ai_gateway_api_key.get_secret_value().strip()

    @property
    def similarity_llm_model_override_value(self) -> str | None:
        if self.similarity_llm_model_override is None:
            return None
        normalized = self.similarity_llm_model_override.strip()
        return normalized or None

    @property
    def scoring_llm_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            attempts=self.scoring_llm_retry_attempts,
            initial_ms=self.scoring_llm_retry_initial_ms,
            max_ms=self.scoring_llm_retry_max_ms,
            jitter=self.scoring_llm_retry_jitter,
        )

    @property
    def similarity_llm_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            attempts=self.similarity_llm_retry_attempts,
            initial_ms=self.similarity_llm_retry_initial_ms,
            max_ms=self.similarity_llm_retry_max_ms,
            jitter=self.similarity_llm_retry_jitter,
        )

    @property
    def llm_model_provider_overrides(self) -> LlmModelProviderOverrides:
        return parse_llm_model_provider_overrides(
            self.llm_model_provider_overrides_json,
            custom_openai_compatible_endpoint_ids=set(self.openai_compatible_endpoints),
        )

    @property
    def openai_compatible_endpoints(self) -> Mapping[str, OpenAiCompatibleEndpointConfig]:
        return parse_openai_compatible_endpoints_json(self.openai_compatible_endpoints_json)


def parse_openai_compatible_endpoints_json(raw: str | None) -> Mapping[str, OpenAiCompatibleEndpointConfig]:
    if raw is None:
        return {}
    normalized_raw = raw.strip()
    if not normalized_raw:
        return {}
    try:
        payload = json.loads(normalized_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON must be valid JSON") from exc
    endpoints = _OPENAI_COMPATIBLE_ENDPOINTS_ADAPTER.validate_python(payload)
    result: dict[str, OpenAiCompatibleEndpointConfig] = {}
    for endpoint in endpoints:
        if endpoint.id in result:
            raise ValueError(f"LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON endpoint id {endpoint.id!r} is duplicated")
        result[endpoint.id] = endpoint
    return result


__all__ = [
    "LlmSettings",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_SCORING_MAX_OUTPUT_TOKENS",
    "OpenAiCompatibleAuthConfig",
    "OpenAiCompatibleBearerTokenEnvAuthConfig",
    "OpenAiCompatibleEndpointConfig",
    "OpenAiCompatibleGoogleIdTokenAuthConfig",
    "OpenAiCompatibleNoAuthConfig",
    "SEARCH_PROVIDER_NAMES",
    "SearchProviderName",
    "parse_openai_compatible_endpoints_json",
    "parse_search_provider_name",
]
