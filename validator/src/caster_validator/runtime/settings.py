"""Configuration helpers for validator runtime wiring."""

from __future__ import annotations

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from caster_commons.config.llm import LlmSettings
from caster_commons.config.observability import ObservabilitySettings
from caster_commons.config.platform_api import PlatformApiSettings
from caster_commons.config.sandbox import SandboxPullPolicy, SandboxSettings
from caster_commons.config.subtensor import SubtensorSettings
from caster_commons.config.vertex import VertexSettings

_DEFAULT_VALIDATOR_SANDBOX_IMAGE = "castersubnet/caster-subnet-sandbox:finney"


class _ValidatorSandboxEnv(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    sandbox_image: str = Field(
        default=_DEFAULT_VALIDATOR_SANDBOX_IMAGE,
        alias="CASTER_SANDBOX_IMAGE",
    )
    sandbox_network: str | None = Field(default="caster-sandbox-net", alias="CASTER_SANDBOX_NETWORK")
    sandbox_pull_policy: SandboxPullPolicy = Field(
        default="always",
        alias="CASTER_SANDBOX_PULL_POLICY",
    )


def load_validator_sandbox_settings() -> SandboxSettings:
    env = _ValidatorSandboxEnv()
    return SandboxSettings.model_construct(
        sandbox_image=env.sandbox_image,
        sandbox_network=env.sandbox_network,
        sandbox_pull_policy=env.sandbox_pull_policy,
    )


class Settings(BaseSettings):
    """Validator runtime configuration resolved from the environment.

    Only includes genuinely configurable settings - internal defaults live
    in their respective domain modules (sandbox.py, worker.py, etc.).
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    # --- Server ---
    rpc_listen_host: str = Field(default="0.0.0.0", alias="CASTER_VALIDATOR_HOST")  # noqa: S104
    rpc_port: int = Field(default=8100, alias="CASTER_VALIDATOR_PORT")

    # --- Component settings ---
    llm: LlmSettings = Field(default_factory=LlmSettings)
    vertex: VertexSettings = Field(default_factory=VertexSettings)
    sandbox: SandboxSettings = Field(default_factory=load_validator_sandbox_settings)
    platform_api: PlatformApiSettings = Field(default_factory=PlatformApiSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    subtensor: SubtensorSettings = Field(default_factory=SubtensorSettings)

    @property
    def desearch_api_key_value(self) -> str:
        return self.llm.desearch_api_key_value

    @property
    def chutes_api_key_value(self) -> str:
        return self.llm.chutes_api_key_value

    @property
    def gcp_sa_credential_b64_value(self) -> str:
        return self.vertex.gcp_sa_credential_b64_value

    # --- Loader ---
    @classmethod
    def load(cls) -> Settings:
        instance = cls()
        logger = logging.getLogger("caster_validator.settings")
        logger.info("validator settings loaded: %r", instance)
        return instance


__all__ = ["Settings", "SubtensorSettings"]
