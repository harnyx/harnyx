"""Configuration helpers for validator runtime wiring."""

from __future__ import annotations

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.observability import ObservabilitySettings
from harnyx_commons.config.platform_api import PlatformApiSettings
from harnyx_commons.config.sandbox import SandboxPullPolicy, SandboxSettings
from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_commons.config.vertex import VertexSettings

_DEFAULT_VALIDATOR_SANDBOX_IMAGE = "harnyx/harnyx-subnet-sandbox:finney"


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
        alias="SANDBOX_IMAGE",
    )
    sandbox_network: str | None = Field(default="harnyx-sandbox-net", alias="SANDBOX_NETWORK")
    sandbox_pull_policy: SandboxPullPolicy = Field(
        default="always",
        alias="SANDBOX_PULL_POLICY",
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
        env_ignore_empty=True,
    )

    # --- Server ---
    rpc_listen_host: str = Field(
        default="0.0.0.0",  # noqa: S104
        alias="VALIDATOR_HOST",
    )
    rpc_port: int = Field(
        default=8100,
        alias="VALIDATOR_PORT",
    )

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
        logger = logging.getLogger("harnyx_validator.settings")
        logger.info("validator settings loaded: %r", instance)
        return instance


__all__ = ["Settings", "SubtensorSettings"]
