"""Configuration helpers for validator runtime wiring."""

from __future__ import annotations

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from caster_commons.config.llm import LlmSettings
from caster_commons.config.observability import ObservabilitySettings
from caster_commons.config.platform_api import PlatformApiSettings
from caster_commons.config.sandbox import SandboxSettings
from caster_commons.config.subtensor import SubtensorSettings
from caster_commons.config.vertex import VertexSettings


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
    rpc_public_host: str = Field(
        default="host.docker.internal", alias="CASTER_VALIDATOR_PUBLIC_HOST"
    )
    rpc_port: int = Field(default=8100, alias="CASTER_VALIDATOR_PORT")

    # --- Component settings ---
    llm: LlmSettings = Field(default_factory=LlmSettings)
    vertex: VertexSettings = Field(default_factory=VertexSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
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
    def openai_api_key_value(self) -> str:
        return self.llm.openai_api_key_value

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
