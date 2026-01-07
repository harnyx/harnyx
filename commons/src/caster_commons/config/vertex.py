"""Vertex AI configuration shared across services."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class VertexSettings(BaseSettings):
    """Project, region, and credential settings for Vertex AI."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    gcp_project_id: str | None = Field(default=None, alias="GCP_PROJECT_ID")
    gcp_location: str | None = Field(default=None, alias="GCP_LOCATION")
    vertex_maas_gcp_location: str = Field(default="us-central1", alias="VERTEX_MAAS_GCP_LOCATION")
    vertex_timeout_seconds: float = Field(default=60.0, alias="VERTEX_TIMEOUT_SECONDS")
    gcp_service_account_credential_b64: SecretStr = Field(
        default_factory=lambda: SecretStr(""), alias="GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64"
    )

    @property
    def gcp_sa_credential_b64_value(self) -> str:
        return self.gcp_service_account_credential_b64.get_secret_value()


__all__ = ["VertexSettings"]
