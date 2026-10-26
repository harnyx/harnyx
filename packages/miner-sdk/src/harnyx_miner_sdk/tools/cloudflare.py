"""Cloudflare's account-scoped credential, uploaded as one atomic value."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, SecretStr, StringConstraints, field_validator

CLOUDFLARE_CHAT_MODELS = (
    "@cf/openai/gpt-oss-20b",
    "@cf/openai/gpt-oss-120b",
    "@cf/qwen/qwen3.8-27b",
    "@cf/zai-org/glm-5.2",
    "@cf/zai-org/glm-5.3-flash",
    "@cf/moonshotai/kimi-k2.6",
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/google/gemma-4-26b-a4b-it",
)


class CloudflareCredential(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    account_id: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[a-fA-F0-9]{32}$")]
    api_token: SecretStr

    @field_validator("api_token")
    @classmethod
    def _token(cls, value: SecretStr) -> SecretStr:
        token = value.get_secret_value().strip()
        if not token:
            raise ValueError("Cloudflare API token must be non-empty")
        return SecretStr(token)

    def to_credential_value(self) -> str:
        """Serialize only for upload/encryption; never use for logs or display."""
        import json

        return json.dumps({"account_id": self.account_id, "api_token": self.api_token.get_secret_value()})
