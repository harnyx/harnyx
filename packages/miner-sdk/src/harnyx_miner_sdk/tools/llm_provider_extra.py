"""Provider-specific extras for miner ``llm_chat`` calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


def _normalize_non_empty_string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be a JSON array")
    if not value:
        raise ValueError(f"{label} must contain at least one entry")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label} entries must be strings")
        provider_name = item.strip()
        if not provider_name:
            raise ValueError(f"{label} entries must be non-empty")
        normalized.append(provider_name)
    return tuple(normalized)


class OpenRouterProviderSelection(BaseModel):
    """OpenRouter provider selection accepted by miner ``llm_chat``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    only: tuple[str, ...] | None = None
    allow_fallbacks: bool | None = None

    @field_validator("only", mode="before")
    @classmethod
    def _normalize_only(cls, value: object) -> tuple[str, ...]:
        return _normalize_non_empty_string_sequence(value, label="OpenRouter provider.only")

    @model_validator(mode="after")
    def _validate_provider_preference(self) -> OpenRouterProviderSelection:
        if self.only is None and self.allow_fallbacks is None:
            raise ValueError("OpenRouter provider_extra.provider must include only or allow_fallbacks")
        return self

    def to_provider_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.only is not None:
            payload["only"] = list(self.only)
        if self.allow_fallbacks is not None:
            payload["allow_fallbacks"] = self.allow_fallbacks
        return payload


class OpenRouterExtra(BaseModel):
    """Provider-specific extras accepted only for ``provider=\"openrouter\"``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: OpenRouterProviderSelection

    def to_request_extra(self) -> dict[str, Any]:
        return {"provider": self.provider.to_provider_payload()}


class AiGatewayProviderSelection(BaseModel):
    """AI Gateway provider selection accepted by miner ``llm_chat``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    only: tuple[str, ...] = Field(min_length=1)

    @field_validator("only", mode="before")
    @classmethod
    def _normalize_only(cls, value: object) -> tuple[str, ...]:
        return _normalize_non_empty_string_sequence(value, label="AI Gateway provider.only")

    def to_provider_payload(self) -> dict[str, Any]:
        return {"only": list(self.only)}


class AiGatewayOptionsGateway(BaseModel):
    """AI Gateway ``providerOptions.gateway`` payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    only: tuple[str, ...] = Field(min_length=1)

    @field_validator("only", mode="before")
    @classmethod
    def _normalize_only(cls, value: object) -> tuple[str, ...]:
        return _normalize_non_empty_string_sequence(
            value,
            label="AI Gateway providerOptions.gateway.only",
        )

    def to_gateway_payload(self) -> dict[str, Any]:
        return {"only": list(self.only)}


class AiGatewayProviderOptions(BaseModel):
    """AI Gateway ``providerOptions`` payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    gateway: AiGatewayOptionsGateway

    def to_provider_options_payload(self) -> dict[str, Any]:
        return {"gateway": self.gateway.to_gateway_payload()}


class AiGatewayExtra(BaseModel):
    """Provider-specific extras accepted only for ``provider=\"ai_gateway\"``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: AiGatewayProviderSelection | None = None
    provider_options: AiGatewayProviderOptions | None = Field(default=None, alias="providerOptions")

    @model_validator(mode="after")
    def _validate_selection_shape(self) -> AiGatewayExtra:
        if self.provider is None and self.provider_options is None:
            raise ValueError("AI Gateway provider_extra must include provider or providerOptions")
        provider_only = self.provider.only if self.provider is not None else None
        options_only = (
            self.provider_options.gateway.only
            if self.provider_options is not None
            else None
        )
        if provider_only is not None and options_only is not None and provider_only != options_only:
            raise ValueError("AI Gateway provider and providerOptions.gateway selections must match")
        return self

    def to_request_extra(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.provider is not None:
            payload["provider"] = self.provider.to_provider_payload()
        if self.provider_options is not None:
            payload["providerOptions"] = self.provider_options.to_provider_options_payload()
        return payload


_OPENROUTER_EXTRA_ADAPTER = TypeAdapter(OpenRouterExtra)
_AI_GATEWAY_EXTRA_ADAPTER = TypeAdapter(AiGatewayExtra)


ProviderExtra = OpenRouterExtra | AiGatewayExtra


def validate_provider_extra(*, provider: str, provider_extra: object) -> ProviderExtra | None:
    if provider_extra is None:
        return None
    if provider == "openrouter":
        return _OPENROUTER_EXTRA_ADAPTER.validate_python(provider_extra)
    if provider == "ai_gateway":
        return _AI_GATEWAY_EXTRA_ADAPTER.validate_python(provider_extra)
    raise ValueError(f"provider_extra is not supported for provider {provider!r}")


__all__ = [
    "AiGatewayExtra",
    "AiGatewayOptionsGateway",
    "AiGatewayProviderOptions",
    "AiGatewayProviderSelection",
    "OpenRouterExtra",
    "OpenRouterProviderSelection",
    "ProviderExtra",
    "validate_provider_extra",
]
