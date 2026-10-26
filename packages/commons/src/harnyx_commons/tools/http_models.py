"""Shared Pydantic shapes for tool-related HTTP endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, model_validator

from harnyx_commons.json_types import JsonObject
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.tools.decision_models import DecisionUsage
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata
from harnyx_miner_sdk.tools.http_models import (
    ToolBudgetDTO,
    ToolExecuteRequestDTO,
    ToolExecuteResponseDTO,
    ToolResultDTO,
    ToolUsageDTO,
)

# HTTP JSON contains an object; strict enclosing DTOs otherwise require a
# preconstructed dataclass. Parse once through the canonical usage type.
LlmUsageDTO = Annotated[LlmUsage, BeforeValidator(TypeAdapter(LlmUsage).validate_python)]


class CloudflareErrorBillingDTO(BaseModel):
    """Shared hosted failure contract: published-rate costs remain equal."""

    model_config = ConfigDict(extra="forbid", strict=True)
    actual_cost_provider: Literal["cloudflare"] = "cloudflare"
    actual_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    reference_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    cost_evidence: JsonObject | None = None
    usage: DecisionUsage | None = None
    llm_usage: LlmUsageDTO | None = None

    @model_validator(mode="after")
    def _equal_costs(self) -> CloudflareErrorBillingDTO:
        if self.actual_cost_usd != self.reference_cost_usd:
            raise ValueError("Cloudflare actual and reference costs must be equal")
        return self

    @classmethod
    def from_billing(cls, billing: ProviderBillingMetadata) -> CloudflareErrorBillingDTO:
        return cls.model_validate({
            "actual_cost_provider": billing.actual_cost_provider,
            "actual_cost_usd": billing.actual_cost_usd,
            "reference_cost_usd": billing.reference_cost_usd,
            "cost_evidence": billing.provider_usage,
            "usage": billing.usage,
            "llm_usage": billing.llm_usage,
        })

    def to_billing(self) -> ProviderBillingMetadata:
        return ProviderBillingMetadata(
            actual_cost_provider=self.actual_cost_provider, source="response_body",
            actual_cost_usd=self.actual_cost_usd, reference_cost_usd=self.reference_cost_usd,
            provider_usage=self.cost_evidence, usage=self.usage, llm_usage=self.llm_usage,
        )


__all__ = [
    "CloudflareErrorBillingDTO",
    "LlmUsageDTO",
    "ToolBudgetDTO",
    "ToolExecuteRequestDTO",
    "ToolExecuteResponseDTO",
    "ToolResultDTO",
    "ToolUsageDTO",
]
