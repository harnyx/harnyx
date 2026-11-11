"""Internal provider billing evidence for runtime tool settlement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Generic, Literal, TypeVar, cast

from pydantic import TypeAdapter

from harnyx_commons.json_types import JsonObject
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.tools.decision_models import DecisionUsage

# Providers whose accepted inference costs can be completed by hosted receipts.
RECOVERABLE_INFERENCE_TOOLS = ("llm_chat", "decision_query", "rerank")
RECOVERABLE_INFERENCE_PROVIDERS = ("openrouter", "cloudflare", "jev", "reax")

ProviderBillingSource = Literal[
    "response_body",
    "response_headers",
    "response_results",
    "request_body",
    "missing_provider_metadata",
]

TSearchResponse = TypeVar("TSearchResponse", covariant=True)
_LLM_USAGE = TypeAdapter(LlmUsage)
ReferenceCostSource = Literal["reported_charge", "usage_pricing", "timeout_zero"]
_REFERENCE_COST_SOURCE = TypeAdapter(ReferenceCostSource)


@dataclass(frozen=True, slots=True)
class OpenRouterAccounting:
    """Complete accounting meaning established at the OpenRouter execution boundary."""

    inference: Literal["not_dispatched", "attempted", "unresolved"]
    actual_cost_usd: float | None
    reference_cost_usd: float | None
    llm_usage: LlmUsage | None = None
    decision_usage: DecisionUsage | None = None
    reference_cost_source: ReferenceCostSource | None = None

    def __post_init__(self) -> None:
        if self.inference not in ("not_dispatched", "attempted", "unresolved"):
            raise ValueError("OpenRouter accounting requires explicit inference classification")
        for amount in (self.actual_cost_usd, self.reference_cost_usd):
            if amount is not None and (
                not isinstance(amount, int | float) or isinstance(amount, bool) or not isfinite(amount) or amount < 0
            ):
                raise ValueError("OpenRouter accounting amounts must be finite and nonnegative")
        if self.inference == "not_dispatched" and (
            self.actual_cost_usd != 0
            or self.reference_cost_usd != 0
            or self.llm_usage is not None
            or self.decision_usage is not None
        ):
            raise ValueError("undispatched OpenRouter call requires zero costs and no usage")
        if (
            self.reference_cost_source is None and self.reference_cost_usd is not None
            and self.inference != "not_dispatched"
        ):
            object.__setattr__(self, "reference_cost_source", "reported_charge")
        if self.reference_cost_source is not None:
            _REFERENCE_COST_SOURCE.validate_python(self.reference_cost_source)
            if self.reference_cost_usd is None:
                raise ValueError("OpenRouter reference source requires a reference amount")
        if self.reference_cost_source == "timeout_zero" and self.reference_cost_usd != 0:
            raise ValueError("OpenRouter timeout-zero reference must be zero")
        if self.reference_cost_source == "usage_pricing" and (
            self.inference != "attempted"
            or self.llm_usage is None
            or self.llm_usage.prompt_tokens is None
            or self.llm_usage.completion_tokens is None
        ):
            raise ValueError("OpenRouter usage-priced reference requires observed complete token usage")
        if (self.reference_cost_usd is not None and self.actual_cost_usd is None
                and self.reference_cost_source not in ("usage_pricing", "timeout_zero")):
            raise ValueError("known OpenRouter reference requires an observed actual charge")
        if (self.inference == "unresolved" and self.reference_cost_usd is not None
                and self.reference_cost_source != "timeout_zero"):
            raise ValueError("unresolved OpenRouter inference cannot have known reference cost")

    @property
    def reference_cost_unavailable(self) -> bool:
        return self.reference_cost_usd is None

    @property
    def counts_as_inference(self) -> bool:
        return self.inference != "not_dispatched"

    @classmethod
    def from_fields(
        cls,
        *,
        actual_cost_usd: float | None,
        reference_cost_usd: float | None,
        evidence: JsonObject | None,
        llm_usage: LlmUsage | None = None,
        decision_usage: DecisionUsage | None = None,
    ) -> OpenRouterAccounting:
        inference = (evidence or {}).get("inference")
        if inference not in ("not_dispatched", "attempted", "unresolved"):
            raise ValueError("OpenRouter accounting requires explicit inference classification")
        raw_decision_usage = (evidence or {}).get("decision_usage")
        recorded_decision_usage = (
            None if raw_decision_usage is None else DecisionUsage.model_validate(raw_decision_usage)
        )
        if (
            decision_usage is not None
            and recorded_decision_usage is not None
            and decision_usage != recorded_decision_usage
        ):
            raise ValueError("OpenRouter accounting has conflicting decision usage evidence")
        raw_usage = (evidence or {}).get("token_usage")
        recorded_usage = None if raw_usage is None else _LLM_USAGE.validate_python(raw_usage)
        if llm_usage is not None and recorded_usage is not None and llm_usage != recorded_usage:
            raise ValueError("OpenRouter accounting has conflicting usage evidence")
        return cls(
            inference=cast(Literal["not_dispatched", "attempted", "unresolved"], inference),
            actual_cost_usd=actual_cost_usd,
            reference_cost_usd=reference_cost_usd,
            llm_usage=llm_usage if llm_usage is not None else recorded_usage,
            decision_usage=decision_usage if decision_usage is not None else recorded_decision_usage,
            reference_cost_source=(
                None if (evidence or {}).get("reference_cost_source") is None
                else _REFERENCE_COST_SOURCE.validate_python((evidence or {})["reference_cost_source"])
            ),
        )

    @classmethod
    def from_billing(cls, billing: ProviderBillingMetadata) -> OpenRouterAccounting:
        if billing.actual_cost_provider != "openrouter":
            raise ValueError("OpenRouter chat accounting requires OpenRouter billing")
        fact = cls.from_fields(
            actual_cost_usd=billing.actual_cost_usd,
            reference_cost_usd=billing.reference_cost_usd,
            evidence=billing.provider_usage,
            llm_usage=billing.llm_usage,
            decision_usage=billing.usage,
        )
        if fact.reference_cost_unavailable != billing.reference_cost_unavailable:
            raise ValueError("OpenRouter accounting has conflicting reference availability")
        return fact

    def to_evidence(self, raw: JsonObject | None = None) -> JsonObject:
        evidence = dict(raw or {})
        evidence["inference"] = self.inference
        evidence.pop("reference_cost_source", None)
        if self.reference_cost_source is not None:
            evidence["reference_cost_source"] = self.reference_cost_source
        evidence.pop("decision_usage", None)
        if self.decision_usage is not None:
            evidence["decision_usage"] = self.decision_usage.model_dump(mode="json", exclude_none=True)
        evidence.pop("token_usage", None)
        if self.llm_usage is not None:
            evidence["token_usage"] = asdict(self.llm_usage)
        return evidence

    def to_billing(
        self,
        evidence: JsonObject | None = None,
        *,
        source: ProviderBillingSource = "response_body",
    ) -> ProviderBillingMetadata:
        return ProviderBillingMetadata(
            actual_cost_provider="openrouter",
            source=source,
            actual_cost_usd=self.actual_cost_usd,
            reference_cost_usd=self.reference_cost_usd,
            reference_cost_unavailable=self.reference_cost_unavailable,
            llm_usage=self.llm_usage,
            usage=self.decision_usage,
            provider_usage=self.to_evidence(evidence),
        )


@dataclass(frozen=True, slots=True)
class ProviderBillingMetadata:
    actual_cost_provider: str
    source: ProviderBillingSource
    actual_cost_usd: float | None = None
    billable_units: int | None = None
    provider_request_id: str | None = None
    usage_count: int | None = None
    service: str | None = None
    currency: str | None = None
    usage: DecisionUsage | None = None
    llm_usage: LlmUsage | None = None
    provider_usage: JsonObject | None = None
    reference_cost_usd: float | None = None
    reference_cost_unavailable: bool = False


def normalize_llm_timeout(
    billing: ProviderBillingMetadata | None, *, provider: str
) -> ProviderBillingMetadata:
    """Resolve unbilled chat timeouts for comparison without inventing a financial bill."""
    if billing is not None and billing.actual_cost_usd is not None:
        return billing
    if provider == "openrouter":
        fact = (OpenRouterAccounting("unresolved", None, None) if billing is None
                else OpenRouterAccounting.from_billing(billing))
        return replace(fact, reference_cost_usd=0.0, reference_cost_source="timeout_zero").to_billing(
            None if billing is None else billing.provider_usage,
            source="missing_provider_metadata" if billing is None else billing.source,
        )
    billing = billing or ProviderBillingMetadata(provider, "missing_provider_metadata")
    return replace(
        billing, reference_cost_usd=0.0, reference_cost_unavailable=False,
        provider_usage={**(billing.provider_usage or {}), "reference_cost_source": "timeout_zero"},
    )


@dataclass(frozen=True, slots=True)
class SearchProviderResult(Generic[TSearchResponse]):
    response: TSearchResponse
    billing: ProviderBillingMetadata


def billing_evidence_payload(billing: ProviderBillingMetadata | None) -> JsonObject | None:
    if billing is None:
        return None
    values = asdict(billing)
    if not billing.reference_cost_unavailable:
        values.pop("reference_cost_unavailable")
    if billing.usage is not None:
        values["usage"] = billing.usage.model_dump(mode="json", exclude_none=True)
    return {key: value for key, value in values.items() if value is not None and _is_json_value(value)}


def failure_billing(tool: str, billing: ProviderBillingMetadata | None) -> ProviderBillingMetadata | None:
    """Keep supported failure evidence without treating uncertain search costs as free."""
    if tool in RECOVERABLE_INFERENCE_TOOLS and billing is not None and billing.actual_cost_provider == "openrouter":
        return billing
    if tool == "llm_chat" and billing is not None and billing.actual_cost_provider == "cloudflare":
        return billing
    if tool == "decision_query":
        return billing
    if (
        tool in {"search_web", "fetch_page"}
        and billing is not None
        and billing.actual_cost_provider == "deepseek"
        and billing.actual_cost_usd == 0.0
    ):
        return billing
    return None


def billing_usage_payload(billing: ProviderBillingMetadata | None) -> JsonObject | None:
    """Preserve complete chat usage without changing the Decision usage contract."""
    if billing is None:
        return None
    if billing.llm_usage is not None:
        return {"usage": asdict(billing.llm_usage)}
    if billing.usage is not None:
        return {"usage": billing.usage.model_dump(mode="json", exclude_none=True)}
    return None


def _is_json_value(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool | list | dict)


__all__ = [
    "OpenRouterAccounting",
    "ProviderBillingMetadata",
    "RECOVERABLE_INFERENCE_PROVIDERS",
    "RECOVERABLE_INFERENCE_TOOLS",
    "ProviderBillingSource",
    "SearchProviderResult",
    "billing_evidence_payload",
    "billing_usage_payload",
    "failure_billing",
    "normalize_llm_timeout",
]
