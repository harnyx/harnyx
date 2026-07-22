"""Provider-boundary LLM cost settlement helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.adapter import canonical_model_for_provider_model
from harnyx_commons.llm.pricing import (
    MINER_TOOL_LLM_PRICING,
    price_miner_llm,
    price_static_llm_model,
)
from harnyx_commons.llm.provider_types import AI_GATEWAY_PROVIDER, OPENROUTER_PROVIDER
from harnyx_commons.llm.schema import LlmResponse, LlmUsage
from harnyx_commons.llm.tool_models import MinerSelectedLlmProviderName


@dataclass(frozen=True, slots=True)
class SettledLlmCost:
    cost_usd: float
    provider: str
    evidence: JsonObject


def with_settled_llm_cost(response: LlmResponse, cost: SettledLlmCost) -> LlmResponse:
    metadata = dict(response.metadata or {})
    metadata["actual_cost_usd"] = cost.cost_usd
    metadata["actual_cost_provider"] = cost.provider
    metadata["actual_cost_evidence"] = cost.evidence
    return replace(response, metadata=metadata)


def settled_cost_from_metadata(metadata: Mapping[str, object]) -> SettledLlmCost | None:
    single_response_cost = (
        normalized_provider_cost(metadata["actual_cost_usd"], field_name="actual_cost_usd")
        if "actual_cost_usd" in metadata
        else None
    )
    total_cost = (
        normalized_provider_cost(metadata["actual_cost_usd_total"], field_name="actual_cost_usd_total")
        if "actual_cost_usd_total" in metadata
        else None
    )
    selected_cost = total_cost if total_cost is not None else single_response_cost
    if selected_cost is None:
        return None

    provider = metadata.get("actual_cost_provider")
    if not isinstance(provider, str) or not provider.strip():
        return None

    evidence = metadata.get("actual_cost_evidence")
    if not isinstance(evidence, Mapping):
        return None

    evidence_payload = cast(JsonObject, dict(evidence))
    if total_cost is not None:
        evidence_payload = {
            "settlement_source": "retry_aggregate",
            "pricing_origin": "actual_cost_usd_total",
            "actual_cost_usd_total": total_cost,
            "final_response_actual_cost_usd": single_response_cost,
            "final_response_evidence": evidence_payload,
        }

    return SettledLlmCost(
        cost_usd=selected_cost,
        provider=provider.strip(),
        evidence=evidence_payload,
    )


def settled_response_cost(
    response: LlmResponse,
    *,
    provider: str,
    model: str,
) -> SettledLlmCost | None:
    normalized_provider = provider.strip()
    if not normalized_provider:
        return None

    if normalized_provider == OPENROUTER_PROVIDER:
        return _settled_openrouter_response_cost(response=response, model=model)
    if normalized_provider == AI_GATEWAY_PROVIDER:
        ai_gateway_cost = _ai_gateway_provider_returned_cost(response=response, model=model)
        if ai_gateway_cost is not None:
            return ai_gateway_cost

    return settled_static_llm_cost(
        provider=normalized_provider,
        model=model,
        usage=response.usage or LlmUsage(),
    )


def settled_static_llm_cost(
    *,
    provider: str,
    model: str,
    usage: LlmUsage,
    additional_evidence: JsonObject | None = None,
) -> SettledLlmCost | None:
    if _has_miner_static_pricing(provider=provider, model=model):
        evidence: JsonObject = {
            "settlement_source": "static_pricing",
            "pricing_origin": "miner_tool_llm_pricing",
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.prompt_tokens or 0,
            "completion_tokens": usage.completion_tokens or 0,
            "reasoning_tokens": usage.reasoning_tokens,
        }
        if additional_evidence is not None:
            evidence.update(additional_evidence)
        return SettledLlmCost(
            cost_usd=price_miner_llm(provider, model, usage),
            provider=provider,
            evidence=evidence,
        )

    canonical_model = canonical_model_for_provider_model(
        provider_name=provider,
        model=model,
    )
    cost_usd = price_static_llm_model(canonical_model, usage)
    if cost_usd is None:
        return None
    evidence = {
        "settlement_source": "static_pricing",
        "pricing_origin": "static_llm_pricing",
        "provider": provider,
        "model": model,
        "canonical_model": canonical_model,
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "reasoning_tokens": usage.reasoning_tokens,
    }
    if additional_evidence is not None:
        evidence.update(additional_evidence)
    return SettledLlmCost(
        cost_usd=cost_usd,
        provider=provider,
        evidence=evidence,
    )


@dataclass(frozen=True, slots=True)
class _OpenRouterUsageBilling:
    is_byok: bool | None
    is_byok_status: str
    cost_usd: float | None
    cost_status: str
    upstream_inference_cost_usd: float | None
    upstream_inference_cost_status: str


class _OpenRouterResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    usage: JsonValue | None = None


class _OpenRouterUsagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    is_byok: JsonValue | None = None
    cost: JsonValue | None = None
    cost_details: JsonValue | None = None


class _OpenRouterCostDetailsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    upstream_inference_cost: JsonValue | None = None


_STRICT_BOOL_ADAPTER = TypeAdapter(bool)


def _settled_openrouter_response_cost(
    *,
    response: LlmResponse,
    model: str,
) -> SettledLlmCost | None:
    billing = _openrouter_usage_billing(response)
    if (
        billing.is_byok is True
        and billing.cost_usd is not None
        and billing.upstream_inference_cost_usd is not None
    ):
        return SettledLlmCost(
            cost_usd=billing.cost_usd + billing.upstream_inference_cost_usd,
            provider=OPENROUTER_PROVIDER,
            evidence={
                "settlement_source": "provider_returned",
                "pricing_origin": "openrouter_usage_cost_and_upstream_inference_cost",
                "provider": OPENROUTER_PROVIDER,
                "model": model,
                "usage": _openrouter_usage_evidence(billing),
            },
        )
    if (
        billing.is_byok is False or billing.is_byok_status == "missing"
    ) and billing.cost_usd is not None:
        evidence: JsonObject = {
            "settlement_source": "provider_returned",
            "pricing_origin": "openrouter_usage_cost",
            "provider": OPENROUTER_PROVIDER,
            "model": model,
            "usage": _openrouter_usage_evidence(billing),
        }
        if billing.is_byok_status == "missing":
            evidence["is_byok_status"] = "missing"
        return SettledLlmCost(
            cost_usd=billing.cost_usd,
            provider=OPENROUTER_PROVIDER,
            evidence=evidence,
        )
    return settled_static_llm_cost(
        provider=OPENROUTER_PROVIDER,
        model=model,
        usage=response.usage or LlmUsage(),
        additional_evidence={
            "is_byok_status": billing.is_byok_status,
            "cost_status": billing.cost_status,
            "upstream_inference_cost_status": billing.upstream_inference_cost_status,
        },
    )


def _openrouter_usage_billing(response: LlmResponse) -> _OpenRouterUsageBilling:
    raw_response = (response.metadata or {}).get("raw_response")
    if raw_response is None:
        return _missing_openrouter_usage_billing()
    try:
        response_payload = _OpenRouterResponsePayload.model_validate(raw_response)
    except ValidationError:
        return _malformed_openrouter_usage_billing()
    if response_payload.usage is None:
        return _missing_openrouter_usage_billing()
    try:
        usage = _OpenRouterUsagePayload.model_validate(response_payload.usage)
    except ValidationError:
        return _malformed_openrouter_usage_billing()
    is_byok, is_byok_status = _normalized_openrouter_is_byok(usage.is_byok)
    cost_usd, cost_status = _normalized_openrouter_cost(
        usage.cost,
        field_name="OpenRouter usage.cost",
    )
    upstream_inference_cost_usd, upstream_inference_cost_status = (
        _normalized_openrouter_upstream_inference_cost(usage.cost_details)
    )
    return _OpenRouterUsageBilling(
        is_byok=is_byok,
        is_byok_status=is_byok_status,
        cost_usd=cost_usd,
        cost_status=cost_status,
        upstream_inference_cost_usd=upstream_inference_cost_usd,
        upstream_inference_cost_status=upstream_inference_cost_status,
    )


def _normalized_openrouter_is_byok(value: JsonValue | None) -> tuple[bool | None, str]:
    if value is None:
        return None, "missing"
    try:
        is_byok = _STRICT_BOOL_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None, "malformed"
    return is_byok, "valid"


def _normalized_openrouter_cost(
    value: JsonValue | None,
    *,
    field_name: str,
) -> tuple[float | None, str]:
    if value is None:
        return None, "missing"
    cost = normalized_provider_cost(
        value,
        field_name=field_name,
        strict=False,
    )
    return (cost, "valid") if cost is not None else (None, "malformed")


def _normalized_openrouter_upstream_inference_cost(
    value: JsonValue | None,
) -> tuple[float | None, str]:
    if value is None:
        return None, "missing"
    try:
        cost_details = _OpenRouterCostDetailsPayload.model_validate(value)
    except ValidationError:
        return None, "malformed"
    return _normalized_openrouter_cost(
        cost_details.upstream_inference_cost,
        field_name="OpenRouter usage.cost_details.upstream_inference_cost",
    )


def _openrouter_usage_evidence(billing: _OpenRouterUsageBilling) -> JsonObject:
    if billing.cost_usd is None:
        raise ValueError("complete OpenRouter usage billing is required for provider-returned evidence")
    usage: JsonObject = {
        "cost": billing.cost_usd,
    }
    if billing.is_byok is not None:
        usage["is_byok"] = billing.is_byok
    if billing.upstream_inference_cost_usd is not None:
        usage["cost_details"] = {
            "upstream_inference_cost": billing.upstream_inference_cost_usd,
        }
    return usage


def _missing_openrouter_usage_billing() -> _OpenRouterUsageBilling:
    return _OpenRouterUsageBilling(
        is_byok=None,
        is_byok_status="missing",
        cost_usd=None,
        cost_status="missing",
        upstream_inference_cost_usd=None,
        upstream_inference_cost_status="missing",
    )


def _malformed_openrouter_usage_billing() -> _OpenRouterUsageBilling:
    return _OpenRouterUsageBilling(
        is_byok=None,
        is_byok_status="malformed",
        cost_usd=None,
        cost_status="malformed",
        upstream_inference_cost_usd=None,
        upstream_inference_cost_status="malformed",
    )


def _ai_gateway_provider_returned_cost(
    *,
    response: LlmResponse,
    model: str,
) -> SettledLlmCost | None:
    raw_response = (response.metadata or {}).get("raw_response")
    if not isinstance(raw_response, Mapping):
        return None
    raw_response_mapping = cast(Mapping[str, object], raw_response)
    provider_metadata = raw_response_mapping.get("providerMetadata")
    if not isinstance(provider_metadata, Mapping):
        return None
    provider_metadata_mapping = cast(Mapping[str, object], provider_metadata)
    gateway = provider_metadata_mapping.get("gateway")
    if not isinstance(gateway, Mapping):
        return None
    gateway_mapping = cast(Mapping[str, object], gateway)
    cost_usd = _normalized_ai_gateway_cost(gateway_mapping.get("cost"))
    if cost_usd is None:
        return None
    return SettledLlmCost(
        cost_usd=cost_usd,
        provider=AI_GATEWAY_PROVIDER,
        evidence={
            "settlement_source": "provider_returned",
            "pricing_origin": "ai_gateway_provider_metadata_cost",
            "provider": AI_GATEWAY_PROVIDER,
            "model": model,
        },
    )


def _has_miner_static_pricing(*, provider: str, model: str) -> bool:
    if provider not in MINER_TOOL_LLM_PRICING:
        return False
    pricing_by_model = MINER_TOOL_LLM_PRICING[cast(MinerSelectedLlmProviderName, provider)]
    return model in pricing_by_model


def normalized_provider_cost(
    value: object,
    *,
    field_name: str,
    strict: bool = True,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        if strict:
            raise ValueError(f"{field_name} must be numeric when supplied")
        return None
    cost = float(value)
    if not math.isfinite(cost):
        if strict:
            raise ValueError(f"{field_name} must be finite when supplied")
        return None
    if cost < 0.0:
        if strict:
            raise ValueError(f"{field_name} must be non-negative when supplied")
        return None
    return cost


def _normalized_ai_gateway_cost(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        cost = float(value)
        if not math.isfinite(cost) or cost < 0.0:
            return None
        return cost
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        decimal_cost = Decimal(normalized)
    except InvalidOperation:
        return None
    if not decimal_cost.is_finite() or decimal_cost < 0:
        return None
    return float(decimal_cost)


__all__ = [
    "SettledLlmCost",
    "normalized_provider_cost",
    "settled_cost_from_metadata",
    "settled_response_cost",
    "settled_static_llm_cost",
    "with_settled_llm_cost",
]
