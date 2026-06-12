"""Actual-cost pricing for Chutes miner-selected LLM calls."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from harnyx_commons.json_types import JsonObject
from harnyx_commons.llm.pricing import ModelPricing
from harnyx_commons.llm.schema import LlmUsage

_DEFAULT_TTL_SECONDS = 3600.0

CHUTES_STATIC_PRICING: Mapping[str, ModelPricing] = {
    "deepseek-ai/DeepSeek-V3.2-TEE": ModelPricing(0.28, 0.42, 0.0),
    "zai-org/GLM-5-TEE": ModelPricing(0.95, 2.55, 0.0),
    "Qwen/Qwen3.6-27B-TEE": ModelPricing(0.30, 2.00, 0.0),
    "google/gemma-4-31B-turbo-TEE": ModelPricing(0.15, 0.42, 0.0),
}

CHUTES_ACTUAL_COST_FALLBACK_PRICING = CHUTES_STATIC_PRICING


@dataclass(frozen=True, slots=True)
class ChutesActualCost:
    cost_usd: float
    provider: Literal["chutes"]
    evidence: JsonObject


class ChutesModelPricingCache:
    """Caches Chutes model pricing and provides repo-owned static rates."""

    def __init__(
        self,
        *,
        cached_pricing: Mapping[str, ModelPricing] | None = None,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        static_pricing: Mapping[str, ModelPricing] = CHUTES_STATIC_PRICING,
        fallback_pricing: Mapping[str, ModelPricing] | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._static_pricing = dict(static_pricing if fallback_pricing is None else fallback_pricing)
        self._snapshot: dict[str, ModelPricing] = dict(cached_pricing or {})
        self._snapshot_loaded_at: float | None = time.monotonic() if cached_pricing is not None else None

    def update_snapshot(self, pricing: Mapping[str, ModelPricing]) -> None:
        self._snapshot = dict(pricing)
        self._snapshot_loaded_at = time.monotonic()

    async def price(self, *, model: str, usage: LlmUsage) -> ChutesActualCost:
        cached = self._cached_pricing(model)
        if cached is not None:
            return _token_price(
                model=model,
                usage=usage,
                pricing=cached,
                settlement_source="cached_provider_pricing",
                pricing_origin="chutes_live_snapshot",
            )

        static_pricing = self._static_pricing[model]
        return _token_price(
            model=model,
            usage=usage,
            pricing=static_pricing,
            settlement_source="static_pricing",
            pricing_origin="chutes_repo_rates",
        )

    def _cached_pricing(self, model: str) -> ModelPricing | None:
        if self._snapshot_loaded_at is None:
            return None
        if time.monotonic() - self._snapshot_loaded_at > self._ttl_seconds:
            return None
        return self._snapshot.get(model)


def _parse_models_payload(payload: object) -> dict[str, ModelPricing]:
    entries = _model_entries(payload)
    pricing_by_model: dict[str, ModelPricing] = {}
    for entry in entries:
        model_id = _optional_text(entry.get("id") or entry.get("name"))
        if model_id is None:
            continue
        pricing = _pricing_from_entry(entry)
        if pricing is None:
            continue
        pricing_by_model[model_id] = pricing
    return pricing_by_model


def _model_entries(payload: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(payload, list):
        raw_entries = payload
    elif isinstance(payload, Mapping):
        payload_mapping = cast(Mapping[str, object], payload)
        data = payload_mapping.get("data")
        if not isinstance(data, list):
            return ()
        raw_entries = data
    else:
        return ()
    entries: list[Mapping[str, object]] = []
    for entry in raw_entries:
        if isinstance(entry, Mapping):
            entries.append(cast(Mapping[str, object], entry))
    return tuple(entries)


def _pricing_from_entry(entry: Mapping[str, object]) -> ModelPricing | None:
    pricing_source = entry.get("pricing")
    if not isinstance(pricing_source, Mapping):
        pricing_source = entry
    pricing_mapping = cast(Mapping[str, object], pricing_source)
    input_rate = _rate_per_million(
        pricing_mapping,
        ("input_per_million", "prompt_per_million", "input", "prompt", "prompt_tokens"),
    )
    output_rate = _rate_per_million(
        pricing_mapping,
        ("output_per_million", "completion_per_million", "output", "completion", "completion_tokens"),
    )
    if input_rate is None or output_rate is None:
        return None
    reasoning_rate = _rate_per_million(pricing_mapping, ("reasoning_per_million", "reasoning"))
    return ModelPricing(input_rate, output_rate, reasoning_rate or 0.0)


def _rate_per_million(source: Mapping[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _optional_float(source.get(key))
        if value is None:
            continue
        if value <= 0.0001:
            return value * 1_000_000
        return value
    return None


def _token_price(
    *,
    model: str,
    usage: LlmUsage,
    pricing: ModelPricing,
    settlement_source: Literal["cached_provider_pricing", "static_pricing"],
    pricing_origin: Literal["chutes_live_snapshot", "chutes_repo_rates"],
) -> ChutesActualCost:
    prompt_tokens = float(usage.prompt_tokens or 0)
    completion_tokens = float(usage.completion_tokens or 0)
    reasoning_tokens = float(usage.reasoning_tokens or 0)
    cost_usd = (
        (prompt_tokens / 1_000_000) * pricing.input_per_million
        + (completion_tokens / 1_000_000) * pricing.output_per_million
        + (reasoning_tokens / 1_000_000) * pricing.billable_reasoning_per_million
    )
    evidence: JsonObject = {
        "settlement_source": settlement_source,
        "pricing_origin": pricing_origin,
        "model": model,
        "input_per_million": pricing.input_per_million,
        "output_per_million": pricing.output_per_million,
        "reasoning_per_million": pricing.billable_reasoning_per_million,
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "reasoning_tokens": usage.reasoning_tokens or 0,
    }
    return ChutesActualCost(cost_usd=cost_usd, provider="chutes", evidence=evidence)


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "CHUTES_ACTUAL_COST_FALLBACK_PRICING",
    "CHUTES_STATIC_PRICING",
    "ChutesActualCost",
    "ChutesModelPricingCache",
    "_parse_models_payload",
]
