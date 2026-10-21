"""Public OpenRouter price snapshots and miner reference-cost normalization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from time import monotonic
from typing import Literal, cast
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

from harnyx_commons.json_types import JsonObject
from harnyx_commons.llm.tool_models import is_free_openrouter_model, validate_paid_openrouter_model

CATALOG_URL = "https://openrouter.ai/api/v1/models"


class EndpointPricing(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)

    prompt: Decimal = Field(ge=0)
    completion: Decimal = Field(ge=0)
    discount: Decimal | None = Field(default=None, ge=0, le=1)
    request: Decimal = Field(default=Decimal(0), ge=0)
    web_search: Decimal = Field(default=Decimal(0), ge=0)
    overrides: tuple[JsonObject, ...] = ()


class PricingEndpoint(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    provider_name: str = Field(min_length=1)
    tag: str = Field(min_length=1)
    model_id: str
    pricing: EndpointPricing


class _EndpointModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: str
    endpoints: tuple[PricingEndpoint, ...] = Field(min_length=1)


class _EndpointsResponse(BaseModel):
    data: _EndpointModel


class _Model(BaseModel):
    id: str
    canonical_slug: str | None = None


class _ModelsResponse(BaseModel):
    data: tuple[_Model, ...]


@dataclass(frozen=True, slots=True)
class OpenRouterPricingSnapshot:
    model: str
    endpoints: tuple[PricingEndpoint, ...]
    fetched_at: str
    provider_only: tuple[str, ...] | None = None
    canonical_model: str | None = None

    def _selectable_endpoints(self) -> tuple[PricingEndpoint, ...]:
        if self.provider_only is None:
            return self.endpoints
        # OpenRouter routing slugs are endpoint tags; a prefix includes its variants/regions.
        return tuple(
            endpoint
            for endpoint in self.endpoints
            if any(
                endpoint.tag.casefold() == slug.casefold() or endpoint.tag.casefold().startswith(slug.casefold() + "/")
                for slug in self.provider_only
            )
        )

    def _provider_discount(self, provider: str) -> Decimal:
        discounts = [e.pricing.discount for e in self.endpoints if e.provider_name.casefold() == provider.casefold()]
        if not discounts or any(discount is None or discount == 1 for discount in discounts):
            raise ValueError("OpenRouter regular price is unavailable: missing or complete discount")
        return max(discount for discount in discounts if discount is not None)

    def preflight(self) -> None:
        endpoints = self._selectable_endpoints()
        if not endpoints:
            raise ValueError("OpenRouter request has no selectable public pricing endpoint")
        for endpoint in endpoints:
            pricing = endpoint.pricing
            self._provider_discount(endpoint.provider_name)
            if pricing.overrides:
                raise ValueError("OpenRouter regular price is unavailable: conditional pricing overrides")
            if pricing.request or pricing.web_search:
                raise ValueError("OpenRouter regular price is unavailable: unseparated non-inference charges")

    def reference_cost(self, *, provider: str, inference: Decimal, separate_charge: Decimal) -> tuple[float, Decimal]:
        matches = [e for e in self._selectable_endpoints() if e.provider_name.casefold() == provider.casefold()]
        if not matches:
            raise ValueError("OpenRouter serving provider is absent from the bound pricing snapshot")
        self.preflight()
        discount = self._provider_discount(provider)
        return float(inference / (1 - discount) + separate_charge), discount


class OpenRouterPricingCatalog:
    """Process-owned cache; HTTP resources exist only for bounded refreshes."""

    def __init__(self, *, ttl_seconds: float = 300, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._ttl = ttl_seconds
        self._transport = transport
        self._cache: dict[str, tuple[float, object]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _get(self, key: str, *, timeout: float) -> object:
        async with self._locks.setdefault(key, asyncio.Lock()):
            cached = self._cache.get(key)
            if cached is not None and monotonic() - cached[0] < self._ttl:
                return cached[1]
            async with httpx.AsyncClient(transport=self._transport, timeout=timeout) as client:
                response = await client.get(CATALOG_URL + key)
                response.raise_for_status()
                value: object = response.json()
            # Parse before publishing, so malformed refreshes cannot replace a valid snapshot.
            if key and not key.startswith("?"):
                parsed = _EndpointsResponse.model_validate(value).data
                result: object = OpenRouterPricingSnapshot(
                    model=parsed.id, endpoints=parsed.endpoints, fetched_at=datetime.now(UTC).isoformat()
                )
            else:
                result = tuple(
                    m for m in _ModelsResponse.model_validate(value).data if not is_free_openrouter_model(m.id)
                )
            self._cache[key] = (monotonic(), result)
            return result

    async def models(self, *, timeout: float = 30) -> tuple[str, ...]:
        return tuple(model.id for model in cast(tuple[_Model, ...], await self._get("", timeout=timeout)))

    async def decision_models(self, *, timeout: float = 30) -> tuple[str, ...]:
        models = cast(tuple[_Model, ...], await self._get("?output_modalities=decisions", timeout=timeout))
        return tuple(
            dict.fromkeys(route for model in models for route in (model.id, model.canonical_slug) if route is not None)
        )

    async def snapshot(
        self,
        model: str,
        *,
        timeout: float = 30,
        provider_only: tuple[str, ...] | None = None,
        output_modality: Literal["decisions"] | None = None,
    ) -> OpenRouterPricingSnapshot:
        validate_paid_openrouter_model(model)
        # Catalog ids are authoritative. Documented routing modifiers retain the same model price basis.
        key = "" if output_modality is None else "?output_modalities=" + output_modality
        entries = cast(tuple[_Model, ...], await self._get(key, timeout=timeout))
        routes = {route: entry for entry in entries for route in (entry.id, entry.canonical_slug) if route is not None}
        catalog_model = model
        if model not in routes:
            base_model, *modifiers = model.split(":")
            if not modifiers or any(modifier not in {"nitro", "floor"} for modifier in modifiers):
                raise ValueError("OpenRouter route has no public regular-price basis")
            catalog_model = base_model
        if catalog_model not in routes:
            raise ValueError("OpenRouter model has no public regular-price basis")
        entry = routes[catalog_model]
        catalog_model = entry.id
        result = cast(
            OpenRouterPricingSnapshot,
            await self._get("/" + quote(catalog_model, safe="/") + "/endpoints", timeout=timeout),
        )
        if entry.canonical_slug is not None and result.canonical_model != entry.canonical_slug:
            result = replace(result, canonical_model=entry.canonical_slug)
        validate_paid_openrouter_model(result.model)
        if provider_only is not None:
            result = replace(result, provider_only=provider_only)
        result.preflight()
        return result


@lru_cache(maxsize=1)
def shared_openrouter_pricing_catalog() -> OpenRouterPricingCatalog:
    return OpenRouterPricingCatalog()
