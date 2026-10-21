from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest

from harnyx_commons.llm.providers.openrouter_pricing import (
    OpenRouterPricingCatalog,
    OpenRouterPricingSnapshot,
    PricingEndpoint,
)
from harnyx_commons.llm.tool_models import validate_paid_openrouter_model

pytestmark = pytest.mark.anyio("asyncio")


def endpoint(provider: str = "Inception", discount: float | None = 0.5, **pricing: object) -> PricingEndpoint:
    return PricingEndpoint.model_validate(
        {
            "provider_name": provider,
            "tag": provider.lower(),
            "model_id": "inception/mercury-3",
            "pricing": {"prompt": "0.0000001", "completion": "0.00000035", "discount": discount, **pricing},
        }
    )


def test_maximum_discount_is_scoped_to_provider_and_keeps_separate_charges() -> None:
    snapshot = OpenRouterPricingSnapshot(
        "inception/mercury-3", (endpoint(discount=0.1), endpoint(), endpoint("Other", 0.9)), "now"
    )
    reference, discount = snapshot.reference_cost(
        provider="Inception", inference=Decimal("0.004"), separate_charge=Decimal("0.001")
    )
    assert reference == pytest.approx(0.009)
    assert discount == Decimal("0.5")
    with pytest.raises(ValueError, match="absent"):
        snapshot.reference_cost(provider="Unknown", inference=Decimal(1), separate_charge=Decimal(0))


@pytest.mark.parametrize("selection", ["Cerebras", "cerebras", "CEREBRAS/FAST"])
def test_provider_selection_casing_cannot_reject_an_available_paid_route(selection: str) -> None:
    snapshot = OpenRouterPricingSnapshot(
        "inception/mercury-3",
        (
            endpoint("Cerebras", 0.1).model_copy(update={"tag": "cerebras/fast"}),
            endpoint("Cerebras", 0.5).model_copy(update={"tag": "cerebras/other"}),
        ),
        "now",
        provider_only=(selection,),
    )
    snapshot.preflight()
    assert snapshot.reference_cost(provider="Cerebras", inference=Decimal("0.004"), separate_charge=Decimal(0)) == (
        0.008,
        Decimal("0.5"),
    )


def test_pinned_route_excludes_other_prices_but_keeps_provider_maximum_discount() -> None:
    snapshot = OpenRouterPricingSnapshot(
        "inception/mercury-3",
        (
            endpoint(discount=0.1),
            endpoint(discount=0.5, overrides=[{"prompt": "0"}]).model_copy(update={"tag": "inception/turbo"}),
            endpoint("Other", None),
        ),
        "now",
        provider_only=("inception/turbo",),
    )
    # The pinned endpoint itself is unsupported.
    with pytest.raises(ValueError, match="conditional"):
        snapshot.preflight()
    snapshot = replace(snapshot, provider_only=("inception",))
    with pytest.raises(ValueError, match="conditional"):
        snapshot.preflight()
    # Pin a supported sibling without narrowing the provider-wide discount.
    snapshot = replace(
        snapshot,
        endpoints=(snapshot.endpoints[0].model_copy(update={"tag": "inception/standard"}), *snapshot.endpoints[1:]),
        provider_only=("inception/standard",),
    )
    reference, discount = snapshot.reference_cost(
        provider="Inception", inference=Decimal("0.004"), separate_charge=Decimal(0)
    )
    assert reference == pytest.approx(0.008)
    assert discount == Decimal("0.5")
    with pytest.raises(ValueError, match="absent"):
        snapshot.reference_cost(provider="Other", inference=Decimal(1), separate_charge=Decimal(0))
    with pytest.raises(ValueError, match="no selectable"):
        replace(snapshot, provider_only=("inception/stand",)).preflight()
    with pytest.raises(ValueError, match="conditional"):
        replace(snapshot, provider_only=None).preflight()


@pytest.mark.parametrize(
    "pricing",
    [
        {"discount": 1},
        {"discount": None},
        {"discount": 0, "overrides": [{"utc_start": "00:00", "prompt": "0"}]},
        {"request": "0.1"},
        {"web_search": "0.1"},
    ],
)
def test_unsupported_regular_basis_is_rejected_even_with_zero_discount(pricing: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="regular price is unavailable"):
        OpenRouterPricingSnapshot("inception/mercury-3", (endpoint(**pricing),), "now").preflight()


@pytest.mark.parametrize("model", ["openrouter/free", "x/model:free", "x/model:free:nitro", "x/model:nitro:free"])
def test_explicit_free_modifiers_cannot_hide_in_routes(model: str) -> None:
    with pytest.raises(ValueError, match="free"):
        validate_paid_openrouter_model(model)


async def test_cache_coalesces_misses_and_binds_immutable_snapshot() -> None:
    calls: list[str] = []

    async def serve(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        await asyncio.sleep(0)
        payload = (
            {"data": {"id": "inception/mercury-3", "endpoints": [endpoint().model_dump(mode="json")]}}
            if request.url.path.endswith("endpoints")
            else {"data": [{"id": "inception/mercury-3"}, {"id": "x/model:free"}, {"id": "openrouter/free"}]}
        )
        return httpx.Response(200, json=payload)

    catalog = OpenRouterPricingCatalog(transport=httpx.MockTransport(serve))
    snapshots = await asyncio.gather(*(catalog.snapshot("inception/mercury-3") for _ in range(8)))
    assert len(calls) == 2
    assert all(s is snapshots[0] for s in snapshots)
    assert await catalog.models() == ("inception/mercury-3",)


async def test_expired_catalog_never_silently_reuses_stale_prices() -> None:
    available = True

    def serve(request: httpx.Request) -> httpx.Response:
        if not available:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"id": "inception/mercury-3"}]})

    catalog = OpenRouterPricingCatalog(ttl_seconds=0, transport=httpx.MockTransport(serve))
    assert await catalog.models()
    available = False
    with pytest.raises(httpx.HTTPStatusError):
        await catalog.snapshot("inception/mercury-3")


@pytest.mark.parametrize("reverse_catalog", [False, True], ids=["regular-first", "batch-first"])
@pytest.mark.parametrize(
    "requested_model,expected_model,serving_provider,expected_reference,expected_discount",
    [
        ("openai/gpt-oss-20b", "openai/gpt-oss-20b", "Darkbloom", 0.008, Decimal("0.5")),
        ("openai/gpt-oss-20b:batch", "openai/gpt-oss-20b:batch", "DeepInfra", 0.005, Decimal("0.2")),
        ("openai/gpt-oss-20b:nitro", "openai/gpt-oss-20b", "Darkbloom", 0.008, Decimal("0.5")),
        ("openai/gpt-oss-20b:floor", "openai/gpt-oss-20b", "Darkbloom", 0.008, Decimal("0.5")),
    ],
)
async def test_exact_catalog_id_wins_over_shared_alias(
    reverse_catalog: bool,
    requested_model: str,
    expected_model: str,
    serving_provider: str,
    expected_reference: float,
    expected_discount: Decimal,
) -> None:
    regular = "openai/gpt-oss-20b"
    batch = regular + ":batch"
    entries = [{"id": regular, "canonical_slug": regular}, {"id": batch, "canonical_slug": regular}]
    if reverse_catalog:
        entries.reverse()
    endpoints = {
        regular: endpoint("Darkbloom", 0.5).model_copy(update={"model_id": regular}),
        batch: endpoint("DeepInfra", 0.2).model_copy(update={"model_id": batch}),
    }
    responses = {
        f"/api/v1/models/{model}/endpoints": {
            "data": {"id": model, "endpoints": [pricing.model_dump(mode="json")]}
        }
        for model, pricing in endpoints.items()
    }
    calls: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        payload = {"data": entries} if request.url.path == "/api/v1/models" else responses[request.url.path]
        return httpx.Response(200, json=payload)

    catalog = OpenRouterPricingCatalog(transport=httpx.MockTransport(serve))
    snapshot = await catalog.snapshot(requested_model)

    assert calls == ["/api/v1/models", f"/api/v1/models/{expected_model}/endpoints"]
    assert snapshot.model == expected_model
    reference, discount = snapshot.reference_cost(
        provider=serving_provider, inference=Decimal("0.004"), separate_charge=Decimal(0)
    )
    assert reference == pytest.approx(expected_reference)
    assert discount == expected_discount
