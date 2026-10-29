"""Public catalog contract checks; no credentials or paid generation requests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingCatalog

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


async def test_public_chat_catalog_preserves_exact_model_id() -> None:
    snapshot = await OpenRouterPricingCatalog().snapshot("openai/gpt-oss-20b")
    assert snapshot.model == "openai/gpt-oss-20b"
    snapshot.preflight()


async def test_public_endpoint_tags_support_pinned_provider_pricing() -> None:
    catalog = OpenRouterPricingCatalog()
    snapshot = await catalog.snapshot("inception/mercury-3", provider_only=("inception",))
    assert snapshot.model == "inception/mercury-3"
    assert any(endpoint.tag == "inception" and endpoint.provider_name == "Inception" for endpoint in snapshot.endpoints)
    # Check the returned tag/discount contract without asserting a transient promotion rate.
    reference, discount = snapshot.reference_cost(
        provider="Inception", inference=Decimal(1), separate_charge=Decimal(0)
    )
    assert 0 <= discount < 1
    assert reference >= 1


async def test_public_zero_discount_conditional_route_has_regular_price_basis() -> None:
    """The shared catalog must admit a real conditional route without an experiment adapter."""
    snapshot = await OpenRouterPricingCatalog().snapshot("openai/gpt-6-luna")
    assert snapshot.endpoints
    assert any(endpoint.pricing.overrides for endpoint in snapshot.endpoints)
    for endpoint in snapshot.endpoints:
        assert endpoint.pricing.discount == 0
        reference, discount = snapshot.reference_cost(
            provider=endpoint.provider_name, inference=Decimal("0.0123"), separate_charge=Decimal("0.001")
        )
        assert discount == 0
        assert reference == 0.0133


async def test_public_decision_catalog_binds_native_and_canonical_ids_without_paid_inference() -> None:
    catalog = OpenRouterPricingCatalog()
    models = await catalog.decision_models()
    assert models and "cloudflare/clef-flash" in models
    snapshot = await catalog.snapshot("cloudflare/clef-flash", output_modality="decisions")
    assert snapshot.model in models
    if snapshot.canonical_model is not None:
        assert snapshot.canonical_model in models
        pinned = await catalog.snapshot(snapshot.canonical_model, output_modality="decisions")
        assert pinned.model == snapshot.model
    snapshot.preflight()


async def test_public_rerank_catalog_binds_paid_text_models_without_inference() -> None:
    catalog = OpenRouterPricingCatalog()
    models = await catalog.rerank_models()
    assert models and "cohere/rerank-v3.5" in models
    assert all(":free" not in model and model != "openrouter/free" for model in models)
    snapshot = await catalog.snapshot("cohere/rerank-v3.5", output_modality="rerank")
    assert snapshot.model in models
    if snapshot.canonical_model is not None:
        assert snapshot.canonical_model in models
        pinned = await catalog.snapshot(snapshot.canonical_model, output_modality="rerank")
        assert pinned.model == snapshot.model
    snapshot.preflight()
