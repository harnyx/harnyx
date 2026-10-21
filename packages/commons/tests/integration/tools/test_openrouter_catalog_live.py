"""Public catalog contract checks; no credentials or paid generation requests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingCatalog

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


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
