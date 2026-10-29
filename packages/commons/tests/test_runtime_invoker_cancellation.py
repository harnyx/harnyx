"""Cancellation must retain billing from every completed recoverable provider result."""

from __future__ import annotations

import asyncio

import pytest

from harnyx_commons.errors import ToolInvocationCancelledError
from harnyx_commons.tools.decision_models import DecisionQueryResponse
from harnyx_commons.tools.ports import DecisionProviderResult, RerankProviderResult
from harnyx_commons.tools.rerank_models import RerankResponse
from harnyx_commons.tools.runtime_invoker import _invoke_with_optional_timeout

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture(params=["decision_query", "rerank"])
def billed_result(request: pytest.FixtureRequest) -> DecisionProviderResult | RerankProviderResult:
    decision = DecisionProviderResult(
        DecisionQueryResponse.model_validate({
            "model": "cloudflare/clef-flash", "answers": {"b": {"type": "boolean", "probability": 0.8}},
        }),
        0.001, "openrouter", {"inference": "attempted", "provider_request_id": "paid-result"}, 0.002,
    )
    if request.param == "decision_query":
        return decision
    return RerankProviderResult(
        RerankResponse.model_validate({
            "provider": "openrouter", "model": "cohere/rerank-v3.5",
            "results": [{"index": 0, "relevance_score": 0.9, "document": "Paris"}],
        }), decision.billing,
    )


@pytest.mark.parametrize("completion", ["already_completed", "finishes_on_cancellation"])
async def test_cancelled_wrapper_retains_completed_provider_billing(
    monkeypatch: pytest.MonkeyPatch, billed_result: DecisionProviderResult | RerankProviderResult, completion: str,
) -> None:
    entered = asyncio.Event()

    async def operation() -> DecisionProviderResult | RerankProviderResult:
        if completion == "already_completed":
            return billed_result
        try:
            entered.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return billed_result
        raise AssertionError("provider should finish during cancellation")

    async def cancel_caller(
        tasks: set[asyncio.Task[DecisionProviderResult | RerankProviderResult]], *, timeout: float,
    ) -> None:
        if completion == "already_completed":
            await asyncio.gather(*tasks)
        else:
            await entered.wait()
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "wait", cancel_caller)
    with pytest.raises(ToolInvocationCancelledError) as caught:
        await _invoke_with_optional_timeout("inference", 1.0, operation)
    assert caught.value.billing == billed_result.billing
    assert caught.value.billing.actual_cost_usd == 0.001
    assert caught.value.billing.reference_cost_usd == 0.002
