from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import ToolCallOutcome
from harnyx_commons.errors import ProviderCredentialUnavailableError, ToolInvocationTimeoutError, ToolProviderError
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.tools.decision_models import DecisionQueryRequest, DecisionQueryResponse, DecisionUsage
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.invocation_clients import _decision_result
from harnyx_commons.tools.ports import DecisionProviderResult
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker
from harnyx_commons.tools.usage_tracker import UsageTracker

pytestmark = pytest.mark.anyio("asyncio")


def _snapshot():
    from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingSnapshot, PricingEndpoint

    return OpenRouterPricingSnapshot(
        "cloudflare/clef-flash",
        (
            PricingEndpoint.model_validate(
                {
                    "provider_name": "Cloudflare",
                    "tag": "cloudflare",
                    "model_id": "cloudflare/clef-flash-versioned",
                    "pricing": {"prompt": "0.00000009", "completion": "0", "discount": 0},
                }
            ),
        ),
        "now",
    )


class _Provider:
    def __init__(self, *, fail: bool | str | None = False) -> None:
        self.fail = fail
        self.calls = 0
        self.entered = asyncio.Event()

    async def query(self, request: DecisionQueryRequest) -> DecisionProviderResult:
        self.calls += 1
        if self.fail == "block":
            self.entered.set()
            await asyncio.Event().wait()
        if self.fail == "timeout":
            raise ToolInvocationTimeoutError("decision deadline expired")
        if self.fail == "cancel":
            raise asyncio.CancelledError()
        if self.fail is True:
            if request.provider == "reax":
                raise ToolProviderError(
                    "invalid native answer", provider="reax",
                    billing=ProviderBillingMetadata(
                        actual_cost_provider="reax", actual_cost_usd=0.02, reference_cost_usd=0.02,
                        source="response_headers", usage=DecisionUsage(input_tokens=100, output_tokens=10),
                        provider_usage={"settlement_source": "published_usd_usage_rate"},
                    ),
                )
            return _decision_result(
                request,
                {
                    "model": request.model,
                    "provider": "Cloudflare",
                    "answers": {},
                    "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.02},
                },
                snapshot=_snapshot(),
            )
        if self.fail is not False:
            raise ToolProviderError(
                "invalid native answer",
                billing=None
                if self.fail is None
                else ProviderBillingMetadata(
                    actual_cost_provider=request.provider,
                    actual_cost_usd=0.02,
                    source="response_body",
                ),
            )
        return DecisionProviderResult(
            DecisionQueryResponse.model_validate(
                {
                    "model": request.model,
                    "answers": {"b": {"type": "boolean", "probability": 0.8}},
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                }
            ),
            0.02,
            request.provider,
            {"settlement_source": "provider_returned", "inference": "attempted"},
            0.02,
        )

    async def aclose(self) -> None:
        pass


def _payload() -> dict:
    return {
        "provider": "openrouter",
        "model": "cloudflare/clef-flash",
        "state": "state",
        "questions": {"b": {"type": "boolean", "instructions": "?"}},
    }


async def test_decision_dispatch_discovery_and_unavailable_gateway() -> None:
    provider = _Provider()
    invoker = RuntimeToolInvoker(InMemoryReceiptLog(), decision_provider=provider)
    output = await invoker.invoke("decision_query", args=(), kwargs=_payload())
    assert output.actual_cost_usd == 0.02
    info = await invoker.invoke("tooling_info", args=(), kwargs={})
    assert info["allowed_decision_provider_models"]["ai_gateway"] == []
    assert info["allowed_decision_provider_models"]["jev"] == ["jev-1.13.0"]
    assert info["pricing"]["decision_query"]["provider_models"]["jev"]["jev-1.13.0"]["input_per_million"] == 0.042
    with pytest.raises(ValidationError):
        await invoker.invoke("decision_query", args=(), kwargs={**_payload(), "provider": "ai_gateway"})
    assert provider.calls == 1


async def test_reax_discovery_distinguishes_key_tier_and_budget_estimate() -> None:
    invoker = RuntimeToolInvoker(InMemoryReceiptLog(), decision_provider=_Provider())
    info = await invoker.invoke("tooling_info", args=(), kwargs={})
    rates = info["pricing"]["decision_query"]
    assert rates["routing"]["reax"] == "api_key_default"
    assert rates["provider_rate_usage"]["reax"] == "conservative_budget_estimate"
    cards = rates["provider_tier_models"]["reax"]
    assert cards["global"]["s1-fast-2"]["input_per_million"] == 0.019
    assert cards["eu"]["s1-fast-2"]["input_per_million"] == 0.021
    assert rates["provider_models"]["reax"] == cards["eu"]


async def test_reax_unsettled_result_retains_reference_and_original_receipt() -> None:
    """The invoker's final settlement gate must not erase adapter billing evidence."""
    from dataclasses import replace

    class UnsettledProvider(_Provider):
        async def query(self, request):
            result = await super().query(request)
            return replace(result, actual_cost_usd=None, actual_cost_evidence={"currency": "EUR"})

    invoker = RuntimeToolInvoker(InMemoryReceiptLog(), decision_provider=UnsettledProvider())
    with pytest.raises(ToolProviderError, match="missing settled cost") as caught:
        await invoker.invoke(
            "decision_query", args=(), kwargs={**_payload(), "provider": "reax", "model": "s1-pro-3"},
        )
    assert caught.value.billing.actual_cost_usd is None
    assert caught.value.billing.reference_cost_usd == 0.02
    assert caught.value.billing.provider_usage == {"currency": "EUR"}


@pytest.mark.parametrize("provider_name", ["openrouter", "reax"])
@pytest.mark.parametrize("fail", [False, True, None, "timeout", "cancel"])
async def test_decision_success_and_billed_failure_charge_without_chat_tokens(
    fail: bool | str | None, provider_name: str,
) -> None:
    now = datetime.now(UTC)
    session = Session(
        session_id=uuid4(), uid=1, task_id=uuid4(), issued_at=now, expires_at=now + timedelta(minutes=5), budget_usd=1.0
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    tokens = InMemoryTokenRegistry()
    tokens.register(session.session_id, "token")
    receipts = InMemoryReceiptLog()
    provider = _Provider(fail=fail)
    invoker = RuntimeToolInvoker(receipts, decision_provider_resolver=lambda _p, _c: provider)
    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=receipts,
        usage_tracker=UsageTracker(),
        tool_invoker=invoker,
        token_registry=tokens,
        clock=lambda: now,
    )
    payload = _payload()
    if provider_name == "reax":
        payload.update(provider="reax", model="s1-pro-3")
    request = ToolInvocationRequest(session.session_id, "token", "decision_query", (), payload)
    if fail is not False:
        with pytest.raises(
            asyncio.CancelledError
            if fail == "cancel"
            else ToolInvocationTimeoutError
            if fail == "timeout"
            else ToolProviderError
        ):
            await executor.execute(request)
    else:
        await executor.execute(request)
    updated = sessions.get(session.session_id)
    assert updated is not None
    assert updated.usage.total_cost_usd == (0 if fail in (None, "timeout", "cancel") else 0.02)
    assert updated.usage.actual_total_cost_usd == (None if fail in (None, "timeout", "cancel") else 0.02)
    assert updated.usage.llm_usage_totals == {}
    calls = receipts.for_session(session.session_id)
    assert len(calls) == 1
    if fail is True:
        assert calls[0].details.response_payload == {"usage": {"input_tokens": 100, "output_tokens": 10}}
    assert calls[0].details.actual_cost_usd == (None if fail in (None, "timeout", "cancel") else 0.02)


@pytest.mark.parametrize(
    "failure",
    [
        "validation",
        "missing_provider",
        "unsupported_provider",
        "gateway",
        "credential",
        "resolver",
        "timeout",
        "cancel",
        "provider_timeout",
        "provider_cancel",
    ],
)
async def test_failed_decision_cost_distinguishes_provider_dispatch(failure: str) -> None:
    """Rejected calls preserve costs and provider attribution; dispatched failures can make costs unknown."""
    now = datetime.now(UTC)
    session = Session(
        session_id=uuid4(), uid=1, task_id=uuid4(), issued_at=now, expires_at=now + timedelta(minutes=5), budget_usd=1.0
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    tokens = InMemoryTokenRegistry()
    tokens.register(session.session_id, "token")
    receipts = InMemoryReceiptLog()
    provider = _Provider()
    resolving = asyncio.Event()
    reject = False

    async def resolve(_provider, _context):
        if not reject or failure in {"provider_timeout", "provider_cancel"}:
            return provider
        if failure == "credential":
            raise ProviderCredentialUnavailableError("openrouter")
        if failure == "resolver":
            raise RuntimeError("provider initialization failed")
        if failure == "timeout":
            raise ToolInvocationTimeoutError("credential lookup timed out")
        resolving.set()
        await asyncio.Event().wait()

    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=receipts,
        usage_tracker=UsageTracker(),
        tool_invoker=RuntimeToolInvoker(receipts, decision_provider_resolver=resolve),
        token_registry=tokens,
        clock=lambda: now,
    )
    request = ToolInvocationRequest(session.session_id, "token", "decision_query", (), _payload())
    await executor.execute(request)
    reject = True
    dispatched = failure in {"provider_timeout", "provider_cancel"}
    if dispatched:
        provider.fail = "block"
    if failure == "provider_timeout":
        request = ToolInvocationRequest(
            session.session_id, "token", "decision_query", (), {**_payload(), "timeout": 0.01}
        )
    validation_failure = failure in {"validation", "missing_provider", "unsupported_provider", "gateway"}
    if validation_failure:
        payload = _payload()
        if failure == "missing_provider":
            del payload["provider"]
        elif failure == "unsupported_provider":
            payload["provider"] = "unsupported"
        elif failure == "gateway":
            payload["provider"] = "ai_gateway"
        else:
            payload["questions"] = {}
        request = ToolInvocationRequest(session.session_id, "token", "decision_query", (), payload)
    if failure in {"cancel", "provider_cancel"}:
        task = asyncio.create_task(executor.execute(request))
        await asyncio.wait_for((provider.entered if dispatched else resolving).wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(
            ValidationError
            if validation_failure
            else ToolInvocationTimeoutError
            if failure in {"timeout", "provider_timeout"}
            else ToolProviderError
        ):
            await executor.execute(request)
    assert provider.calls == (2 if dispatched else 1)
    updated = sessions.get(session.session_id)
    assert updated is not None
    assert updated.usage.total_cost_usd == 0.02
    assert updated.usage.actual_total_cost_usd == (None if dispatched else 0.02)
    expected_costs = {"openrouter": 0.02}
    if failure == "gateway":
        expected_costs["ai_gateway"] = 0.0
    assert updated.usage.cost_by_provider == expected_costs
    assert updated.usage.reference_cost_by_provider == expected_costs
    assert updated.usage.actual_cost_by_provider == expected_costs
    calls = receipts.for_session(session.session_id)
    assert len(calls) == 2
    failed_call = next(call for call in calls if call.outcome is not ToolCallOutcome.OK)
    assert failed_call.details.actual_cost_usd == (None if dispatched else 0.0)
    assert failed_call.details.actual_cost_provider == (
        None
        if failure in {"missing_provider", "unsupported_provider"}
        else "ai_gateway"
        if failure == "gateway"
        else "openrouter"
    )
    assert bool(failed_call.details.extra.get("reference_cost_unavailable")) is dispatched
