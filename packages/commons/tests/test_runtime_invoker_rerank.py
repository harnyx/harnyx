"""Rerank dispatch must respect credential ownership and complete-operation deadlines."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.session import ProviderCredentialSource, Session
from harnyx_commons.domain.tool_call import ToolCallOutcome, ToolResultPolicy
from harnyx_commons.errors import (
    ProviderCredentialUnavailableError,
    ToolInvocationCancelledError,
    ToolInvocationTimeoutError,
    ToolProviderError,
)
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.ports import RerankProviderPort, RerankProviderResult
from harnyx_commons.tools.provider_billing import OpenRouterAccounting
from harnyx_commons.tools.receipt_accounting import usage_from_receipts
from harnyx_commons.tools.rerank_models import RerankRequest, RerankResponse
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker
from harnyx_commons.tools.usage_tracker import UsageTracker

pytestmark = pytest.mark.anyio("asyncio")


class _Provider:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.cancelled = False
        self.requests: list[RerankRequest] = []

    async def rerank(self, request: RerankRequest) -> RerankProviderResult:
        self.requests.append(request)
        if self.block:
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
        return RerankProviderResult(
            RerankResponse.model_validate(
                {
                    "provider": "openrouter",
                    "model": request.model,
                    "results": [{"index": 0, "relevance_score": 0.9, "document": request.documents[0]}],
                }
            ),
            OpenRouterAccounting("attempted", 0.001, 0.002).to_billing({"settlement_source": "provider_returned"}),
        )

    async def aclose(self) -> None:
        return None


def _payload(**extra: object) -> dict[str, object]:
    return {
        "provider": "openrouter",
        "model": "cohere/rerank-v3.5",
        "query": "capital?",
        "documents": ["Paris"],
        **extra,
    }


@pytest.mark.parametrize("source", [ProviderCredentialSource.MINER, ProviderCredentialSource.PLATFORM])
async def test_rerank_resolves_only_the_selected_credential_source(source: ProviderCredentialSource) -> None:
    miner, platform = _Provider(), _Provider()
    runtime = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        rerank_provider_resolver=lambda *_: miner,
        platform_rerank_provider_resolver=lambda *_: platform,
    )
    context = ToolInvocationContext(
        receipt_id="rerank", session_id=uuid4(), active_attempt=1, uid=1, provider_credential_source=source
    )
    output = await runtime.invoke("rerank", args=(), kwargs=_payload(), context=context)
    assert isinstance(output, ToolInvocationOutput)
    assert output.actual_cost_usd == 0.001 and output.reference_cost_usd == 0.002
    assert len(platform.requests) == (1 if source is ProviderCredentialSource.PLATFORM else 0)
    assert len(miner.requests) == (1 if source is ProviderCredentialSource.MINER else 0)


async def test_rerank_platform_context_never_falls_back_to_miner_provider() -> None:
    provider = _Provider()
    runtime = RuntimeToolInvoker(InMemoryReceiptLog(), rerank_provider=provider)
    context = ToolInvocationContext(
        receipt_id="rerank",
        session_id=uuid4(),
        active_attempt=1,
        uid=1,
        provider_credential_source=ProviderCredentialSource.PLATFORM,
    )
    with pytest.raises(ToolProviderError):
        await runtime.invoke("rerank", args=(), kwargs=_payload(), context=context)
    assert provider.requests == []


async def test_rerank_deadline_cancels_and_awaits_provider_work() -> None:
    provider = _Provider(block=True)
    runtime = RuntimeToolInvoker(InMemoryReceiptLog(), rerank_provider=provider)
    with pytest.raises((ToolInvocationTimeoutError, ToolInvocationCancelledError)) as error:
        await runtime.invoke("rerank", args=(), kwargs=_payload(timeout=0.01))
    assert provider.cancelled
    assert error.value.billing is not None and error.value.billing.actual_cost_usd is None


def _build_executor(
    invoker: RuntimeToolInvoker, receipts: InMemoryReceiptLog, *, account_failures: bool = False,
) -> tuple[ToolExecutor, InMemorySessionRegistry, Session]:
    now = datetime.now(UTC)
    session = Session(
        session_id=uuid4(), uid=1, task_id=uuid4(), issued_at=now,
        expires_at=now + timedelta(minutes=5), budget_usd=1.0,
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    tokens = InMemoryTokenRegistry()
    tokens.register(session.session_id, "token")
    return ToolExecutor(
        session_registry=sessions, receipt_log=receipts, usage_tracker=UsageTracker(),
        tool_invoker=invoker, token_registry=tokens, clock=lambda: now,
        account_provider_failures=account_failures,
    ), sessions, session


@pytest.mark.parametrize("account_failures", [False, True])
async def test_successful_rerank_reaches_caller_and_settles_one_log_only_receipt(account_failures: bool) -> None:
    receipts = InMemoryReceiptLog()
    provider = _Provider()
    executor, sessions, session = _build_executor(
        RuntimeToolInvoker(receipts, rerank_provider=provider), receipts, account_failures=account_failures,
    )
    result = await executor.execute(ToolInvocationRequest(session.session_id, "token", "rerank", (), _payload()))
    response = RerankResponse.model_validate(result.response_payload)
    assert response.results[0].document == "Paris" and response.results[0].index == 0
    assert len(provider.requests) == 1
    recorded = receipts.for_session(session.session_id)
    assert recorded == (result.receipt,)
    assert result.receipt.outcome is ToolCallOutcome.OK
    assert result.receipt.details.result_policy is ToolResultPolicy.LOG_ONLY
    assert result.receipt.details.actual_cost_usd == 0.001
    assert result.receipt.details.reference_cost_usd == 0.002
    updated = sessions.get(session.session_id)
    assert updated is not None
    for usage in (updated.usage, usage_from_receipts(recorded)):
        assert usage.total_cost_usd == 0.002
        assert usage.actual_total_cost_usd == 0.001
        assert usage.reference_cost_by_provider == {"openrouter": 0.002}
        assert usage.actual_cost_by_provider == {"openrouter": 0.001}
        assert usage.llm_usage_totals == {}
    assert result.budget.session_used_budget_usd == 0.002
    assert result.usage is None


@pytest.mark.parametrize("account_failures", [False, True])
@pytest.mark.parametrize(
    "failure", ["validation", "missing_provider", "unsupported_provider", "credential", "resolver"],
)
async def test_rejected_rerank_keeps_zero_cost_when_receipts_are_rebuilt(failure: str, account_failures: bool) -> None:
    receipts = InMemoryReceiptLog()
    provider = _Provider()

    async def resolve(_provider: str, _context: ToolInvocationContext | None) -> RerankProviderPort:
        if failure == "credential":
            raise ProviderCredentialUnavailableError("openrouter")
        if failure == "resolver":
            raise RuntimeError("provider initialization failed")
        return provider

    executor, sessions, session = _build_executor(
        RuntimeToolInvoker(receipts, rerank_provider_resolver=resolve), receipts, account_failures=account_failures,
    )
    payload = _payload()
    if failure == "validation":
        payload["query"] = " "
    elif failure == "missing_provider":
        payload.pop("provider")
    elif failure == "unsupported_provider":
        payload["provider"] = "chutes"
    with pytest.raises(ValidationError if failure in {"validation", "missing_provider", "unsupported_provider"}
                       else ToolProviderError):
        await executor.execute(ToolInvocationRequest(session.session_id, "token", "rerank", (), payload))
    assert provider.requests == []
    recorded = receipts.for_session(session.session_id)
    assert len(recorded) == 1
    receipt = recorded[0]
    assert receipt.details.actual_cost_usd == receipt.details.reference_cost_usd == 0.0
    assert receipt.details.actual_cost_provider == (
        None if failure in {"missing_provider", "unsupported_provider"} else "openrouter"
    )
    assert not (receipt.details.extra or {}).get("reference_cost_unavailable")
    updated = sessions.get(session.session_id)
    assert updated is not None
    assert updated.usage.actual_total_cost_usd == usage_from_receipts(recorded).actual_total_cost_usd == 0.0


@pytest.mark.parametrize("known_charge", [False, True])
async def test_dispatched_rerank_failure_preserves_known_or_unknown_charge(known_charge: bool) -> None:
    class FailingProvider(_Provider):
        async def rerank(self, request: RerankRequest) -> RerankProviderResult:
            self.requests.append(request)
            raise ToolProviderError(
                "invalid provider output", provider="openrouter",
                billing=OpenRouterAccounting(
                    "attempted", 0.001 if known_charge else None, 0.002 if known_charge else None,
                ).to_billing(),
            )

    provider = FailingProvider()
    receipts = InMemoryReceiptLog()
    executor, sessions, session = _build_executor(RuntimeToolInvoker(receipts, rerank_provider=provider), receipts)
    with pytest.raises(ToolProviderError):
        await executor.execute(ToolInvocationRequest(session.session_id, "token", "rerank", (), _payload()))
    assert len(provider.requests) == 1
    recorded = receipts.for_session(session.session_id)
    assert len(recorded) == 1
    assert recorded[0].details.actual_cost_usd == (0.001 if known_charge else None)
    assert recorded[0].details.reference_cost_usd == (0.002 if known_charge else None)
    updated = sessions.get(session.session_id)
    assert updated is not None
    for usage in (updated.usage, usage_from_receipts(recorded)):
        assert usage.total_cost_usd == (0.002 if known_charge else 0.0)
        assert usage.actual_total_cost_usd == (0.001 if known_charge else None)
        assert usage.llm_usage_totals == {}
