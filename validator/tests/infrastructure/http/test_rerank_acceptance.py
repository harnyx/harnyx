"""A native rerank result must survive the miner SDK, host, executor and accounting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import ToolCallOutcome, ToolResultPolicy
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.providers.openrouter import OpenRouterRerankClient
from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingCatalog
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.invocation_clients import OpenRouterRerankProvider
from harnyx_commons.tools.receipt_accounting import usage_from_receipts
from harnyx_commons.tools.runtime_invoker import build_miner_sandbox_tool_invoker
from harnyx_commons.tools.token_semaphore import DEFAULT_TOOL_CONCURRENCY_LIMITS, ToolConcurrencyLimiter
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_miner_sdk._internal.tool_invoker import bind_tool_invoker
from harnyx_miner_sdk.api import rerank
from harnyx_miner_sdk.tools.proxy import ToolProxy
from harnyx_validator.infrastructure.http.routes import ToolRouteDeps, add_tool_routes

pytestmark = pytest.mark.anyio("asyncio")
MODEL = "cohere/rerank-v3.5"


async def test_sdk_rerank_returns_ranked_documents_with_one_successful_billed_receipt() -> None:
    paid_requests: list[httpx.Request] = []

    def openrouter(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{
                "id": MODEL, "architecture": {"input_modalities": ["text"], "output_modalities": ["rerank"]},
            }]})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json={"data": {"id": MODEL, "endpoints": [{
                "provider_name": "Cohere", "tag": "cohere", "model_id": MODEL,
                "pricing": {"prompt": "0", "completion": "0", "discount": 0.5},
            }]}})
        assert request.method == "POST" and request.url.path == "/api/v1/rerank"
        paid_requests.append(request)
        return httpx.Response(200, json={
            "model": "rerank-v3.5", "provider": "Cohere", "id": "rerank-contract",
            "results": [{"index": 1, "relevance_score": 0.9, "document": {"text": "Paris"}}],
            "usage": {"cost": 0.001, "search_units": 1},
        })

    now = datetime.now(UTC)
    session = Session(
        session_id=uuid4(), uid=7, task_id=uuid4(), issued_at=now,
        expires_at=now + timedelta(minutes=5), budget_usd=1.0,
    )
    sessions = InMemorySessionRegistry()
    sessions.create(session)
    tokens = InMemoryTokenRegistry()
    token = uuid4().hex
    tokens.register(session.session_id, token)
    receipts = InMemoryReceiptLog()
    transport = httpx.MockTransport(openrouter)
    provider = OpenRouterRerankProvider(
        api_key=SecretStr("test-key"), timeout_seconds=5,
        pricing_catalog=OpenRouterPricingCatalog(transport=transport),
    )
    async with httpx.AsyncClient(base_url="https://openrouter.ai/api/v1/", transport=transport) as native_client:
        provider._clients[MODEL] = OpenRouterRerankClient(
            model=MODEL, api_key=SecretStr("test-key"), client=native_client,
        )
        invoker = build_miner_sandbox_tool_invoker(receipts, rerank_provider=provider)
        executor = ToolExecutor(
            session_registry=sessions, receipt_log=receipts, token_registry=tokens,
            usage_tracker=UsageTracker(), tool_invoker=invoker, clock=lambda: now,
        )
        deps = ToolRouteDeps(
            tool_executor=executor, tool_concurrency_limiter=ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS),
        )
        app = FastAPI()
        add_tool_routes(app, lambda: deps)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tool-host") as client:
            proxy = ToolProxy("http://tool-host", token, session_id=str(session.session_id), client=client)
            with bind_tool_invoker(proxy):
                result = await rerank("capital?", ["Berlin", "Paris"], provider="openrouter", model=MODEL, top_n=1)
        await provider.aclose()

    assert len(paid_requests) == 1
    assert result.response.model == MODEL
    assert result.response.results[0].index == 1
    assert result.response.results[0].document == "Paris"
    assert result.response.usage is not None and result.response.usage.search_units == 1
    assert result.result_policy == "log_only"
    assert all(item.url is None for item in result.results)
    recorded = receipts.for_session(session.session_id)
    assert len(recorded) == 1 and recorded[0].receipt_id == result.receipt_id
    receipt = recorded[0]
    assert receipt.outcome is ToolCallOutcome.OK
    assert receipt.details.result_policy is ToolResultPolicy.LOG_ONLY
    assert receipt.details.actual_cost_usd == 0.001
    assert receipt.details.reference_cost_usd == 0.002
    assert receipt.details.extra is not None
    evidence = receipt.details.extra["actual_cost_evidence"]
    assert isinstance(evidence, dict) and evidence["provider_request_id"] == "rerank-contract"
    updated = sessions.get(session.session_id)
    assert updated is not None
    for usage in (updated.usage, usage_from_receipts(recorded)):
        assert usage.actual_total_cost_usd == 0.001
        assert usage.reference_total_cost_usd == 0.002
        assert usage.actual_cost_by_provider == {"openrouter": 0.001}
        assert usage.reference_cost_by_provider == {"openrouter": 0.002}
        assert usage.llm_usage_totals == {}
    assert result.budget.session_used_budget_usd == 0.002
    assert result.budget.session_remaining_budget_usd == 0.998
    assert result.usage is None
