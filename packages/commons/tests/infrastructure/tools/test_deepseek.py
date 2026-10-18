"""Protect DeepSeek's wire contract and per-request settlement end to end."""

import json
from uuid import uuid4

import httpx
import pytest

from harnyx_commons.domain.session import ProviderCredentialSource
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.tools.deepseek import DeepSeekClient
from harnyx_commons.tools.executor import ToolInvocationContext
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker
from harnyx_commons.tools.search_models import SearchWebSearchRequest

pytestmark = pytest.mark.anyio("asyncio")


@pytest.mark.parametrize("count", [0, 1, 5])
async def test_search_bills_one_nonempty_request_regardless_of_result_count(count):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"results": [
            {"url": f"https://example.com/{i}", "title": "Example", "snippet": "short",
             "summary": "long summary", "content": "", "rerank_score": 2.4,
             "site_name": None, "site_icon": None, "date": None, "authority": -1} for i in range(count)
        ]})

    async with httpx.AsyncClient(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(base_url="https://api.deepseek.com", api_key="miner-key", client=http)
        invoker = RuntimeToolInvoker(
            InMemoryReceiptLog(), web_search_client=client, web_search_provider_name="deepseek",
        )
        result = await invoker.invoke("search_web", args=(), kwargs={
            "provider": "deepseek", "search_queries": ["one", "two"], "num": 5,
            "provider_extra": {"ori_question": "original question", "localization": "8.8.8.8"},
        })
    assert requests[0].url.path == "/v1/tools/search"
    assert requests[0].headers["authorization"] == "Bearer miner-key"
    assert json.loads(requests[0].content) == {
        "query": "(one) OR (two)", "count": 5, "need_content": False,
        "ori_question": "original question", "localization": "8.8.8.8",
    }
    assert len(result.public_payload["data"]) == count
    if count:
        assert result.public_payload["data"][0]["snippet"] == "long summary"
    assert result.actual_cost_usd == pytest.approx(0.00075 if count else 0)
    assert result.actual_cost_provider == "deepseek"


@pytest.mark.parametrize("content", ["# Page\n\nBody", ""])
async def test_fetch_maps_markdown_and_bills_only_nonempty_content(content):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"title": "Page", "content": content})

    async with httpx.AsyncClient(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(base_url="https://api.deepseek.com", api_key="miner-key", client=http)
        invoker = RuntimeToolInvoker(
            InMemoryReceiptLog(), fetch_page_client=client, fetch_page_provider_name="deepseek",
        )
        result = await invoker.invoke("fetch_page", args=(), kwargs={"provider": "deepseek", "url": "https://example.com"})
    assert requests[0].url.path == "/v1/tools/fetch"
    assert json.loads(requests[0].content) == {"url": "https://example.com"}
    assert result.actual_cost_usd == pytest.approx(0.00075 if content else 0)
    if content:
        assert result.public_payload["data"][0]["content"] == content


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 408, 429, 500])
async def test_errors_preserve_auth_failure_meaning_and_retry_only_transient_statuses(status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "provider error"}})

    async with httpx.AsyncClient(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(
            base_url="https://api.deepseek.com", api_key="key", client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        with pytest.raises(ToolProviderError) as error:
            await client.search_web(SearchWebSearchRequest(provider="deepseek", search_queries=("query",)))
    assert len(requests) == (2 if status in {408, 429, 500} else 1)
    assert error.value.failure_code == (
        ToolProviderFailureCode.AUTHENTICATION_FAILED if status == 401 else ToolProviderFailureCode.PROVIDER_FAILED
    )
    assert error.value.billing is not None
    assert error.value.billing.actual_cost_usd == 0.0


async def test_zero_requested_results_never_contacts_provider():
    def handler(request):
        raise AssertionError("Zero results must not issue a paid request")

    async with httpx.AsyncClient(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(base_url="https://api.deepseek.com", api_key="key", client=http)
        result = await client.search_web(SearchWebSearchRequest(provider="deepseek", search_queries=("query",), num=0))
    assert result.response.data == []
    assert result.billing.billable_units == 0


@pytest.mark.parametrize("raw", [{}, {"results": None}, {"results": [{"url": "https://example.com"}]}])
async def test_malformed_search_response_fails_instead_of_settling_as_free(raw):
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw)),
    ) as http:
        client = DeepSeekClient(base_url="https://api.deepseek.com", api_key="key", client=http)
        with pytest.raises(ToolProviderError, match="response invalid"):
            await client.search_web(SearchWebSearchRequest(provider="deepseek", search_queries=("query",)))


@pytest.mark.parametrize("prior_transport_failure", [False, True])
@pytest.mark.parametrize("source", [ProviderCredentialSource.MINER, ProviderCredentialSource.PLATFORM])
async def test_rejection_reports_zero_only_when_all_attempts_are_confirmed_unbilled(prior_transport_failure, source):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if prior_transport_failure and calls == 1:
            raise httpx.ReadTimeout("unknown outcome", request=request)
        return httpx.Response(402, json={"error": "insufficient balance"})

    async with httpx.AsyncClient(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(
            base_url="https://api.deepseek.com", api_key="key", client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        invoker = RuntimeToolInvoker(
            InMemoryReceiptLog(), web_search_client=client, web_search_provider_name="deepseek",
            web_search_provider_resolver=lambda *_: client,
            platform_web_search_provider_resolver=lambda *_: client,
        )
        with pytest.raises(ToolProviderError) as error:
            await invoker.invoke(
                "search_web", args=(), kwargs={"provider": "deepseek", "search_queries": ["query"]},
                context=ToolInvocationContext(
                    receipt_id="receipt", session_id=uuid4(), active_attempt=0, uid=1,
                    provider_credential_source=source,
                ),
            )
    if prior_transport_failure:
        assert error.value.billing is None
    else:
        assert error.value.billing is not None
        assert error.value.billing.actual_cost_usd == 0
        assert error.value.billing.billable_units == 0


@pytest.mark.parametrize("operation", ["search", "fetch"])
@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout, httpx.ReadError, httpx.WriteTimeout, httpx.WriteError, httpx.RemoteProtocolError,
    httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout,
])
async def test_transport_retries_never_repeat_a_possibly_billed_deepseek_request(operation, failure):
    from harnyx_commons.tools.search_models import FetchPageRequest

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure("transport failure", request=request)
        return httpx.Response(200, json={
            "results": [{
                "url": "https://example.com", "title": "Page", "snippet": "", "summary": "body", "content": "",
                "rerank_score": 2.4, "site_name": None, "site_icon": None, "date": None, "authority": -1,
            }],
        } if operation == "search" else {"title": "Page", "content": "body"})

    safe_to_retry = failure in {httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout}
    async with httpx.AsyncClient(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(
            base_url="https://api.deepseek.com", api_key="key", client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        async def invoke():
            if operation == "search":
                return await client.search_web(SearchWebSearchRequest(provider="deepseek", search_queries=("query",)))
            return await client.fetch_page(FetchPageRequest(provider="deepseek", url="https://example.com"))
        if safe_to_retry:
            result = await invoke()
            assert result.billing.billable_units == 1
        else:
            with pytest.raises(ToolProviderError) as error:
                await invoke()
            assert error.value.billing is None
    assert calls == (2 if safe_to_retry else 1)


def _complete_search_result():
    return {
        "rerank_score": 2.4, "url": "https://example.com", "title": "Page",
        "snippet": "short", "summary": "long summary", "content": "",
        "site_name": "Example", "site_icon": "https://example.com/favicon.ico",
        "date": "2026-10-17T09:00:12.045Z", "authority": 3,
    }


@pytest.mark.parametrize("missing", [
    "rerank_score", "url", "title", "snippet", "summary", "content", "site_name", "site_icon", "date", "authority",
])
async def test_search_rejects_missing_required_fields_even_when_nullable(missing):
    result = _complete_search_result()
    del result[missing]
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"results": [result]})),
    ) as http:
        client = DeepSeekClient(base_url="https://api.deepseek.com", api_key="key", client=http)
        with pytest.raises(ToolProviderError, match="response invalid") as error:
            await client.search_web(SearchWebSearchRequest(provider="deepseek", search_queries=("query",)))
    assert error.value.billing is None


@pytest.mark.parametrize("field,value", [
    ("rerank_score", -0.1), ("rerank_score", 3.1), ("rerank_score", "2.4"),
    ("rerank_score", True), ("rerank_score", None),
    ("rerank_score", float("nan")), ("rerank_score", float("inf")),
    ("authority", -2), ("authority", 5), ("authority", 1.5), ("authority", True),
    ("authority", "3"), ("authority", None),
    ("site_name", 1), ("site_icon", []),
    ("date", "2026-02-30T00:00:00Z"), ("date", "2026-10-17T09:00:00"),
    ("date", "2026-10-17"), ("date", 1792227600), ("date", "1792227600"), ("date", ""),
])
async def test_search_rejects_metadata_outside_documented_wire_contract(field, value):
    result = _complete_search_result() | {field: value}
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps({"results": [result]}))),
    ) as http:
        client = DeepSeekClient(base_url="https://api.deepseek.com", api_key="key", client=http)
        with pytest.raises(ToolProviderError, match="response invalid") as error:
            await client.search_web(SearchWebSearchRequest(provider="deepseek", search_queries=("query",)))
    assert error.value.billing is None


@pytest.mark.parametrize("score,authority", [(0, -1), (3, 4)])
@pytest.mark.parametrize("date", [
    None, "2026-10-17T09:00:12Z", "2026-10-17T09:00:12.045+00:00", "2026-10-17t09:00:12z",
])
async def test_search_preserves_documented_nulls_bounds_and_additive_fields(score, authority, date):
    result = _complete_search_result() | {
        "rerank_score": score, "authority": authority, "site_name": None, "site_icon": None,
        "date": date, "future_metadata": {"new": True},
    }
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"results": [result]})),
    ) as http:
        client = DeepSeekClient(base_url="https://api.deepseek.com", api_key="key", client=http)
        response = await client.search_web(SearchWebSearchRequest(provider="deepseek", search_queries=("query",)))
    assert response.response.data[0].snippet == "long summary"
    assert response.billing.billable_units == 1
