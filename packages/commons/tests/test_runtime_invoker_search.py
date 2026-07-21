from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from harnyx_commons.domain.session import ProviderCredentialSource
from harnyx_commons.errors import (
    ProviderCredentialUnavailableError,
    ToolInvocationTimeoutError,
    ToolProviderError,
    ToolProviderFailureCode,
)
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.tools import runtime_invoker as runtime_invoker_module
from harnyx_commons.tools.executor import ToolInvocationContext
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.runtime_invoker import DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS, RuntimeToolInvoker
from harnyx_commons.tools.search_models import SearchWebResult, SearchWebSearchRequest, SearchWebSearchResponse

pytestmark = pytest.mark.anyio("asyncio")


class _CapturingSearchProvider:
    def __init__(self) -> None:
        self.requests: list[SearchWebSearchRequest] = []

    async def search_web(self, request: SearchWebSearchRequest) -> SearchProviderResult[SearchWebSearchResponse]:
        self.requests.append(request)
        return SearchProviderResult(
            response=SearchWebSearchResponse(
                data=[SearchWebResult(link="https://example.com", title="Example", snippet="Result")]
            ),
            billing=ProviderBillingMetadata(
                actual_cost_provider=request.provider,
                source="response_results",
            ),
        )

    async def aclose(self) -> None:
        return None


class _BlockingSearchProvider(_CapturingSearchProvider):
    async def search_web(self, request: SearchWebSearchRequest) -> SearchProviderResult[SearchWebSearchResponse]:
        self.requests.append(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _context(source: ProviderCredentialSource) -> ToolInvocationContext:
    return ToolInvocationContext(
        receipt_id=f"receipt-{source.value}",
        session_id=uuid4(),
        active_attempt=0,
        uid=1,
        provider_credential_source=source,
    )


@pytest.mark.parametrize("provider", ["desearch", "parallel"])
async def test_platform_credential_session_resolves_requested_search_without_miner_fallback(provider: str) -> None:
    platform_provider = _CapturingSearchProvider()
    miner_resolver_calls: list[str] = []
    platform_resolver_calls: list[str] = []

    def miner_resolver(requested: str, _context: ToolInvocationContext | None) -> _CapturingSearchProvider:
        miner_resolver_calls.append(requested)
        return _CapturingSearchProvider()

    def platform_resolver(requested: str, _context: ToolInvocationContext | None) -> _CapturingSearchProvider:
        platform_resolver_calls.append(requested)
        return platform_provider

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        web_search_provider_resolver=miner_resolver,
        platform_web_search_provider_resolver=platform_resolver,
    )

    await invoker.invoke(
        "search_web",
        args=(),
        kwargs={"provider": provider, "search_queries": ["harnyx"]},
        context=_context(ProviderCredentialSource.PLATFORM),
    )

    assert platform_resolver_calls == [provider]
    assert miner_resolver_calls == []
    assert len(platform_provider.requests) == 1
    assert platform_provider.requests[0].timeout == DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS


async def test_context_free_search_uses_matching_direct_provider_without_resolver() -> None:
    direct_provider = _CapturingSearchProvider()
    resolver_calls: list[str] = []

    def resolver(requested: str, _context: ToolInvocationContext | None) -> _CapturingSearchProvider:
        resolver_calls.append(requested)
        return _CapturingSearchProvider()

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        web_search_client=direct_provider,
        web_search_provider_name="desearch",
        web_search_provider_resolver=resolver,
    )

    await invoker.invoke(
        "search_web",
        args=(),
        kwargs={"provider": "desearch", "search_queries": ["harnyx"]},
    )

    assert len(direct_provider.requests) == 1
    assert resolver_calls == []


async def test_miner_credential_search_uses_miner_resolver_without_direct_fallback() -> None:
    direct_provider = _CapturingSearchProvider()
    miner_provider = _CapturingSearchProvider()
    resolver_calls: list[str] = []

    def resolver(requested: str, _context: ToolInvocationContext | None) -> _CapturingSearchProvider:
        resolver_calls.append(requested)
        return miner_provider

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        web_search_client=direct_provider,
        web_search_provider_name="desearch",
        web_search_provider_resolver=resolver,
    )

    await invoker.invoke(
        "search_web",
        args=(),
        kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
        context=_context(ProviderCredentialSource.MINER),
    )

    assert resolver_calls == ["parallel"]
    assert len(miner_provider.requests) == 1
    assert direct_provider.requests == []


async def test_platform_search_credential_source_error_is_mapped_at_invoker_boundary() -> None:
    def resolver(requested: str, _context: ToolInvocationContext | None) -> _CapturingSearchProvider:
        raise ProviderCredentialUnavailableError(requested)

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        platform_web_search_provider_resolver=resolver,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            context=_context(ProviderCredentialSource.PLATFORM),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE
    assert exc_info.value.provider == "parallel"


@pytest.mark.parametrize(
    "credential_source",
    [ProviderCredentialSource.MINER, ProviderCredentialSource.PLATFORM],
)
async def test_omitted_search_timeout_uses_same_outer_deadline_for_every_credential_source(
    monkeypatch: pytest.MonkeyPatch,
    credential_source: ProviderCredentialSource,
) -> None:
    provider = _BlockingSearchProvider()
    monkeypatch.setattr(runtime_invoker_module, "DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS", 0.01)

    def resolver(_requested: str, _context: ToolInvocationContext | None) -> _BlockingSearchProvider:
        return provider

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        web_search_provider_resolver=resolver,
        platform_web_search_provider_resolver=resolver,
    )

    with pytest.raises(ToolInvocationTimeoutError, match="search_web timed out after 0.01 seconds"):
        await asyncio.wait_for(
            invoker.invoke(
                "search_web",
                args=(),
                kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
                context=_context(credential_source),
            ),
            timeout=1.0,
        )

    assert provider.requests[0].timeout == 0.01
