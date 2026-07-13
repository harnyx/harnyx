from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.errors import ToolInvocationTimeoutError, ToolProviderError
from harnyx_commons.llm.pricing import price_embedding, price_miner_llm, price_search
from harnyx_commons.llm.provider import LlmProviderConfigurationError
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_provider_timeout_seconds
from harnyx_commons.tools.embedding_models import (
    QWEN3_CHUTES_EMBEDDING_MODEL,
    QWEN3_OPENROUTER_EMBEDDING_MODEL,
    EmbeddingUsage,
    EmbedTextRequest,
    EmbedTextResponse,
    TextEmbeddingResult,
)
from harnyx_commons.tools.executor import ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.ports import EmbeddingProviderResult
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.runtime_invoker import (
    DEFAULT_EMBEDDING_TOOL_TIMEOUT_SECONDS,
    DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS,
    DEFAULT_TOOL_LLM_TIMEOUT_SECONDS,
    effective_tool_timeout_seconds,
)
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)
from harnyx_validator.runtime.bootstrap import ALLOWED_TOOL_MODELS, RuntimeToolInvoker
from validator.tests.fixtures.fakes import FakeReceiptLog

pytestmark = pytest.mark.anyio("asyncio")
CHUTES_TOOL_MODEL = "deepseek-ai/DeepSeek-V3.2-TEE"
OPENROUTER_NATIVE_TOOL_MODEL = "deepseek/deepseek-v3.2"
OPENROUTER_TOOL_MODELS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    OPENROUTER_NATIVE_TOOL_MODEL,
    "z-ai/glm-5",
    "qwen/qwen3.6-27b",
    "google/gemma-4-31b-it",
)


def test_effective_tool_timeout_uses_request_timeout_then_runtime_default() -> None:
    assert effective_tool_timeout_seconds(
        "search_web",
        args=(),
        kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
    ) == pytest.approx(DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS)
    assert effective_tool_timeout_seconds(
        "search_web",
        args=(),
        kwargs={"provider": "parallel", "search_queries": ["harnyx"], "timeout": 5},
    ) == pytest.approx(5.0)
    llm_kwargs = {
        "provider": "chutes",
        "model": CHUTES_TOOL_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert effective_tool_timeout_seconds("llm_chat", args=(), kwargs=llm_kwargs) == pytest.approx(
        DEFAULT_TOOL_LLM_TIMEOUT_SECONDS
    )
    assert effective_tool_timeout_seconds(
        "llm_chat",
        args=(),
        kwargs={**llm_kwargs, "timeout": 7},
    ) == pytest.approx(7.0)
    assert effective_tool_timeout_seconds(
        "embed_text",
        args=(),
        kwargs={"provider": "chutes", "texts": ["harnyx"], "input_type": "query"},
    ) == pytest.approx(DEFAULT_EMBEDDING_TOOL_TIMEOUT_SECONDS)
    assert effective_tool_timeout_seconds(
        "embed_text",
        args=(),
        kwargs={"provider": "chutes", "texts": ["harnyx"], "input_type": "query", "timeout": 9},
    ) == pytest.approx(9.0)


def test_effective_tool_timeout_does_not_validate_provider_selection() -> None:
    assert effective_tool_timeout_seconds(
        "search_web",
        args=(),
        kwargs={"provider": "chutes", "search_queries": ["harnyx"], "timeout": 5},
    ) == pytest.approx(5.0)
    assert effective_tool_timeout_seconds(
        "search_web",
        args=(),
        kwargs={"provider": "chutes", "search_queries": ["harnyx"]},
    ) == pytest.approx(DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS)
    assert effective_tool_timeout_seconds(
        "llm_chat",
        args=(),
        kwargs={
            "provider": "desearch",
            "model": CHUTES_TOOL_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "timeout": 7,
        },
    ) == pytest.approx(7.0)


class StubDeSearchClient:
    def __init__(self, *, actual_cost_usd: float = 0.005) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.actual_cost_usd = actual_cost_usd
        self.search_ai_response = SearchAiSearchResponse(
            data=[
                {
                    "url": "https://example.com",
                    "title": "Example",
                    "note": "Summary",
                }
            ]
        )
        self.fetch_page_response = FetchPageResponse(
            data=[{"url": "https://example.com", "content": "page text", "title": "Example"}]
        )

    def _result(self, request: object, response: Any, *, billable_units: int) -> SearchProviderResult[Any]:
        provider = getattr(request, "provider", None)
        provider_name = provider if isinstance(provider, str) else "desearch"
        return SearchProviderResult(
            response=response,
            billing=ProviderBillingMetadata(
                actual_cost_provider=provider_name,
                actual_cost_usd=self.actual_cost_usd,
                billable_units=billable_units,
                provider_request_id="provider-request-id",
                source="response_results" if provider_name == "parallel" else "response_headers",
            ),
        )

    async def search_web(self, request: SearchWebSearchRequest) -> SearchProviderResult[SearchWebSearchResponse]:
        data = request.model_dump(exclude_none=True, exclude={"provider"})
        self.calls.append(("web", data))
        response = SearchWebSearchResponse(data=[])
        return self._result(request, response, billable_units=len(response.data))

    async def search_ai(self, request: SearchAiSearchRequest) -> SearchProviderResult[SearchAiSearchResponse]:
        data = request.model_dump(exclude_none=True, exclude={"provider"})
        self.calls.append(("search_ai", data))
        return self._result(request, self.search_ai_response, billable_units=len(self.search_ai_response.data))

    async def fetch_page(self, request: FetchPageRequest) -> SearchProviderResult[FetchPageResponse]:
        data = request.model_dump(exclude_none=True, exclude={"provider"})
        self.calls.append(("fetch_page", data))
        return self._result(request, self.fetch_page_response, billable_units=len(self.fetch_page_response.data))

    async def aclose(self) -> None:
        return None


class CostedSearchClient(StubDeSearchClient):
    def __init__(
        self,
        *,
        provider: str,
        actual_cost_usd: float = 0.0075,
        returned_results: int = 1,
    ) -> None:
        super().__init__(actual_cost_usd=actual_cost_usd)
        self.provider = provider
        self.returned_results = returned_results

    async def search_web(
        self,
        request: SearchWebSearchRequest,
    ) -> SearchProviderResult[SearchWebSearchResponse]:
        self.calls.append(("web", request.model_dump(exclude_none=True, exclude={"provider"})))
        response = SearchWebSearchResponse(
            data=[
                {
                    "link": f"https://example.com/{index}",
                    "title": f"Example {index}",
                    "snippet": "Summary",
                }
                for index in range(self.returned_results)
            ]
        )
        return self._result(request, response, billable_units=len(response.data))

    async def search_ai(
        self,
        request: SearchAiSearchRequest,
    ) -> SearchProviderResult[SearchAiSearchResponse]:
        self.calls.append(("search_ai", request.model_dump(exclude_none=True, exclude={"provider"})))
        response = SearchAiSearchResponse(
            data=[
                {
                    "url": f"https://example.com/{index}",
                    "title": f"Example {index}",
                    "note": "Summary",
                }
                for index in range(self.returned_results)
            ]
        )
        return self._result(request, response, billable_units=len(response.data))

    async def fetch_page(
        self,
        request: FetchPageRequest,
    ) -> SearchProviderResult[FetchPageResponse]:
        self.calls.append(("fetch_page", request.model_dump(exclude_none=True, exclude={"provider"})))
        response = FetchPageResponse(data=[{"url": request.url, "content": "page text", "title": "Example"}])
        return self._result(request, response, billable_units=len(response.data))


class SlowFetchPageClient(StubDeSearchClient):
    async def fetch_page(self, request: FetchPageRequest) -> SearchProviderResult[FetchPageResponse]:
        await asyncio.sleep(1.0)
        return self._result(request, self.fetch_page_response, billable_units=len(self.fetch_page_response.data))


class SlowSearchWebClient(StubDeSearchClient):
    async def search_web(self, request: SearchWebSearchRequest) -> SearchProviderResult[SearchWebSearchResponse]:
        await asyncio.sleep(1.0)
        response = SearchWebSearchResponse(data=[])
        return self._result(request, response, billable_units=len(response.data))


class CancellableSearchWebClient(StubDeSearchClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def search_web(self, request: SearchWebSearchRequest) -> SearchProviderResult[SearchWebSearchResponse]:
        self.started.set()
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        response = SearchWebSearchResponse(data=[])
        return self._result(request, response, billable_units=len(response.data))


class ProviderTimeoutSearchWebClient(StubDeSearchClient):
    async def search_web(self, request: SearchWebSearchRequest) -> SearchProviderResult[SearchWebSearchResponse]:
        raise TimeoutError("provider timed out")


def _raise_provider_response_validation_error() -> None:
    FetchPageResponse.model_validate({"data": [{"url": "https://example.com", "content": ""}]})
    raise AssertionError("invalid provider response unexpectedly validated")


class ResponseValidationFetchPageClient(StubDeSearchClient):
    async def fetch_page(self, request: FetchPageRequest) -> SearchProviderResult[FetchPageResponse]:
        data = request.model_dump(exclude_none=True, exclude={"provider"})
        self.calls.append(("fetch_page", data))
        _raise_provider_response_validation_error()


class SlowSearchAiClient(StubDeSearchClient):
    async def search_ai(self, request: SearchAiSearchRequest) -> SearchProviderResult[SearchAiSearchResponse]:
        await asyncio.sleep(1.0)
        return self._result(request, self.search_ai_response, billable_units=len(self.search_ai_response.data))


class StubChutesProvider:
    def __init__(self) -> None:
        self.calls: list[LlmRequest] = []
        self.response_payload: Mapping[str, Any] = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "id": "test",
            "model": "demo-model",
        }

    @staticmethod
    def _metadata_for_request(request: LlmRequest) -> Mapping[str, Any]:
        if request.provider == "openrouter":
            return {
                "effective_provider": "openrouter",
                "raw_response": {"usage": {"cost": 0.0042}},
                "actual_cost_provider": "openrouter",
                "actual_cost_usd": 0.0042,
                "actual_cost_evidence": {
                    "settlement_source": "provider_returned",
                    "pricing_origin": "openrouter_usage_cost",
                },
            }
        return {
            "actual_cost_provider": "chutes",
            "actual_cost_usd": 0.000123,
            "actual_cost_evidence": {
                "settlement_source": "cached_provider_pricing",
                "pricing_origin": "chutes_live_snapshot",
            },
        }

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        return LlmResponse(
            id="resp-test",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                        tool_calls=(
                            LlmMessageToolCall(
                                id="tool-call-1",
                                type="function",
                                name="lookup",
                                arguments='{"q":"hi"}',
                            ),
                        ),
                        refusal={"reason": "ignored"},
                        reasoning="ignored",
                    ),
                    finish_reason="stop",
                    logprobs={"token_logprobs": []},
                ),
            ),
            usage=LlmUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_cached_tokens=4,
                reasoning_tokens=3,
                web_search_calls=1,
            ),
            metadata=self._metadata_for_request(request),
            postprocessed={"structured": True},
            finish_reason="stop",
        )


class StubOpenRouterProvider(StubChutesProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        return LlmResponse(
            id="resp-test",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            metadata={
                "effective_provider": "openrouter",
                "raw_response": {"usage": {"cost": 0.0042}},
                "actual_cost_provider": "openrouter",
                "actual_cost_usd": 0.0042,
                "actual_cost_evidence": {
                    "settlement_source": "provider_returned",
                    "pricing_origin": "openrouter_usage_cost",
                },
            },
            finish_reason="stop",
        )


class RetryAggregateCostOpenRouterProvider(StubOpenRouterProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        response = await super().invoke(request)
        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=response.usage,
            metadata={
                **dict(response.metadata or {}),
                "actual_cost_usd": 0.02,
                "actual_cost_usd_total": 0.03,
            },
            finish_reason=response.finish_reason,
        )


class StubChutesActualCostProvider(StubChutesProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        response = await super().invoke(request)
        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=response.usage,
            metadata={
                "actual_cost_provider": "chutes",
                "actual_cost_usd": 0.000123,
                "actual_cost_evidence": {
                    "settlement_source": "cached_provider_pricing",
                    "pricing_origin": "chutes_live_snapshot",
                },
            },
            finish_reason=response.finish_reason,
        )


class StubTtftProvider(StubChutesProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        response = await super().invoke(request)
        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=response.usage,
            metadata={
                **dict(self._metadata_for_request(request)),
                "ttft_ms": 123.0,
                "provider": "chutes",
            },
            finish_reason=response.finish_reason,
        )


class SlowLlmProvider(StubChutesProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        await asyncio.sleep(1.0)
        return await super().invoke(request)


class ProviderTimeoutLlmProvider(StubChutesProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        raise TimeoutError("provider timed out")


class ProviderConfigurationErrorLlmProvider(StubChutesProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        raise LlmProviderConfigurationError("OPENROUTER_API_KEY must be configured")


class ProviderCapabilityValueErrorLlmProvider(StubChutesProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        raise ValueError("Bedrock first cut does not support tool definitions")


class StubEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[EmbedTextRequest] = []

    async def embed_text(self, request: EmbedTextRequest) -> EmbeddingProviderResult:
        self.calls.append(request)
        response = EmbedTextResponse(
            provider=request.provider,
            model=request.model,
            input_type=request.input_type,
            data=[TextEmbeddingResult(index=0, embedding=[0.1, 0.2, 0.3])],
            dimensions=3,
            usage=EmbeddingUsage(prompt_tokens=8, total_tokens=8),
        )
        return EmbeddingProviderResult(
            response=response,
            actual_cost_usd=price_embedding(request.provider, request.model, input_tokens=8),
            actual_cost_provider=request.provider,
            actual_cost_evidence={"settlement_source": "static_pricing"},
        )

    async def aclose(self) -> None:
        return None


async def _invoke(
    invoker: RuntimeToolInvoker,
    tool: str,
    args: Sequence[object] | None = None,
    kwargs: Mapping[str, object] | None = None,
    context: ToolInvocationContext | None = None,
) -> Any:
    return await invoker.invoke(
        tool,
        args=tuple(args or ()),
        kwargs=dict(kwargs or {}),
        context=context,
    )


def _tool_invocation_context() -> ToolInvocationContext:
    return ToolInvocationContext(
        receipt_id="receipt-1",
        session_id=uuid4(),
        active_attempt=0,
        uid=7,
        miner_hotkey_ss58="selected-hotkey",
    )


async def test_runtime_invoker_routes_search_payload() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="parallel",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_web",
        kwargs={"provider": "parallel", "search_queries": ["harnyx", "subnet"]},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload == {"data": []}
    assert result.actual_cost_usd == pytest.approx(0.005)
    assert result.actual_cost_provider == "parallel"
    assert stub_desearch.calls == [("web", {"search_queries": ("harnyx", "subnet")})]


async def test_runtime_invoker_uses_search_provider_resolver_for_requested_provider() -> None:
    configured_client = StubDeSearchClient()
    parallel_client = StubDeSearchClient()
    resolved_providers: list[str] = []
    resolved_contexts: list[ToolInvocationContext | None] = []
    context = _tool_invocation_context()

    def resolve_search_provider(
        provider: str,
        invocation_context: ToolInvocationContext | None,
    ) -> StubDeSearchClient:
        resolved_providers.append(provider)
        resolved_contexts.append(invocation_context)
        return parallel_client

    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=configured_client,
        web_search_provider_name="desearch",
        web_search_provider_resolver=resolve_search_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_web",
        kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
        context=context,
    )

    assert isinstance(result, ToolInvocationOutput)
    assert resolved_providers == ["parallel"]
    assert resolved_contexts == [context]
    assert parallel_client.calls == [("web", {"search_queries": ("harnyx",)})]
    assert configured_client.calls == []


async def test_runtime_invoker_search_provider_resolver_failure_does_not_fall_back_to_configured_provider() -> None:
    configured_client = StubDeSearchClient()

    def resolve_search_provider(
        provider: str,
        invocation_context: ToolInvocationContext | None,
    ) -> StubDeSearchClient:
        _ = (provider, invocation_context)
        raise ToolProviderError("selected miner search credential missing")

    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=configured_client,
        web_search_provider_name="parallel",
        web_search_provider_resolver=resolve_search_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError, match="selected miner search credential missing"):
        await _invoke(
            invoker,
            "search_web",
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            context=_tool_invocation_context(),
        )

    assert configured_client.calls == []


async def test_runtime_invoker_parallel_actual_cost_uses_provider_metadata() -> None:
    client = CostedSearchClient(provider="parallel", actual_cost_usd=0.0061, returned_results=1)
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=client,
        web_search_provider_name="parallel",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_web",
        kwargs={"provider": "parallel", "search_queries": ["harnyx"], "num": 25},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.actual_cost_usd == pytest.approx(0.0061)
    assert result.actual_cost_provider == "parallel"
    assert result.actual_cost_evidence is not None
    assert result.actual_cost_evidence["settlement_source"] == "provider_returned"
    provider_billing = result.actual_cost_evidence["provider_billing"]
    assert isinstance(provider_billing, dict)
    assert provider_billing["provider_request_id"] == "provider-request-id"
    assert provider_billing["billable_units"] == 1
    assert "provider_request_id" not in result.public_payload


async def test_runtime_invoker_desearch_actual_cost_uses_provider_billing_metadata() -> None:
    client = CostedSearchClient(provider="desearch", actual_cost_usd=0.00015, returned_results=2)
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=client,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_web",
        kwargs={"provider": "desearch", "search_queries": ["harnyx"], "num": 10},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.actual_cost_usd == pytest.approx(0.00015)
    assert result.actual_cost_provider == "desearch"
    assert result.actual_cost_evidence is not None
    assert result.actual_cost_evidence["settlement_source"] == "provider_returned"
    provider_billing = result.actual_cost_evidence["provider_billing"]
    assert isinstance(provider_billing, dict)
    assert provider_billing["source"] == "response_headers"


async def test_runtime_invoker_desearch_missing_provider_actual_cost_uses_static_pricing() -> None:
    client = CostedSearchClient(provider="desearch", actual_cost_usd=0.0075, returned_results=3)
    client.actual_cost_usd = None
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=client,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_ai",
        kwargs={"provider": "desearch", "prompt": "harnyx subnet", "count": 10},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.actual_cost_usd == pytest.approx(
        price_search("search_ai", referenceable_results=3)
    )
    assert result.actual_cost_provider == "desearch"
    assert result.actual_cost_evidence is not None
    assert result.actual_cost_evidence["settlement_source"] == "static_pricing"
    assert result.actual_cost_evidence["provider"] == "desearch"
    assert result.actual_cost_evidence["referenceable_results"] == 3


async def test_runtime_invoker_rejects_nonfinite_search_actual_cost_as_provider_error() -> None:
    client = CostedSearchClient(provider="desearch", actual_cost_usd=float("nan"), returned_results=1)
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=client,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await _invoke(
            invoker,
            "search_web",
            kwargs={"provider": "desearch", "search_queries": ["harnyx"], "num": 10},
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "actual_cost_usd must be finite" in str(exc_info.value.__cause__)


async def test_runtime_invoker_routes_search_web_timeout() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_web",
        kwargs={"provider": "desearch", "search_queries": ["harnyx"], "timeout": 5},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload == {"data": []}
    assert stub_desearch.calls == [("web", {"search_queries": ("harnyx",), "timeout": 5.0})]


async def test_runtime_invoker_enforces_search_web_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=SlowSearchWebClient(),
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolInvocationTimeoutError, match="search_web timed out after 0.01 seconds"):
        await _invoke(
            invoker,
            "search_web",
            kwargs={"provider": "desearch", "search_queries": ["harnyx"], "timeout": 0.01},
        )


async def test_runtime_invoker_cancels_timed_search_web_provider_when_parent_cancelled() -> None:
    stub_desearch = CancellableSearchWebClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    invocation = asyncio.create_task(
        _invoke(
            invoker,
            "search_web",
            kwargs={"provider": "desearch", "search_queries": ["harnyx"], "timeout": 30.0},
        )
    )
    await stub_desearch.started.wait()

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    await asyncio.wait_for(stub_desearch.cancelled.wait(), timeout=1.0)


async def test_runtime_invoker_preserves_search_web_provider_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=ProviderTimeoutSearchWebClient(),
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as excinfo:
        await _invoke(
            invoker,
            "search_web",
            kwargs={"provider": "desearch", "search_queries": ["harnyx"], "timeout": 5},
        )
    assert isinstance(excinfo.value.__cause__, TimeoutError)
    assert str(excinfo.value.__cause__) == "provider timed out"


async def test_runtime_invoker_rejects_prompt_for_search_web() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "search_web", kwargs={"provider": "desearch", "prompt": "harnyx subnet"})
    assert any(err.get("type") == "extra_forbidden" and err.get("loc") == ("prompt",) for err in excinfo.value.errors())


async def test_runtime_invoker_rejects_negative_search_web_num_before_parallel_pricing() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(
            invoker,
            "search_web",
            kwargs={"provider": "desearch", "search_queries": ["harnyx"], "num": -1},
        )

    assert any(err.get("type") == "greater_than_equal" and err.get("loc") == ("num",) for err in excinfo.value.errors())
    assert stub_desearch.calls == []


async def test_runtime_invoker_routes_fetch_page() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "fetch_page",
        kwargs={"provider": "desearch", "url": "https://example.com"},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload["data"][0]["content"] == "page text"
    assert stub_desearch.calls[-1] == ("fetch_page", {"url": "https://example.com"})


async def test_runtime_invoker_routes_fetch_page_timeout() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "fetch_page",
        kwargs={"provider": "desearch", "url": "https://example.com", "timeout": 5},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload["data"][0]["content"] == "page text"
    assert stub_desearch.calls[-1] == ("fetch_page", {"url": "https://example.com", "timeout": 5.0})


async def test_runtime_invoker_maps_provider_response_validation_to_tool_provider_error() -> None:
    client = ResponseValidationFetchPageClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=client,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as excinfo:
        await _invoke(
            invoker,
            "fetch_page",
            kwargs={"provider": "desearch", "url": "https://example.com"},
        )

    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert client.calls[-1] == ("fetch_page", {"url": "https://example.com"})


async def test_runtime_invoker_maps_timed_provider_response_validation_to_tool_provider_error() -> None:
    client = ResponseValidationFetchPageClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=client,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as excinfo:
        await _invoke(
            invoker,
            "fetch_page",
            kwargs={"provider": "desearch", "url": "https://example.com", "timeout": 5},
        )

    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert client.calls[-1] == ("fetch_page", {"url": "https://example.com", "timeout": 5.0})


async def test_runtime_invoker_enforces_fetch_page_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=SlowFetchPageClient(),
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolInvocationTimeoutError, match="fetch_page timed out after 0.01 seconds"):
        await _invoke(
            invoker,
            "fetch_page",
            kwargs={"provider": "desearch", "url": "https://example.com", "timeout": 0.01},
        )


async def test_runtime_invoker_rejects_prompt_for_fetch_page() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "fetch_page", kwargs={"provider": "desearch", "prompt": "#harnyx"})
    assert any(err.get("type") == "extra_forbidden" and err.get("loc") == ("prompt",) for err in excinfo.value.errors())


async def test_runtime_invoker_routes_search_ai() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_ai",
        kwargs={"provider": "desearch", "prompt": "harnyx subnet", "count": 10},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload["data"][0]["url"] == "https://example.com"
    assert result.public_payload["data"][0]["title"] == "Example"
    assert result.public_payload["data"][0]["note"] == "Summary"

    assert stub_desearch.calls[-1] == ("search_ai", {"prompt": "harnyx subnet", "count": 10})


async def test_runtime_invoker_routes_search_ai_timeout() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_ai",
        kwargs={"provider": "desearch", "prompt": "harnyx subnet", "count": 10, "timeout": 5},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload["data"][0]["url"] == "https://example.com"
    assert stub_desearch.calls[-1] == ("search_ai", {"prompt": "harnyx subnet", "count": 10, "timeout": 5.0})


async def test_runtime_invoker_enforces_search_ai_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=SlowSearchAiClient(),
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolInvocationTimeoutError, match="search_ai timed out after 0.01 seconds"):
        await _invoke(
            invoker,
            "search_ai",
            kwargs={"provider": "desearch", "prompt": "harnyx", "count": 10, "timeout": 0.01},
        )


async def test_runtime_invoker_rejects_repo_tools_as_unregistered() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(LookupError, match=r"tool 'search_repo' is not registered"):
        await _invoke(
            invoker,
            "search_repo",
            kwargs={
                "repo_url": "https://github.com/org/repo",
                "commit_sha": "a" * 40,
                "query": "alpha beta",
            },
        )

    with pytest.raises(LookupError, match=r"tool 'get_repo_file' is not registered"):
        await _invoke(
            invoker,
            "get_repo_file",
            kwargs={
                "repo_url": "https://github.com/org/repo",
                "commit_sha": "b" * 40,
                "path": "docs/a.md",
            },
        )


async def test_runtime_invoker_routes_embed_text() -> None:
    embedding_provider = StubEmbeddingProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        embedding_provider=embedding_provider,
        embedding_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    invocation_output = await _invoke(
        invoker,
        "embed_text",
        kwargs={
            "provider": "openrouter",
            "model": QWEN3_OPENROUTER_EMBEDDING_MODEL,
            "texts": ["What is Harnyx?"],
            "input_type": "query",
        },
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    assert invocation_output.public_payload["model"] == QWEN3_OPENROUTER_EMBEDDING_MODEL
    assert invocation_output.public_payload["input_type"] == "query"
    assert invocation_output.public_payload["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert invocation_output.actual_cost_provider == "openrouter"
    recorded = embedding_provider.calls[0]
    assert recorded.texts == ("What is Harnyx?",)
    assert recorded.input_type == "query"


@pytest.mark.parametrize("model", OPENROUTER_TOOL_MODELS)
async def test_runtime_invoker_routes_llm_chat(model: str) -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "openrouter",
            "messages": [{"role": "user", "content": "hi"}],
            "model": model,
            "temperature": 0.1,
        },
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    result = invocation_output.public_payload
    assert result["choices"][0]["message"]["content"][0]["text"] == "ok"
    assert result["choices"][0]["message"]["tool_calls"][0]["name"] == "lookup"
    assert result["usage"]["total_tokens"] == 15
    assert "metadata" not in result
    assert "postprocessed" not in result
    assert "logprobs" not in result["choices"][0]
    assert result["choices"][0]["message"]["reasoning"] == "ignored"
    assert "refusal" not in result["choices"][0]["message"]
    assert result["usage"]["prompt_cached_tokens"] == 4
    assert result["usage"]["reasoning_tokens"] == 3
    assert result["usage"]["web_search_calls"] == 1
    assert "harnyx_provider" not in result
    assert "harnyx_model" not in result
    recorded = stub_chutes.calls[0]
    assert recorded.model == model
    assert recorded.temperature == 0.1
    assert recorded.messages[0].content[0].type == "input_text"
    assert recorded.messages[0].content[0].text == "hi"
    assert recorded.provider == "openrouter"
    assert recorded.timeout_seconds == pytest.approx(120.0)


async def test_runtime_invoker_uses_llm_provider_resolver_for_requested_provider() -> None:
    configured_chutes = StubChutesProvider()
    openrouter = StubOpenRouterProvider()
    resolved_providers: list[str] = []
    resolved_contexts: list[ToolInvocationContext | None] = []
    context = _tool_invocation_context()

    async def resolve_llm_provider(
        provider: str,
        invocation_context: ToolInvocationContext | None,
    ) -> StubOpenRouterProvider:
        resolved_providers.append(provider)
        resolved_contexts.append(invocation_context)
        return openrouter

    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=configured_chutes,
        llm_provider_name="chutes",
        llm_provider_resolver=resolve_llm_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "openrouter",
            "messages": [{"role": "user", "content": "hi"}],
            "model": OPENROUTER_NATIVE_TOOL_MODEL,
        },
        context=context,
    )

    assert isinstance(result, ToolInvocationOutput)
    assert resolved_providers == ["openrouter"]
    assert resolved_contexts == [context]
    assert openrouter.calls[0].provider == "openrouter"
    assert configured_chutes.calls == []


async def test_runtime_invoker_llm_provider_resolver_failure_does_not_fall_back_to_configured_provider() -> None:
    configured_chutes = StubChutesProvider()

    async def resolve_llm_provider(
        provider: str,
        invocation_context: ToolInvocationContext | None,
    ) -> StubOpenRouterProvider:
        _ = (provider, invocation_context)
        raise ToolProviderError("selected miner llm credential missing")

    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=configured_chutes,
        llm_provider_name="openrouter",
        llm_provider_resolver=resolve_llm_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError, match="selected miner llm credential missing"):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "openrouter",
                "messages": [{"role": "user", "content": "hi"}],
                "model": OPENROUTER_NATIVE_TOOL_MODEL,
            },
            context=_tool_invocation_context(),
        )

    assert configured_chutes.calls == []


@pytest.mark.parametrize(
    ("tool_spec", "expected_message"),
    [
        ({"type": "function", "function": "not-object"}, "tools.0.function"),
        ({"type": "web_search", "config": "not-object"}, "tools.0.type"),
    ],
)
async def test_runtime_invoker_validates_llm_tools_before_resolving_provider(
    tool_spec: Mapping[str, object],
    expected_message: str,
) -> None:
    resolved_providers: list[str] = []

    async def resolve_llm_provider(
        provider: str,
        invocation_context: ToolInvocationContext | None,
    ) -> StubOpenRouterProvider:
        _ = invocation_context
        resolved_providers.append(provider)
        return StubOpenRouterProvider()

    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider_resolver=resolve_llm_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError, match=expected_message):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "openrouter",
                "messages": [{"role": "user", "content": "hi"}],
                "model": OPENROUTER_NATIVE_TOOL_MODEL,
                "tools": [tool_spec],
            },
            context=_tool_invocation_context(),
        )

    assert resolved_providers == []


async def test_runtime_invoker_routes_llm_chat_from_first_positional_payload() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    model = CHUTES_TOOL_MODEL

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        args=(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "provider": "chutes",
                "model": model,
                "temperature": 0.2,
            },
        ),
        kwargs={},
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    recorded = stub_chutes.calls[0]
    assert recorded.model == model
    assert recorded.temperature == 0.2


async def test_runtime_invoker_routes_llm_chat_short_timeout_keeps_provider_default() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    model = CHUTES_TOOL_MODEL

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "chutes",
            "messages": [{"role": "user", "content": "hi"}],
            "model": model,
            "timeout": 5,
        },
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    recorded = stub_chutes.calls[0]
    assert recorded.timeout_seconds == pytest.approx(120.0)


async def test_runtime_invoker_routes_llm_chat_long_timeout_to_provider_request() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    model = CHUTES_TOOL_MODEL

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "chutes",
            "messages": [{"role": "user", "content": "hi"}],
            "model": model,
            "timeout": 180,
        },
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    recorded = stub_chutes.calls[0]
    assert recorded.timeout_seconds == pytest.approx(platform_tool_proxy_provider_timeout_seconds(180.0))


async def test_runtime_invoker_enforces_llm_chat_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=SlowLlmProvider(),
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolInvocationTimeoutError, match="llm_chat timed out after 0.01 seconds"):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "chutes",
                "messages": [{"role": "user", "content": "hi"}],
                "model": CHUTES_TOOL_MODEL,
                "timeout": 0.01,
            },
        )


async def test_runtime_invoker_preserves_llm_chat_provider_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=ProviderTimeoutLlmProvider(),
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as excinfo:
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "chutes",
                "messages": [{"role": "user", "content": "hi"}],
                "model": CHUTES_TOOL_MODEL,
                "timeout": 5,
            },
        )
    assert isinstance(excinfo.value.__cause__, TimeoutError)
    assert str(excinfo.value.__cause__) == "provider timed out"


async def test_runtime_invoker_maps_llm_provider_error_to_tool_provider_error() -> None:
    provider = ProviderConfigurationErrorLlmProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as excinfo:
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "openrouter",
                "messages": [{"role": "user", "content": "hi"}],
                "model": OPENROUTER_NATIVE_TOOL_MODEL,
                "timeout": 5,
            },
        )
    assert isinstance(excinfo.value.__cause__, LlmProviderConfigurationError)
    assert str(excinfo.value.__cause__) == "OPENROUTER_API_KEY must be configured"
    assert len(provider.calls) == 1


async def test_runtime_invoker_preserves_llm_chat_provider_capability_value_error() -> None:
    provider = ProviderCapabilityValueErrorLlmProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValueError, match="does not support tool definitions"):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "chutes",
                "messages": [{"role": "user", "content": "hi"}],
                "model": CHUTES_TOOL_MODEL,
                "timeout": 5,
            },
        )
    assert len(provider.calls) == 1


async def test_runtime_invoker_accepts_local_tool_timeouts() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    test_result = await _invoke(invoker, "test_tool", args=("ping",), kwargs={"timeout": 5})
    tooling_result = await _invoke(invoker, "tooling_info", kwargs={"timeout": 5})

    assert test_result == {"status": "ok", "echo": "ping"}
    assert "tool_names" in tooling_result
    assert "embed_text" in tooling_result["tool_names"]
    assert tooling_result["pricing"]["embed_text"]["provider_models"]["chutes"][QWEN3_CHUTES_EMBEDDING_MODEL][
        "usd_per_second"
    ] == pytest.approx(0.0005)
    assert tooling_result["pricing"]["embed_text"]["provider_models"]["openrouter"][QWEN3_OPENROUTER_EMBEDDING_MODEL][
        "input_per_million"
    ] == pytest.approx(0.01)


async def test_runtime_invoker_prefers_kwargs_over_first_positional_payload() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    model = CHUTES_TOOL_MODEL

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        args=(
            {
                "messages": [{"role": "user", "content": "from args"}],
                "model": "unauthorized/model",
            },
        ),
        kwargs={"provider": "chutes", "messages": [{"role": "user", "content": "from kwargs"}], "model": model},
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    recorded = stub_chutes.calls[0]
    assert recorded.model == model
    assert recorded.messages[0].content[0].text == "from kwargs"


async def test_runtime_invoker_does_not_expose_internal_provider_metadata_for_llm_chat() -> None:
    stub_provider = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_provider,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "chutes",
            "messages": [{"role": "user", "content": "hi"}],
            "model": CHUTES_TOOL_MODEL,
        },
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    assert "harnyx_provider" not in invocation_output.public_payload
    assert "harnyx_model" not in invocation_output.public_payload
    assert stub_provider.calls[0].provider == "chutes"
    assert stub_provider.calls[0].model == CHUTES_TOOL_MODEL


async def test_runtime_invoker_returns_llm_chat_ttft_execution_fact() -> None:
    stub_provider = StubTtftProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_provider,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "chutes",
            "messages": [{"role": "user", "content": "hi"}],
            "model": CHUTES_TOOL_MODEL,
        },
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    assert invocation_output.execution is not None
    assert invocation_output.execution.ttft_ms == 123.0
    assert "metadata" not in invocation_output.public_payload


async def test_runtime_invoker_forwards_llm_chat_thinking_config() -> None:
    stub_provider = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_provider,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "chutes",
            "messages": [{"role": "user", "content": "hi"}],
            "model": CHUTES_TOOL_MODEL,
            "thinking": {"enabled": True, "effort": "high"},
        },
    )

    thinking = stub_provider.calls[0].thinking
    assert thinking is not None
    assert thinking.enabled is True
    assert thinking.effort == "high"
    assert thinking.budget is None


async def test_runtime_invoker_rejects_llm_chat_thinking_effort_and_budget() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=StubChutesProvider(),
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "chutes",
                "messages": [{"role": "user", "content": "hi"}],
                "model": CHUTES_TOOL_MODEL,
                "thinking": {"enabled": True, "effort": "high", "budget": 1024},
            },
        )


async def test_runtime_invoker_rejects_coerced_llm_chat_thinking_scalars() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=StubChutesProvider(),
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "chutes",
                "messages": [{"role": "user", "content": "hi"}],
                "model": CHUTES_TOOL_MODEL,
                "thinking": {"enabled": "false", "budget": True},
            },
        )


async def test_runtime_invoker_rejects_raw_llm_chat_provider_body_kwargs() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=StubChutesProvider(),
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "chutes",
                "messages": [{"role": "user", "content": "hi"}],
                "model": CHUTES_TOOL_MODEL,
                "chat_template_kwargs": {"thinking": True},
            },
        )

    assert any(
        err.get("type") == "extra_forbidden" and err.get("loc") == ("chat_template_kwargs",)
        for err in excinfo.value.errors()
    )


async def test_runtime_invoker_returns_public_payload_plus_execution_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_chutes = StubOpenRouterProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    perf_counter_values = iter((10.0, 11.25))
    monkeypatch.setattr("harnyx_commons.tools.runtime_invoker.time.perf_counter", lambda: next(perf_counter_values))

    result = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "openrouter",
            "messages": [{"role": "user", "content": "hi"}],
            "model": ALLOWED_TOOL_MODELS[0],
        },
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.execution is not None
    assert result.execution.elapsed_ms == pytest.approx(1250.0)
    assert result.actual_cost_usd == pytest.approx(0.0042)
    assert result.actual_cost_provider == "openrouter"
    assert result.actual_cost_evidence is not None
    assert result.actual_cost_evidence["settlement_source"] == "provider_returned"
    assert "elapsed_ms" not in result.public_payload
    assert "actual_cost_usd" not in result.public_payload


async def test_runtime_invoker_uses_retry_aggregate_llm_cost() -> None:
    provider = RetryAggregateCostOpenRouterProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "openrouter",
            "messages": [{"role": "user", "content": "hi"}],
            "model": ALLOWED_TOOL_MODELS[0],
        },
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.actual_cost_usd == pytest.approx(0.03)
    assert result.actual_cost_evidence is not None
    assert result.actual_cost_evidence["settlement_source"] == "retry_aggregate"
    assert result.actual_cost_evidence["final_response_actual_cost_usd"] == pytest.approx(0.02)


async def test_runtime_invoker_openrouter_missing_usage_cost_uses_static_pricing() -> None:
    class MissingUsageCostOpenRouterProvider(StubOpenRouterProvider):
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            response = await super().invoke(request)
            return LlmResponse(
                id=response.id,
                choices=response.choices,
                usage=response.usage,
                metadata={
                    "effective_provider": "openrouter",
                    "raw_response": {"usage": {}},
                    "actual_cost_provider": "openrouter",
                    "actual_cost_usd": price_miner_llm("openrouter", OPENROUTER_NATIVE_TOOL_MODEL, response.usage),
                    "actual_cost_evidence": {"settlement_source": "static_pricing"},
                },
                finish_reason=response.finish_reason,
            )

    provider = MissingUsageCostOpenRouterProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "openrouter",
            "messages": [{"role": "user", "content": "hi"}],
            "model": OPENROUTER_NATIVE_TOOL_MODEL,
        },
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.actual_cost_usd == pytest.approx(
        price_miner_llm(
            "openrouter",
            OPENROUTER_NATIVE_TOOL_MODEL,
            LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    assert result.actual_cost_provider == "openrouter"
    assert result.actual_cost_evidence is not None
    assert result.actual_cost_evidence["settlement_source"] == "static_pricing"


async def test_runtime_invoker_rejects_boolean_normalized_llm_actual_cost_as_provider_error() -> None:
    class BooleanActualCostOpenRouterProvider(StubOpenRouterProvider):
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            response = await super().invoke(request)
            return LlmResponse(
                id=response.id,
                choices=response.choices,
                usage=response.usage,
                metadata={
                    "actual_cost_provider": "openrouter",
                    "actual_cost_usd": True,
                    "actual_cost_evidence": {"settlement_source": "provider_returned"},
                },
                finish_reason=response.finish_reason,
            )

    provider = BooleanActualCostOpenRouterProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "openrouter",
                "messages": [{"role": "user", "content": "hi"}],
                "model": OPENROUTER_NATIVE_TOOL_MODEL,
            },
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "actual_cost_usd must be numeric" in str(exc_info.value.__cause__)


async def test_runtime_invoker_rejects_llm_chat_without_normalized_cost_metadata() -> None:
    class MissingSettledCostProvider(StubOpenRouterProvider):
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            response = await super().invoke(request)
            return LlmResponse(
                id=response.id,
                choices=response.choices,
                usage=response.usage,
                metadata={"effective_provider": "openrouter", "raw_response": {"usage": {"cost": 0.0042}}},
                finish_reason=response.finish_reason,
            )

    provider = MissingSettledCostProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "openrouter",
                "messages": [{"role": "user", "content": "hi"}],
                "model": OPENROUTER_NATIVE_TOOL_MODEL,
            },
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "missing settled cost" in str(exc_info.value.__cause__)


async def test_runtime_invoker_rejects_llm_chat_without_actual_cost_evidence() -> None:
    class MissingActualCostEvidenceProvider(StubOpenRouterProvider):
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            response = await super().invoke(request)
            return LlmResponse(
                id=response.id,
                choices=response.choices,
                usage=response.usage,
                metadata={
                    "actual_cost_provider": "openrouter",
                    "actual_cost_usd": 0.0042,
                },
                finish_reason=response.finish_reason,
            )

    provider = MissingActualCostEvidenceProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="openrouter",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "openrouter",
                "messages": [{"role": "user", "content": "hi"}],
                "model": OPENROUTER_NATIVE_TOOL_MODEL,
            },
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "missing settled cost" in str(exc_info.value.__cause__)


async def test_runtime_invoker_uses_chutes_actual_cost_metadata_and_keeps_payload_public() -> None:
    provider = StubChutesActualCostProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "provider": "chutes",
            "messages": [{"role": "user", "content": "hi"}],
            "model": CHUTES_TOOL_MODEL,
        },
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.actual_cost_usd == pytest.approx(0.000123)
    assert result.actual_cost_provider == "chutes"
    assert result.actual_cost_evidence == {
        "settlement_source": "cached_provider_pricing",
        "pricing_origin": "chutes_live_snapshot",
    }
    assert "actual_cost_usd" not in result.public_payload
    assert "metadata" not in result.public_payload


async def test_runtime_invoker_rejects_nonfinite_llm_actual_cost_as_provider_error() -> None:
    class NonFiniteCostChutesProvider(StubChutesProvider):
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            response = await super().invoke(request)
            return LlmResponse(
                id=response.id,
                choices=response.choices,
                usage=response.usage,
                metadata={
                    "actual_cost_provider": "chutes",
                    "actual_cost_usd": float("nan"),
                    "actual_cost_evidence": {"settlement_source": "cached_provider_pricing"},
                },
                finish_reason=response.finish_reason,
            )

    provider = NonFiniteCostChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=provider,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "chutes",
                "messages": [{"role": "user", "content": "hi"}],
                "model": CHUTES_TOOL_MODEL,
            },
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "actual_cost_usd must be finite" in str(exc_info.value.__cause__)


async def test_runtime_invoker_rejects_blank_search_ai_prompt() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="desearch",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(
            invoker,
            "search_ai",
            kwargs={"provider": "desearch", "prompt": "   ", "count": 10},
        )
    assert any(
        err.get("type") == "string_too_short" and err.get("loc") == ("prompt",) for err in excinfo.value.errors()
    )


async def test_runtime_invoker_rejects_missing_clients() -> None:
    invoker = RuntimeToolInvoker(FakeReceiptLog(), allowed_models=ALLOWED_TOOL_MODELS)

    with pytest.raises(LookupError):
        await _invoke(invoker, "search_web", kwargs={})

    with pytest.raises(LookupError):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={"provider": "chutes", "messages": [{"role": "user", "content": "hi"}], "model": "demo"},
        )


async def test_runtime_invoker_blocks_disallowed_models() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValueError, match="not supported for miner-selected provider"):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "provider": "chutes",
                "messages": [{"role": "user", "content": "hi"}],
                "model": "unauthorized/model",
            },
        )
