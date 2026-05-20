from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from harnyx_commons.errors import ToolInvocationTimeoutError, ToolProviderError
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
from harnyx_commons.tools.executor import ToolInvocationOutput
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


class StubDeSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
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

    async def search_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        data = request.model_dump(exclude_none=True)
        self.calls.append(("web", data))
        return SearchWebSearchResponse(data=[])

    async def search_ai(self, request: SearchAiSearchRequest) -> SearchAiSearchResponse:
        data = request.model_dump(exclude_none=True)
        self.calls.append(("search_ai", data))
        return self.search_ai_response

    async def fetch_page(self, request: FetchPageRequest) -> FetchPageResponse:
        data = request.model_dump(exclude_none=True)
        self.calls.append(("fetch_page", data))
        return self.fetch_page_response

    async def aclose(self) -> None:
        return None


class SlowFetchPageClient(StubDeSearchClient):
    async def fetch_page(self, request: FetchPageRequest) -> FetchPageResponse:
        await asyncio.sleep(1.0)
        return self.fetch_page_response


class SlowSearchWebClient(StubDeSearchClient):
    async def search_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        await asyncio.sleep(1.0)
        return SearchWebSearchResponse(data=[])


class CancellableSearchWebClient(StubDeSearchClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def search_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        self.started.set()
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return SearchWebSearchResponse(data=[])


class ProviderTimeoutSearchWebClient(StubDeSearchClient):
    async def search_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        raise TimeoutError("provider timed out")


class SlowSearchAiClient(StubDeSearchClient):
    async def search_ai(self, request: SearchAiSearchRequest) -> SearchAiSearchResponse:
        await asyncio.sleep(1.0)
        return self.search_ai_response


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
            metadata={"provider": "chutes"},
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
            },
            finish_reason="stop",
        )


class StubTtftProvider(StubChutesProvider):
    async def invoke(self, request: LlmRequest) -> LlmResponse:
        response = await super().invoke(request)
        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=response.usage,
            metadata={"ttft_ms": 123.0, "provider": "chutes"},
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


async def _invoke(
    invoker: RuntimeToolInvoker,
    tool: str,
    args: Sequence[object] | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> Any:
    return await invoker.invoke(
        tool,
        args=tuple(args or ()),
        kwargs=dict(kwargs or {}),
    )


async def test_runtime_invoker_routes_search_payload() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="parallel",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(invoker, "search_web", kwargs={"search_queries": ["harnyx", "subnet"]})

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload == {"data": []}
    assert result.actual_cost_usd == pytest.approx(0.005)
    assert result.actual_cost_provider == "parallel"
    assert stub_desearch.calls == [("web", {"search_queries": ("harnyx", "subnet")})]


async def test_runtime_invoker_routes_search_web_timeout() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(invoker, "search_web", kwargs={"search_queries": ["harnyx"], "timeout": 5})

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload == {"data": []}
    assert stub_desearch.calls == [("web", {"search_queries": ("harnyx",), "timeout": 5.0})]


async def test_runtime_invoker_enforces_search_web_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=SlowSearchWebClient(),
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolInvocationTimeoutError, match="search_web timed out after 0.01 seconds"):
        await _invoke(invoker, "search_web", kwargs={"search_queries": ["harnyx"], "timeout": 0.01})


async def test_runtime_invoker_cancels_timed_search_web_provider_when_parent_cancelled() -> None:
    stub_desearch = CancellableSearchWebClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    invocation = asyncio.create_task(
        _invoke(invoker, "search_web", kwargs={"search_queries": ["harnyx"], "timeout": 30.0})
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
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolProviderError) as excinfo:
        await _invoke(invoker, "search_web", kwargs={"search_queries": ["harnyx"], "timeout": 5})
    assert isinstance(excinfo.value.__cause__, TimeoutError)
    assert str(excinfo.value.__cause__) == "provider timed out"


async def test_runtime_invoker_rejects_prompt_for_search_web() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "search_web", kwargs={"prompt": "harnyx subnet"})
    assert any(
        err.get("type") == "extra_forbidden" and err.get("loc") == ("prompt",)
        for err in excinfo.value.errors()
    )


async def test_runtime_invoker_rejects_negative_search_web_num_before_parallel_pricing() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        web_search_provider_name="parallel",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "search_web", kwargs={"search_queries": ["harnyx"], "num": -1})

    assert any(
        err.get("type") == "greater_than_equal" and err.get("loc") == ("num",)
        for err in excinfo.value.errors()
    )
    assert stub_desearch.calls == []


async def test_runtime_invoker_routes_fetch_page() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(invoker, "fetch_page", kwargs={"url": "https://example.com"})

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload["data"][0]["content"] == "page text"
    assert stub_desearch.calls[-1] == ("fetch_page", {"url": "https://example.com"})


async def test_runtime_invoker_routes_fetch_page_timeout() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(invoker, "fetch_page", kwargs={"url": "https://example.com", "timeout": 5})

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload["data"][0]["content"] == "page text"
    assert stub_desearch.calls[-1] == ("fetch_page", {"url": "https://example.com", "timeout": 5.0})


async def test_runtime_invoker_enforces_fetch_page_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=SlowFetchPageClient(),
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolInvocationTimeoutError, match="fetch_page timed out after 0.01 seconds"):
        await _invoke(invoker, "fetch_page", kwargs={"url": "https://example.com", "timeout": 0.01})


async def test_runtime_invoker_rejects_prompt_for_fetch_page() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "fetch_page", kwargs={"prompt": "#harnyx"})
    assert any(
        err.get("type") == "extra_forbidden" and err.get("loc") == ("prompt",)
        for err in excinfo.value.errors()
    )


async def test_runtime_invoker_routes_search_ai() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_ai",
        kwargs={"prompt": "harnyx subnet", "count": 10},
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
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_ai",
        kwargs={"prompt": "harnyx subnet", "count": 10, "timeout": 5},
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.public_payload["data"][0]["url"] == "https://example.com"
    assert stub_desearch.calls[-1] == ("search_ai", {"prompt": "harnyx subnet", "count": 10, "timeout": 5.0})


async def test_runtime_invoker_enforces_search_ai_timeout() -> None:
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=SlowSearchAiClient(),
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ToolInvocationTimeoutError, match="search_ai timed out after 0.01 seconds"):
        await _invoke(invoker, "search_ai", kwargs={"prompt": "harnyx", "count": 10, "timeout": 0.01})


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


@pytest.mark.parametrize("model", ALLOWED_TOOL_MODELS)
async def test_runtime_invoker_routes_llm_chat(model: str) -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
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
    assert recorded.provider == "chutes"
    assert recorded.timeout_seconds == pytest.approx(120.0)


async def test_runtime_invoker_routes_llm_chat_from_first_positional_payload() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    model = ALLOWED_TOOL_MODELS[0]

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        args=(
            {
                "messages": [{"role": "user", "content": "hi"}],
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


async def test_runtime_invoker_routes_llm_chat_timeout_without_changing_provider_timeout() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    model = ALLOWED_TOOL_MODELS[0]

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "messages": [{"role": "user", "content": "hi"}],
            "model": model,
            "timeout": 5,
        },
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    recorded = stub_chutes.calls[0]
    assert recorded.timeout_seconds == pytest.approx(120.0)


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
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
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
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
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
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
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
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
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


async def test_runtime_invoker_prefers_kwargs_over_first_positional_payload() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    model = ALLOWED_TOOL_MODELS[0]

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        args=(
            {
                "messages": [{"role": "user", "content": "from args"}],
                "model": "unauthorized/model",
            },
        ),
        kwargs={"messages": [{"role": "user", "content": "from kwargs"}], "model": model},
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
        llm_provider_name="vertex",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    invocation_output = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "messages": [{"role": "user", "content": "hi"}],
            "model": ALLOWED_TOOL_MODELS[0],
        },
    )

    assert isinstance(invocation_output, ToolInvocationOutput)
    assert "harnyx_provider" not in invocation_output.public_payload
    assert "harnyx_model" not in invocation_output.public_payload
    assert stub_provider.calls[0].provider == "vertex"
    assert stub_provider.calls[0].model == ALLOWED_TOOL_MODELS[0]


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
            "messages": [{"role": "user", "content": "hi"}],
            "model": ALLOWED_TOOL_MODELS[0],
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
            "messages": [{"role": "user", "content": "hi"}],
            "model": ALLOWED_TOOL_MODELS[0],
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
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
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
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
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
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
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
            "messages": [{"role": "user", "content": "hi"}],
            "model": ALLOWED_TOOL_MODELS[0],
        },
    )

    assert isinstance(result, ToolInvocationOutput)
    assert result.execution is not None
    assert result.execution.elapsed_ms == pytest.approx(1250.0)
    assert result.actual_cost_usd == pytest.approx(0.0042)
    assert result.actual_cost_provider == "openrouter"
    assert "elapsed_ms" not in result.public_payload
    assert "actual_cost_usd" not in result.public_payload


async def test_runtime_invoker_rejects_blank_search_ai_prompt() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(
            invoker,
            "search_ai",
            kwargs={"prompt": "   ", "count": 10},
        )
    assert any(
        err.get("type") == "string_too_short" and err.get("loc") == ("prompt",)
        for err in excinfo.value.errors()
    )


async def test_runtime_invoker_rejects_missing_clients() -> None:
    invoker = RuntimeToolInvoker(FakeReceiptLog(), allowed_models=ALLOWED_TOOL_MODELS)

    with pytest.raises(LookupError):
        await _invoke(invoker, "search_web", kwargs={})

    with pytest.raises(LookupError):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={"messages": [{"role": "user", "content": "hi"}], "model": "demo"},
        )


async def test_runtime_invoker_blocks_disallowed_models() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValueError, match="not allowed"):
        await _invoke(
            invoker,
            "llm_chat",
            kwargs={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "unauthorized/model",
            },
        )
