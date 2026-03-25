from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmUsage,
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


async def _invoke(
    invoker: RuntimeToolInvoker,
    tool: str,
    args: Sequence[object] | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> Mapping[str, Any]:
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
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(invoker, "search_web", kwargs={"search_queries": ["harnyx", "subnet"]})

    assert result == {"data": []}
    assert stub_desearch.calls == [("web", {"search_queries": ("harnyx", "subnet")})]


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


async def test_runtime_invoker_routes_fetch_page() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        web_search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(invoker, "fetch_page", kwargs={"url": "https://example.com"})

    assert result["data"][0]["content"] == "page text"
    assert stub_desearch.calls[-1] == ("fetch_page", {"url": "https://example.com"})


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
        kwargs={"prompt": "harnyx subnet", "count": 1},
    )

    assert result["data"][0]["url"] == "https://example.com"
    assert result["data"][0]["title"] == "Example"
    assert result["data"][0]["note"] == "Summary"

    assert stub_desearch.calls[-1] == ("search_ai", {"prompt": "harnyx subnet", "count": 1})


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


async def test_runtime_invoker_routes_llm_chat() -> None:
    stub_chutes = StubChutesProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        llm_provider=stub_chutes,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "llm_chat",
        kwargs={
            "messages": [{"role": "user", "content": "hi"}],
            "model": ALLOWED_TOOL_MODELS[0],
            "temperature": 0.1,
        },
    )

    assert result["choices"][0]["message"]["content"][0]["text"] == "ok"
    assert result["choices"][0]["message"]["tool_calls"][0]["name"] == "lookup"
    assert result["usage"]["total_tokens"] == 15
    assert "metadata" not in result
    assert "postprocessed" not in result
    assert "logprobs" not in result["choices"][0]
    assert "reasoning" not in result["choices"][0]["message"]
    assert "refusal" not in result["choices"][0]["message"]
    assert result["usage"]["prompt_cached_tokens"] == 4
    assert result["usage"]["reasoning_tokens"] == 3
    assert result["usage"]["web_search_calls"] == 1
    recorded = stub_chutes.calls[0]
    assert recorded.model == ALLOWED_TOOL_MODELS[0]
    assert recorded.temperature == 0.1
    assert recorded.messages[0].content[0].type == "input_text"
    assert recorded.messages[0].content[0].text == "hi"
    assert recorded.provider == "chutes"


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
            kwargs={"prompt": "   ", "count": 1},
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
