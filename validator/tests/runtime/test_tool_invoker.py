from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from caster_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from caster_commons.tools.desearch import (
    DeSearchAiDateFilter,
    DeSearchAiModel,
    DeSearchAiResultType,
    DeSearchAiTool,
)
from caster_commons.tools.search_models import (
    SearchWebSearchRequest,
    SearchWebSearchResponse,
    SearchXResult,
    SearchXSearchRequest,
    SearchXSearchResponse,
)
from caster_validator.runtime.bootstrap import ALLOWED_TOOL_MODELS, RuntimeToolInvoker
from validator.tests.fixtures.fakes import FakeReceiptLog

pytestmark = pytest.mark.anyio("asyncio")


class StubDeSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.ai_search_response: Mapping[str, Any] = {
            "youtube_search": [
                {
                    "title": "Example",
                    "link": "https://example.com",
                    "snippet": "Summary",
                }
            ],
            "completion": "hello",
        }

    async def post(self, endpoint: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        data = dict(payload)
        self.calls.append((endpoint, data))
        return {"endpoint": endpoint, "payload": data}

    async def search_links_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        data = request.model_dump(exclude_none=True)
        self.calls.append(("web", data))
        return SearchWebSearchResponse(data=[])

    async def search_links_twitter(
        self,
        request: SearchXSearchRequest,
    ) -> SearchXSearchResponse:
        data = request.model_dump(exclude_none=True)
        self.calls.append(("twitter", data))
        return SearchXSearchResponse(data=[])

    async def fetch_twitter_post(self, *, post_id: str) -> SearchXResult | None:
        self.calls.append(("twitter_post", {"id": post_id}))
        return None

    async def ai_search(
        self,
        *,
        prompt: str,
        tools: tuple[DeSearchAiTool, ...],
        model: DeSearchAiModel,
        count: int,
        date_filter: DeSearchAiDateFilter | None,
        result_type: DeSearchAiResultType,
        system_message: str,
    ) -> Mapping[str, Any]:
        self.calls.append(
            (
                "ai_search",
                {
                    "prompt": prompt,
                    "tools": [tool.value for tool in tools],
                    "model": model.value,
                    "count": count,
                    "date_filter": date_filter.value if date_filter is not None else None,
                    "result_type": result_type.value,
                    "system_message": system_message,
                },
            )
        )
        return self.ai_search_response


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
                    ),
                ),
            ),
            usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
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
        search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(invoker, "search_web", kwargs={"query": "caster subnet"})

    assert result == {"data": []}
    assert stub_desearch.calls == [("web", {"query": "caster subnet"})]


async def test_runtime_invoker_rejects_prompt_for_search_web() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "search_web", kwargs={"prompt": "caster subnet"})
    assert any(
        err.get("type") == "extra_forbidden" and err.get("loc") == ("prompt",)
        for err in excinfo.value.errors()
    )


async def test_runtime_invoker_routes_search_x() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(invoker, "search_x", kwargs={"query": "#caster"})

    assert result == {"data": []}
    assert stub_desearch.calls[-1] == ("twitter", {"query": "#caster"})


async def test_runtime_invoker_rejects_prompt_for_search_x() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "search_x", kwargs={"prompt": "#caster"})
    assert any(
        err.get("type") == "extra_forbidden" and err.get("loc") == ("prompt",)
        for err in excinfo.value.errors()
    )


async def test_runtime_invoker_routes_search_ai() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_ai",
        kwargs={"prompt": "caster subnet", "tools": ["youtube"], "count": 1},
    )

    assert result["data"][0]["url"] == "https://example.com"
    assert result["data"][0]["title"] == "Example"
    assert result["data"][0]["note"] == "Summary"
    assert result["data"][0]["source"] == "youtube"

    assert stub_desearch.calls[-1][0] == "ai_search"
    assert stub_desearch.calls[-1][1]["prompt"] == "caster subnet"
    assert stub_desearch.calls[-1][1]["tools"] == ["youtube"]
    assert stub_desearch.calls[-1][1]["model"] == "HORIZON"
    assert stub_desearch.calls[-1][1]["result_type"] == "LINKS_WITH_FINAL_SUMMARY"


async def test_runtime_invoker_routes_search_ai_docs_response() -> None:
    stub_desearch = StubDeSearchClient()
    stub_desearch.ai_search_response = {
        "search": [
            {
                "title": "Example",
                "link": "https://example.com",
                "snippet": "Snippet",
            }
        ],
        "tweets": [
            {
                "id": "123",
                "url": "https://x.com/foo/status/123",
                "text": "hi",
                "user": {"username": "foo"},
            }
        ],
        "completion": "hello",
    }
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_ai",
        kwargs={"prompt": "caster subnet", "tools": ["web", "twitter"], "count": 2},
    )

    assert result["data"][0]["url"] == "https://example.com"
    assert result["data"][0]["title"] == "Example"
    assert result["data"][0]["note"] == "Snippet"
    assert result["data"][0]["source"] == "web"

    assert result["data"][1]["url"] == "https://x.com/foo/status/123"
    assert result["data"][1]["note"] == "hi"
    assert result["data"][1]["source"] == "twitter"


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
    assert result["usage"]["total_tokens"] == 15
    recorded = stub_chutes.calls[0]
    assert recorded.model == ALLOWED_TOOL_MODELS[0]
    assert recorded.temperature == 0.1
    assert recorded.messages[0].content[0].type == "input_text"
    assert recorded.messages[0].content[0].text == "hi"
    assert recorded.provider == "chutes"


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
