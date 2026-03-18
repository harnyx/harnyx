from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.tools.desearch import (
    DeSearchAiDateFilter,
    DeSearchAiModel,
    DeSearchAiResultType,
    DeSearchAiTool,
)
from harnyx_commons.tools.search_models import (
    SearchWebSearchRequest,
    SearchWebSearchResponse,
    SearchXResult,
    SearchXSearchRequest,
    SearchXSearchResponse,
)
from harnyx_validator.runtime.bootstrap import ALLOWED_TOOL_MODELS, RuntimeToolInvoker
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


class StubRepoSearchProvider:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.get_file_calls: list[dict[str, object]] = []
        self.search_response: Mapping[str, Any] = {
            "data": [
                {
                    "path": "docs/z.md",
                    "bm25": 2.0,
                    "url": "https://github.com/org/repo/blob/sha/docs/z.md",
                    "excerpt": "z" * 1_200,
                    "title": "docs/z.md",
                    "text": "this field must not be exposed by search_repo",
                },
                {
                    "path": "docs/a.md",
                    "bm25": 2.0,
                    "url": "https://github.com/org/repo/blob/sha/docs/a.md",
                    "excerpt": "alpha",
                    "title": "docs/a.md",
                },
                {
                    "path": "docs/b.md",
                    "bm25": 1.5,
                    "url": "https://github.com/org/repo/blob/sha/docs/b.md",
                    "excerpt": "beta",
                    "title": "docs/b.md",
                },
            ]
        }
        self.get_file_response: Mapping[str, Any] = {
            "data": [
                {
                    "path": "docs/a.md",
                    "url": "https://github.com/org/repo/blob/sha/docs/a.md",
                    "text": "full file text",
                    "excerpt": "e" * 1_500,
                    "title": "docs/a.md",
                }
            ]
        }

    async def search_repo(
        self,
        *,
        repo_url: str,
        commit_sha: str,
        query: str,
        path_glob: str | None,
        limit: int,
    ) -> Mapping[str, Any]:
        self.search_calls.append(
            {
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "query": query,
                "path_glob": path_glob,
                "limit": limit,
            }
        )
        return self.search_response

    async def get_repo_file(
        self,
        *,
        repo_url: str,
        commit_sha: str,
        path: str,
        start_line: int | None,
        end_line: int | None,
    ) -> Mapping[str, Any]:
        self.get_file_calls.append(
            {
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
            }
        )
        return self.get_file_response


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

    result = await _invoke(invoker, "search_web", kwargs={"query": "harnyx subnet"})

    assert result == {"data": []}
    assert stub_desearch.calls == [("web", {"query": "harnyx subnet"})]


async def test_runtime_invoker_rejects_prompt_for_search_web() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "search_web", kwargs={"prompt": "harnyx subnet"})
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

    result = await _invoke(invoker, "search_x", kwargs={"query": "#harnyx"})

    assert result == {"data": []}
    assert stub_desearch.calls[-1] == ("twitter", {"query": "#harnyx"})


async def test_runtime_invoker_rejects_prompt_for_search_x() -> None:
    stub_desearch = StubDeSearchClient()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        search_client=stub_desearch,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    with pytest.raises(ValidationError) as excinfo:
        await _invoke(invoker, "search_x", kwargs={"prompt": "#harnyx"})
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
        kwargs={"prompt": "harnyx subnet", "tools": ["youtube"], "count": 1},
    )

    assert result["data"][0]["url"] == "https://example.com"
    assert result["data"][0]["title"] == "Example"
    assert result["data"][0]["note"] == "Summary"
    assert result["data"][0]["source"] == "youtube"

    assert stub_desearch.calls[-1][0] == "ai_search"
    assert stub_desearch.calls[-1][1]["prompt"] == "harnyx subnet"
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
        kwargs={"prompt": "harnyx subnet", "tools": ["web", "twitter"], "count": 2},
    )

    assert result["data"][0]["url"] == "https://example.com"
    assert result["data"][0]["title"] == "Example"
    assert result["data"][0]["note"] == "Snippet"
    assert result["data"][0]["source"] == "web"

    assert result["data"][1]["url"] == "https://x.com/foo/status/123"
    assert result["data"][1]["note"] == "hi"
    assert result["data"][1]["source"] == "twitter"


async def test_runtime_invoker_routes_search_repo_with_deterministic_ordering_and_excerpt_cap() -> None:
    repo_provider = StubRepoSearchProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        repo_search_provider=repo_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "search_repo",
        kwargs={
            "repo_url": "https://github.com/org/repo",
            "commit_sha": "a" * 40,
            "query": "alpha beta",
            "path_glob": "docs/*.md",
            "limit": 10,
        },
    )

    assert repo_provider.search_calls == [
        {
            "repo_url": "https://github.com/org/repo",
            "commit_sha": "a" * 40,
            "query": "alpha beta",
            "path_glob": "docs/*.md",
            "limit": 10,
        }
    ]
    assert [item["path"] for item in result["data"]] == ["docs/b.md", "docs/a.md", "docs/z.md"]
    assert len(result["data"][2]["excerpt"]) == 1_000
    assert "text" not in result["data"][0]
    assert "text" not in result["data"][1]
    assert "text" not in result["data"][2]


async def test_runtime_invoker_routes_get_repo_file_with_text_and_excerpt_cap() -> None:
    repo_provider = StubRepoSearchProvider()
    invoker = RuntimeToolInvoker(
        FakeReceiptLog(),
        repo_search_provider=repo_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )

    result = await _invoke(
        invoker,
        "get_repo_file",
        kwargs={
            "repo_url": "https://github.com/org/repo",
            "commit_sha": "b" * 40,
            "path": "docs/a.md",
            "start_line": 10,
            "end_line": 20,
        },
    )

    assert repo_provider.get_file_calls == [
        {
            "repo_url": "https://github.com/org/repo",
            "commit_sha": "b" * 40,
            "path": "docs/a.md",
            "start_line": 10,
            "end_line": 20,
        }
    ]
    assert result["data"][0]["text"] == "full file text"
    assert len(result["data"][0]["excerpt"]) == 1_000


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
            "search_repo",
            kwargs={
                "repo_url": "https://github.com/org/repo",
                "commit_sha": "a" * 40,
                "query": "harnyx",
            },
        )

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
