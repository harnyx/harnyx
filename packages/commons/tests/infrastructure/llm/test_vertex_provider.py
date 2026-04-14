from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from typing import Any, cast

import pytest
from pydantic import BaseModel

from harnyx_commons.clients import CHUTES
from harnyx_commons.llm.provider_types import normalize_reasoning_effort
from harnyx_commons.llm.providers.openai_stream import (
    OpenAiChoiceState,
    OpenAiStreamError,
    OpenAiStreamState,
    OpenAiToolCallState,
    _OpenAiStreamEvent,
    _OpenAiToolCallDelta,
    normalize_openai_text_fragments,
)
from harnyx_commons.llm.providers.vertex.codec import (
    _VertexMaasChatRequest,
    _VertexMaasChatResponse,
    build_choices,
    normalize_messages,
    resolve_thinking_config,
    vertex_maas_openai_chat_model_name,
)
from harnyx_commons.llm.providers.vertex.provider import VertexLlmProvider, _vertex_stream_text_fragments
from harnyx_commons.llm.schema import (
    GroundedLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmTool,
    LlmUsage,
)

pytestmark = pytest.mark.anyio("asyncio")


class FakeUsage:
    def __init__(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_token_count = prompt
        self.cached_content_token_count = None
        self.candidates_token_count = completion
        self.thoughts_token_count = None
        self.total_token_count = total


class FakeResponse:
    def __init__(self) -> None:
        self.text = "ok"
        self.response_id = "fake-response-id"
        self.usage_metadata = FakeUsage(12, 5, 17)
        self.candidates = [self._candidate()]

    @staticmethod
    def _candidate() -> Any:
        class _FunctionCall:
            id = None
            name = "lookup"
            args = {"query": "harnyx"}

        class _Part:
            text = "ok"
            function_call = _FunctionCall()
            thought = False
            thought_signature = None

        class _Content:
            parts = [_Part()]

        class _Candidate:
            content = _Content()
            finish_reason = None
            grounding_metadata = None

        return _Candidate()

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"text": self.text}


@pytest.mark.parametrize("code", [500, 502, 503, 504, "500"])
def test_vertex_classify_stream_error_preserves_server_retry_policy(code: int | str) -> None:
    exc = OpenAiStreamError(
        message="temporarily unavailable",
        error_type="server_error",
        code=code,
    )

    retryable, reason = VertexLlmProvider._classify_exception(exc)

    assert retryable is True
    assert reason == f"stream_error:{code}:server_error:temporarily unavailable"


@pytest.fixture(autouse=True)
def anthropic_clients(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    created: list[Any] = []

    class _FailingMessages:
        async def create(self, **kwargs: Any) -> Any:
            raise AssertionError(f"unexpected AsyncAnthropicVertex.messages.create call: {kwargs!r}")

        def stream(self, **kwargs: Any) -> Any:
            raise AssertionError(f"unexpected AsyncAnthropicVertex.messages.stream call: {kwargs!r}")

    class _FakeAsyncAnthropicVertex:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.messages = _FailingMessages()
            self.closed = False
            created.append(self)

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "harnyx_commons.llm.providers.vertex.provider.AsyncAnthropicVertex",
        _FakeAsyncAnthropicVertex,
    )
    return created


def _patch_google_client(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    *,
    response_factory: Callable[[], Any] = FakeResponse,
) -> None:
    class _FakeAsyncModels:
        async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
            latest = {
                "model": model,
                "contents": contents,
                "config": config,
            }
            captured["model_call"] = latest
            return response_factory()

        async def generate_content_stream(self, *, model: str, contents: Any, config: Any) -> Any:
            latest = {
                "model": model,
                "contents": contents,
                "config": config,
            }
            captured["model_stream_call"] = latest

            async def _stream() -> Any:
                response = response_factory()
                chunks = response if isinstance(response, list) else [response]
                for chunk in chunks:
                    yield chunk

            return _stream()

    class _FakeAsyncClient:
        def __init__(self) -> None:
            self.models = _FakeAsyncModels()

        async def aclose(self) -> None:
            captured["google_async_closed"] = True

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.aio = _FakeAsyncClient()

        def close(self) -> None:
            captured["google_sync_closed"] = True

    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.provider.genai.Client", _FakeClient)


def _patch_vertex_maas_http_client(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    *,
    response_payload: dict[str, Any] | None = None,
) -> None:
    payload = response_payload or {
        "id": "chatcmpl-vertex",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "56",
                    "reasoning_content": "I need to multiply 7 by 8.",
                    "tool_calls": None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "reasoning_tokens": 5,
            "total_tokens": 15,
        },
    }

    class _FakeHttpResponse:
        def __init__(self, json_payload: dict[str, Any]) -> None:
            self._json_payload = json_payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self) -> Any:
            yield f"data: {json.dumps(self._json_payload)}"
            yield ""
            yield "data: [DONE]"
            yield ""

    class _FakeStreamContext:
        def __init__(self, response: _FakeHttpResponse) -> None:
            self._response = response

        async def __aenter__(self) -> _FakeHttpResponse:
            return self._response

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    class _FakeAsyncHttpClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["http_client_kwargs"] = kwargs

        def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamContext:
            captured["http_call"] = {"method": method, "url": url, **kwargs}
            return _FakeStreamContext(_FakeHttpResponse(payload))

        async def aclose(self) -> None:
            captured["http_closed"] = True

    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.provider.httpx.AsyncClient", _FakeAsyncHttpClient)


def _async_return(value: Any) -> Callable[[], Any]:
    async def _inner() -> Any:
        return value

    return _inner


async def test_vertex_provider_invokes_generative_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=CHUTES.timeout_seconds,
    )

    request = LlmRequest(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-20b-maas",
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("stay concise"),),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hi"),),
            ),
        ),
        temperature=0.3,
        max_output_tokens=256,
        output_mode="json_object",
        tools=(
            LlmTool(
                type="function",
                function={
                    "name": "lookup",
                    "description": "Lookup info",
                    "parameters": {"type": "object", "properties": {}},
                },
            ),
        ),
        tool_choice="required",
    )

    response = await provider.invoke(request)

    client_kwargs = captured["client_kwargs"]
    assert client_kwargs["project"] == "demo-project"
    assert client_kwargs["location"] == "us-central1"
    http_options = client_kwargs["http_options"]
    assert http_options.api_version == "v1beta1"
    assert client_kwargs["credentials"] is None

    model_call = captured["model_stream_call"]
    assert model_call["model"] == "publishers/openai/models/gpt-oss-20b-maas"
    assert model_call["contents"][0].role == "user"
    config = model_call["config"]
    assert config.system_instruction == "stay concise"
    assert config.temperature == pytest.approx(0.3)
    assert config.max_output_tokens == 256
    assert config.response_mime_type == "application/json"
    assert config.tools and len(config.tools) == 1
    assert config.tool_config.function_calling_config.mode.name == "ANY"
    assert config.thinking_config is None

    assert response.raw_text == "ok"
    assert response.usage.total_tokens == 17
    tool_calls = response.tool_calls
    assert tool_calls[0].name == "lookup"
    assert response.metadata is not None
    raw_response = response.metadata["raw_response"]
    assert isinstance(raw_response, dict)
    assert raw_response["text"] == "ok"
    assert "ttft_ms" not in response.metadata

    records = [record for record in caplog.records if record.message == "llm.vertex.stream.ttft"]
    assert records
    data = records[0].__dict__["data"]
    assert data["branch"] == "gemini"
    assert isinstance(data["ttft_ms"], float)
    assert data["ttft_ms"] >= 0.0


async def test_vertex_provider_gemini_stream_aggregates_text_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    class _Usage(FakeUsage):
        pass

    class _GroundingMetadata:
        web_search_queries = ["harnyx subnet"]

    class _ThoughtPart:
        text = "reasoning step"
        function_call = None
        thought = True
        thought_signature = "sig-1"

    class _TextPartHello:
        text = "Hello "
        function_call = None
        thought = False
        thought_signature = None

    class _TextPartWorld:
        text = "world"
        function_call = None
        thought = False
        thought_signature = None

    class _ChunkOneContent:
        parts = [_ThoughtPart(), _TextPartHello()]

    class _ChunkTwoContent:
        parts = [_TextPartWorld()]

    class _FinishReason:
        value = "STOP"

    class _ChunkOneCandidate:
        content = _ChunkOneContent()
        finish_reason = None
        grounding_metadata = _GroundingMetadata()

    class _ChunkTwoCandidate:
        content = _ChunkTwoContent()
        finish_reason = _FinishReason()
        grounding_metadata = None

    class _ChunkOne:
        text = "Hello "
        response_id = "gemini-stream-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkOneCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {
                "text": "Hello ",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thought": True,
                                    "text": "reasoning step",
                                    "thought_signature": "sig-1",
                                },
                                {
                                    "text": "Hello ",
                                },
                            ]
                        },
                        "grounding_metadata": {
                            "web_search_queries": ["harnyx subnet"],
                        },
                    }
                ],
            }

    class _ChunkTwo:
        text = "world"
        response_id = "gemini-stream-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkTwoCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {
                "text": "world",
                "candidates": [
                    {
                        "finish_reason": "STOP",
                        "content": {
                            "parts": [
                                {
                                    "text": "world",
                                }
                            ]
                        }
                    }
                ],
            }

    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured, response_factory=lambda: [_ChunkOne(), _ChunkTwo()])

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = LlmRequest(
        provider="vertex",
        model="gemini-2.5-pro",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
    )

    response = await provider.invoke(request)

    assert response.raw_text == "Hello world"
    assert response.usage.web_search_calls == 1
    assert response.metadata is not None
    assert response.metadata["web_search_queries"] == ("harnyx subnet",)
    raw_response = response.metadata["raw_response"]
    assert raw_response["text"] == "Hello world"
    assert raw_response["candidates"][0]["grounding_metadata"]["web_search_queries"] == ["harnyx subnet"]
    assert raw_response["candidates"][0]["content"]["parts"][0]["thought_signature"] == "sig-1"
    assert raw_response["candidates"][0]["finish_reason"] == "STOP"


async def test_vertex_provider_gemini_stream_preserves_reasoning_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    class _Usage(FakeUsage):
        pass

    class _ThoughtPartOne:
        text = "think "
        function_call = None
        thought = True
        thought_signature = "sig-1"

    class _ThoughtPartTwo:
        text = "more"
        function_call = None
        thought = True
        thought_signature = "sig-2"

    class _TextPart:
        text = "done"
        function_call = None
        thought = False
        thought_signature = None

    class _ChunkOneContent:
        parts = [_ThoughtPartOne()]

    class _ChunkTwoContent:
        parts = [_ThoughtPartTwo(), _TextPart()]

    class _FinishReason:
        value = "STOP"

    class _ChunkOneCandidate:
        content = _ChunkOneContent()
        finish_reason = None
        grounding_metadata = None

    class _ChunkTwoCandidate:
        content = _ChunkTwoContent()
        finish_reason = _FinishReason()
        grounding_metadata = None

    class _ChunkOne:
        text = ""
        response_id = "gemini-stream-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkOneCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {
                "text": "",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thought": True,
                                    "text": "think ",
                                    "thought_signature": "sig-1",
                                }
                            ]
                        }
                    }
                ],
            }

    class _ChunkTwo:
        text = "done"
        response_id = "gemini-stream-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkTwoCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {
                "text": "done",
                "candidates": [
                    {
                        "finish_reason": "STOP",
                        "content": {
                            "parts": [
                                {
                                    "thought": True,
                                    "text": "more",
                                    "thought_signature": "sig-2",
                                },
                                {
                                    "text": "done",
                                },
                            ]
                        },
                    }
                ],
            }

    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured, response_factory=lambda: [_ChunkOne(), _ChunkTwo()])

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = LlmRequest(
        provider="vertex",
        model="gemini-2.5-pro",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
    )

    response = await provider.invoke(request)

    assert response.raw_text == "done"
    assert response.choices[0].message.reasoning == "think more"


async def test_vertex_provider_gemini_stream_dedupes_repeated_search_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    class _Usage(FakeUsage):
        pass

    class _GroundingMetadata:
        web_search_queries = ["harnyx subnet"]

    class _TextPartOne:
        text = "Hello "
        function_call = None
        thought = False
        thought_signature = None

    class _TextPartTwo:
        text = "world"
        function_call = None
        thought = False
        thought_signature = None

    class _ChunkOneContent:
        parts = [_TextPartOne()]

    class _ChunkTwoContent:
        parts = [_TextPartTwo()]

    class _ChunkOneCandidate:
        content = _ChunkOneContent()
        finish_reason = None
        grounding_metadata = _GroundingMetadata()

    class _ChunkTwoCandidate:
        content = _ChunkTwoContent()
        finish_reason = None
        grounding_metadata = _GroundingMetadata()

    class _ChunkOne:
        text = "Hello "
        response_id = "gemini-stream-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkOneCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {
                "text": "Hello ",
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hello "}]},
                        "grounding_metadata": {"web_search_queries": ["harnyx subnet"]},
                    }
                ],
            }

    class _ChunkTwo:
        text = "world"
        response_id = "gemini-stream-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkTwoCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {
                "text": "world",
                "candidates": [
                    {
                        "content": {"parts": [{"text": "world"}]},
                        "grounding_metadata": {"web_search_queries": ["harnyx subnet"]},
                    }
                ],
            }

    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured, response_factory=lambda: [_ChunkOne(), _ChunkTwo()])

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    response = await provider.invoke(
        LlmRequest(
            provider="vertex",
            model="gemini-2.5-pro",
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text("hello"),),
                ),
            ),
            temperature=None,
            max_output_tokens=64,
            output_mode="text",
        )
    )

    assert response.raw_text == "Hello world"
    assert response.usage.web_search_calls == 1
    assert response.metadata is not None
    assert response.metadata["web_search_queries"] == ("harnyx subnet",)


async def test_vertex_provider_gemini_stream_merges_snapshot_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    class _Usage(FakeUsage):
        pass

    class _FunctionCallOne:
        id = "call-1"
        name = "lookup"
        args = {"query": "har"}

    class _FunctionCallTwo:
        id = "call-1"
        name = "lookup"
        args = {"query": "harnyx"}

    class _PartOne:
        text = None
        function_call = _FunctionCallOne()
        thought = False
        thought_signature = None

    class _PartTwo:
        text = None
        function_call = _FunctionCallTwo()
        thought = False
        thought_signature = None

    class _ContentOne:
        parts = [_PartOne()]

    class _ContentTwo:
        parts = [_PartTwo()]

    class _ChunkOneCandidate:
        content = _ContentOne()
        finish_reason = None
        grounding_metadata = None

    class _FinishReason:
        value = "STOP"

    class _ChunkTwoCandidate:
        content = _ContentTwo()
        finish_reason = _FinishReason()
        grounding_metadata = None

    class _ChunkOne:
        text = None
        response_id = "gemini-tool-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkOneCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {"candidates": [{"content": {"parts": [{}]}}]}

    class _ChunkTwo:
        text = None
        response_id = "gemini-tool-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkTwoCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {"candidates": [{"content": {"parts": [{}]}, "finish_reason": "STOP"}]}

    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured, response_factory=lambda: [_ChunkOne(), _ChunkTwo()])

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    response = await provider.invoke(
        LlmRequest(
            provider="vertex",
            model="gemini-2.5-pro",
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text("hello"),),
                ),
            ),
            temperature=None,
            max_output_tokens=64,
            output_mode="text",
        )
    )

    assert response.raw_text is None
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "harnyx"}
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].id == "call-1"


async def test_vertex_provider_gemini_stream_overwrites_partial_tool_call_same_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    class _Usage(FakeUsage):
        pass

    class _FunctionCallOne:
        id = None
        name = None
        args = {"query": "har"}

    class _FunctionCallTwo:
        id = "call-1"
        name = "lookup"
        args = {"query": "harnyx"}

    class _PartOne:
        text = None
        function_call = _FunctionCallOne()
        thought = False
        thought_signature = None

    class _PartTwo:
        text = None
        function_call = _FunctionCallTwo()
        thought = False
        thought_signature = None

    class _ContentOne:
        parts = [_PartOne()]

    class _ContentTwo:
        parts = [_PartTwo()]

    class _ChunkOneCandidate:
        content = _ContentOne()
        finish_reason = None
        grounding_metadata = None

    class _FinishReason:
        value = "STOP"

    class _ChunkTwoCandidate:
        content = _ContentTwo()
        finish_reason = _FinishReason()
        grounding_metadata = None

    class _ChunkOne:
        text = None
        response_id = "gemini-tool-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkOneCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {"candidates": [{"content": {"parts": [{}]}}]}

    class _ChunkTwo:
        text = None
        response_id = "gemini-tool-response"
        usage_metadata = _Usage(12, 5, 17)
        candidates = [_ChunkTwoCandidate()]

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {"candidates": [{"content": {"parts": [{}]}, "finish_reason": "STOP"}]}

    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured, response_factory=lambda: [_ChunkOne(), _ChunkTwo()])

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    response = await provider.invoke(
        LlmRequest(
            provider="vertex",
            model="gemini-2.5-pro",
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text("hello"),),
                ),
            ),
            temperature=None,
            max_output_tokens=64,
            output_mode="text",
        )
    )

    assert response.raw_text is None
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "harnyx"}
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) == 1
    assert response.choices[0].message.tool_calls[0].id == "call-1"


def test_openai_choice_state_skips_tool_call_without_function_name() -> None:
    state = OpenAiChoiceState(
        tool_calls={
            0: OpenAiToolCallState(
                id="tc-1",
                type="function",
                arguments_text='{"query":"harnyx"}',
            )
        }
    )

    assert state.tool_call_values() is None


def test_openai_tool_call_state_replaces_dict_argument_snapshots() -> None:
    state = OpenAiToolCallState(id="tc-1", type="function", name="lookup")

    assert state.merge_delta(
        _OpenAiToolCallDelta.model_validate(
            {"function": {"arguments": {"query": "a"}}}
        )
    )
    assert state.merge_delta(
        _OpenAiToolCallDelta.model_validate(
            {"function": {"arguments": {"query": "ab"}}}
        )
    )

    tool_call = state.to_tool_call(index=0)
    assert tool_call is not None
    assert tool_call.arguments == '{"query": "ab"}'


async def test_vertex_maas_gpt_oss_routes_to_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)
    _patch_vertex_maas_http_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )
    monkeypatch.setattr(provider, "_vertex_maas_access_token", _async_return("access-token"))

    request = LlmRequest(
        provider="vertex-maas",
        model="publishers/openai/models/gpt-oss-120b-maas",
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("stay concise"),),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("What is 7 times 8?"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
        reasoning_effort="high",
    )

    response = await provider.invoke(request)

    assert "model_stream_call" not in captured
    http_call = captured["http_call"]
    assert http_call["method"] == "POST"
    assert http_call["url"].endswith("/endpoints/openapi/chat/completions")
    assert http_call["headers"]["Authorization"] == "Bearer access-token"
    payload = http_call["json"]
    assert payload["model"] == "openai/gpt-oss-120b-maas"
    assert payload["stream"] is True
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 64
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert response.raw_text == "56"
    assert response.choices[0].message.reasoning == "I need to multiply 7 by 8."
    assert response.usage.reasoning_tokens == 5
    assert response.metadata is not None
    assert "ttft_ms" not in response.metadata

    records = [record for record in caplog.records if record.message == "llm.vertex.stream.ttft"]
    assert records
    data = records[0].__dict__["data"]
    assert data["branch"] == "vertex_maas_openai"
    assert isinstance(data["ttft_ms"], float)
    assert data["ttft_ms"] >= 0.0


async def test_vertex_provider_keeps_vertex_gpt_oss_on_generate_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)
    _patch_vertex_maas_http_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )
    monkeypatch.setattr(provider, "_vertex_maas_access_token", _async_return("access-token"))

    request = LlmRequest(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-120b-maas",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=32,
    )

    await provider.invoke(request)

    assert "model_stream_call" in captured
    assert "http_call" not in captured


async def test_vertex_provider_raw_response_metadata_is_json_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    thought_signature = "ZmFrZS10aG91Z2h0LXNpZw=="

    class _RawResponseWithThoughtSignature(FakeResponse):
        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            if mode == "json":
                return {
                    "text": self.text,
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "thought_signature": thought_signature,
                                    }
                                ]
                            }
                        }
                    ],
                }
            return {
                "text": self.text,
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thought_signature": b"\xff\xfe",
                                }
                            ]
                        }
                    }
                ],
            }

    _patch_google_client(
        monkeypatch,
        {},
        response_factory=_RawResponseWithThoughtSignature,
    )

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = LlmRequest(
        provider="vertex",
        model="gemini-2.5-pro",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
        reasoning_effort="high",
    )

    response = await provider.invoke(request)

    assert response.metadata is not None
    raw_response = response.metadata["raw_response"]
    assert isinstance(raw_response, dict)
    signature_value = raw_response["candidates"][0]["content"]["parts"][0]["thought_signature"]
    assert isinstance(signature_value, str)
    assert signature_value == thought_signature
    payload_signature = response.payload["metadata"]["raw_response"]["candidates"][0]["content"]["parts"][0][
        "thought_signature"
    ]
    assert payload_signature == thought_signature


async def test_vertex_provider_normalizes_assistant_and_tool_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = LlmRequest(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-20b-maas",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("user-turn"),),
            ),
            LlmMessage(
                role="assistant",
                content=(LlmMessageContentPart.input_text("assistant-turn"),),
            ),
            LlmMessage(
                role="tool",
                content=(LlmMessageContentPart.input_text("tool-turn"),),
            ),
        ),
        temperature=None,
        max_output_tokens=128,
        output_mode="text",
    )

    await provider.invoke(request)

    contents = captured["model_stream_call"]["contents"]
    assert [entry.role for entry in contents] == ["user", "model", "user"]


def test_vertex_codec_fails_fast_on_unknown_request_role() -> None:
    with pytest.raises(ValueError, match="unsupported Vertex request role: 'critic'"):
        normalize_messages(
            (
                LlmMessage(
                    role=cast(Any, "critic"),
                    content=(LlmMessageContentPart.input_text("invalid-role"),),
                ),
            )
        )


def test_vertex_resolve_thinking_config_returns_none_when_effort_is_null() -> None:
    config = resolve_thinking_config(model="gemini-2.5-pro", reasoning_effort=None)
    assert config is None


def test_vertex_resolve_thinking_config_returns_none_when_effort_is_zero() -> None:
    config = resolve_thinking_config(model="gemini-2.5-pro", reasoning_effort="0")
    assert config is None


def test_normalize_reasoning_effort_rejects_non_positive_budgets() -> None:
    assert normalize_reasoning_effort(None) is None
    assert normalize_reasoning_effort("  ") is None
    assert normalize_reasoning_effort("0") is None
    assert normalize_reasoning_effort("-1") is None
    assert normalize_reasoning_effort("high") == "high"
    assert normalize_reasoning_effort(" 256 ") == "256"


def test_vertex_resolve_thinking_config_sets_level_with_include_thoughts() -> None:
    config = resolve_thinking_config(model="gemini-2.5-pro", reasoning_effort="high")
    assert config is not None
    assert config.include_thoughts is True
    assert config.thinking_level is not None


def test_vertex_codec_build_choices_separates_thought_text_from_assistant_output() -> None:
    class _ThoughtPart:
        text = "deliberation"
        function_call = None
        thought = True
        thought_signature = "sig-1"

    class _AssistantPart:
        text = "final answer"
        function_call = None
        thought = False
        thought_signature = None

    class _Content:
        parts = [_ThoughtPart(), _AssistantPart()]

    class _Candidate:
        content = _Content()
        finish_reason = None
        grounding_metadata = None

    class _Response:
        candidates = [_Candidate()]

    choices = build_choices(_Response())
    message = choices[0].message

    assert tuple(part.text for part in message.content) == ("final answer",)
    assert message.reasoning == "deliberation"


def test_vertex_codec_build_choices_preserves_signature_only_text_as_output() -> None:
    class _ThoughtPart:
        text = "deliberation"
        function_call = None
        thought = False
        thought_signature = "sig-2"

    class _AssistantPart:
        text = "final answer"
        function_call = None
        thought = False
        thought_signature = None

    class _Content:
        parts = [_ThoughtPart(), _AssistantPart()]

    class _Candidate:
        content = _Content()
        finish_reason = None
        grounding_metadata = None

    class _Response:
        candidates = [_Candidate()]

    choices = build_choices(_Response())
    message = choices[0].message

    assert tuple(part.text for part in message.content) == ("deliberation", "final answer")
    assert message.reasoning is None


class _StructuredPairwisePreference(BaseModel):
    preferred_position: str


def test_vertex_maas_chat_payload_supports_structured_output() -> None:
    request = LlmRequest(
        provider="vertex-maas",
        model="publishers/openai/models/gpt-oss-120b-maas",
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("Return JSON."),),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Choose first or second."),),
            ),
        ),
        output_mode="structured",
        output_schema=_StructuredPairwisePreference,
        temperature=None,
        max_output_tokens=64,
        reasoning_effort="high",
    )

    payload = _VertexMaasChatRequest.from_request(request).model_dump(mode="python", exclude_none=True)

    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "_StructuredPairwisePreference"
    assert payload["reasoning_effort"] == "high"
    assert "temperature" not in payload


def test_vertex_maas_response_payload_maps_reasoning_tool_calls_and_usage() -> None:
    payload = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"preferred_position":"first"}',
                        }
                    ],
                    "reasoning_content": [{"text": "I should prefer the first answer."}],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "paris"},
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 14,
            "completion_tokens": 7,
            "reasoning_tokens": 3,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }

    response = _VertexMaasChatResponse.model_validate(payload).to_llm_response()

    assert response.id == "chatcmpl-123"
    assert response.raw_text == '{"preferred_position":"first"}'
    assert response.choices[0].message.reasoning == "I should prefer the first answer."
    assert response.choices[0].message.tool_calls[0].arguments == '{"query": "paris"}'
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "paris"}
    assert response.usage.prompt_tokens == 14
    assert response.usage.prompt_cached_tokens == 2
    assert response.usage.completion_tokens == 7
    assert response.usage.reasoning_tokens == 3
    assert response.usage.total_tokens == 24


def test_openai_stream_state_deduplicates_vertex_reasoning_keys_per_event() -> None:
    state = OpenAiStreamState()
    event = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "reasoning": "step",
                    "reasoning_content": "step",
                },
            }
        ],
    }

    merged = state.merge_event(
        event=_OpenAiStreamEvent.model_validate(event),
        reasoning_keys=("reasoning_content", "reasoning"),
    )

    assert merged is True
    payload = _VertexMaasChatResponse.from_stream_state(state)
    assert payload.raw_payload() == {
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "message": {"content": "", "reasoning_content": "step"}}],
        "usage": None,
    }


def test_openai_stream_state_preserves_vertex_multipart_join_semantics() -> None:
    state = OpenAiStreamState()
    event = _OpenAiStreamEvent.model_validate(
        {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": [
                            {"text": "first paragraph"},
                            {"text": "second paragraph"},
                        ],
                        "reasoning_content": [
                            {"text": "step one"},
                            {"text": "step two"},
                        ],
                    },
                }
            ],
        }
    )

    merged = state.merge_event(
        event=event,
        reasoning_keys=("reasoning_content", "reasoning"),
        normalize_content_fragment=lambda value: normalize_openai_text_fragments(value, multipart_joiner="\n\n"),
        normalize_reasoning_fragment=lambda value: normalize_openai_text_fragments(value, multipart_joiner="\n\n"),
    )

    assert merged is True
    payload = _VertexMaasChatResponse.from_stream_state(state)
    assert payload.raw_payload() == {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": "first paragraph\n\nsecond paragraph",
                    "reasoning_content": "step one\n\nstep two",
                },
            }
        ],
        "usage": None,
    }


def test_vertex_maas_response_payload_preserves_multi_event_interleaving() -> None:
    state = OpenAiStreamState()

    first_event = _OpenAiStreamEvent.model_validate(
        {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "first ",
                        "reasoning_content": "think-1",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"q":',
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )
    second_event = _OpenAiStreamEvent.model_validate(
        {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "second",
                        "reasoning": "think-2",
                        "reasoning_content": "think-2",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": ' "paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
    )

    assert state.merge_event(
        first_event,
        reasoning_keys=("reasoning_content", "reasoning"),
        normalize_content_fragment=_vertex_stream_text_fragments,
        normalize_reasoning_fragment=_vertex_stream_text_fragments,
    )
    assert state.merge_event(
        second_event,
        reasoning_keys=("reasoning_content", "reasoning"),
        normalize_content_fragment=_vertex_stream_text_fragments,
        normalize_reasoning_fragment=_vertex_stream_text_fragments,
    )

    payload = _VertexMaasChatResponse.from_stream_state(state)
    response = payload.to_llm_response()

    assert response.raw_text == "first second"
    assert response.choices[0].message.reasoning == "think-1think-2"
    assert response.choices[0].message.tool_calls[0].name == "lookup"
    assert response.choices[0].message.tool_calls[0].arguments == '{"q": "paris"}'
    assert response.usage.total_tokens == 8


def test_vertex_maas_openai_chat_model_name_strips_publisher_prefix() -> None:
    assert (
        vertex_maas_openai_chat_model_name("publishers/openai/models/gpt-oss-120b-maas")
        == "openai/gpt-oss-120b-maas"
    )
    assert vertex_maas_openai_chat_model_name("openai/gpt-oss-120b-maas") == "openai/gpt-oss-120b-maas"


def test_vertex_verify_response_still_rejects_reasoning_only_output() -> None:
    response = LlmResponse(
        id="reasoning-only",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(),
                    tool_calls=None,
                    reasoning="I reasoned but produced no final answer.",
                ),
                finish_reason="stop",
            ),
        ),
        usage=LlmUsage(reasoning_tokens=5),
        finish_reason="stop",
    )

    assert VertexLlmProvider._verify_response(response) == (False, True, "empty_output")


async def test_vertex_provider_routes_claude_models_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {"vertex_calls": 0, "anthropic_calls": 0}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=CHUTES.timeout_seconds,
    )

    async def fake_call_claude(request: Any) -> LlmResponse:
        captured["anthropic_calls"] += 1
        captured["anthropic_model"] = request.model
        return LlmResponse(
            id="claude-response",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(),
            finish_reason="stop",
        )

    monkeypatch.setattr(provider, "_call_claude_anthropic", fake_call_claude)

    request = LlmRequest(
        provider="vertex",
        model="/anthropic/models/claude-sonnet-4-5@20250929",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
    )

    response = await provider.invoke(request)
    assert response.raw_text == "ok"

    assert captured["anthropic_calls"] == 1
    assert "model_stream_call" not in captured


async def test_vertex_claude_stream_default_reconstructs_final_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=CHUTES.timeout_seconds,
    )

    class _FakeFinalAnthropicMessage:
        id = "claude-stream-response"

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {"id": self.id, "mode": mode}

    class _FakeStreamManager:
        def __init__(self) -> None:
            self._final_message = _FakeFinalAnthropicMessage()

        async def __aenter__(self) -> _FakeStreamManager:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        @property
        def text_stream(self) -> Any:
            async def _iter() -> Any:
                yield "ok"

            return _iter()

        async def get_final_message(self) -> _FakeFinalAnthropicMessage:
            return self._final_message

    captured_stream_kwargs: dict[str, Any] = {}

    def fake_stream(**kwargs: Any) -> _FakeStreamManager:
        captured_stream_kwargs.update(kwargs)
        return _FakeStreamManager()

    monkeypatch.setattr(provider._anthropic_client.messages, "stream", fake_stream)
    monkeypatch.setattr(
        "harnyx_commons.llm.providers.vertex.provider.build_anthropic_response",
        lambda response: LlmResponse(
            id=response.id,
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            metadata=None,
            finish_reason="stop",
        ),
    )

    request = LlmRequest(
        provider="vertex",
        model="/anthropic/models/claude-sonnet-4-5@20250929",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
    )

    response = await provider.invoke(request)

    assert captured_stream_kwargs["model"] == "claude-sonnet-4-5@20250929"
    assert response.raw_text == "ok"
    assert response.metadata is not None
    assert response.metadata["raw_response"] == {"id": "claude-stream-response", "mode": "json"}
    assert "ttft_ms" not in response.metadata

    records = [record for record in caplog.records if record.message == "llm.vertex.stream.ttft"]
    assert records
    data = records[0].__dict__["data"]
    assert data["branch"] == "claude"
    assert isinstance(data["ttft_ms"], float)
    assert data["ttft_ms"] >= 0.0


async def test_vertex_maas_payload_forces_stream_even_when_extra_overrides() -> None:
    request = LlmRequest(
        provider="vertex-maas",
        model="publishers/openai/models/gpt-oss-120b-maas",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
        extra={"stream": False},
    )

    payload = _VertexMaasChatRequest.from_request(request).model_dump(mode="python", exclude_none=True)

    assert payload["stream"] is True


async def test_vertex_provider_aclose_closes_owned_clients(
    monkeypatch: pytest.MonkeyPatch,
    anthropic_clients: list[Any],
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)
    _patch_vertex_maas_http_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    await provider.aclose()

    assert captured["google_async_closed"] is True
    assert captured["google_sync_closed"] is True
    assert captured["http_closed"] is True
    assert len(anthropic_clients) == 1
    assert anthropic_clients[0].closed is True


def test_vertex_provider_writes_base64_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}

    class FakeCredentials:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        @classmethod
        def from_service_account_info(
            cls,
            info: dict[str, Any],
            scopes: tuple[str, ...],
        ) -> FakeCredentials:
            captured["creds_info"] = info
            captured["creds_scopes"] = scopes
            return cls("info")

        @classmethod
        def from_service_account_file(
            cls,
            path: str,
            scopes: tuple[str, ...],
        ) -> FakeCredentials:
            captured["creds_path"] = path
            captured["creds_scopes"] = scopes
            return cls("file")

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.aio = self._AsyncClient()

        def close(self) -> None:
            captured["google_sync_closed"] = True

        class _AsyncClient:
            def __init__(self) -> None:
                self.models = self._Models()

            async def aclose(self) -> None:
                return None

            class _Models:
                async def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                    return FakeResponse()

    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.credentials.ServiceAccountCredentials", FakeCredentials)
    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

    service_account_payload = json.dumps({
        "type": "service_account",
        "client_email": "vertex@test-project.iam.gserviceaccount.com",
        "private_key_id": "abc123",
        "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    encoded = base64.b64encode(service_account_payload.encode()).decode()

    VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
        service_account_b64=encoded,
    )


async def test_vertex_provider_injects_google_search_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = GroundedLlmRequest(
        provider="vertex",
        model="gemini-2.0",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("summarize"),),
            ),
        ),
        temperature=None,
        max_output_tokens=None,
    )

    await provider.invoke(request)

    config = captured["model_stream_call"]["config"]
    response_mime_type = config.response_mime_type
    assert response_mime_type is None
    assert config.tools
    tool = config.tools[0]
    assert tool.google_search is not None
    assert config.thinking_config is None


async def test_vertex_provider_includes_provider_native_grounded_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = GroundedLlmRequest(
        provider="vertex",
        model="gemini-2.0",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("check novelty"),),
            ),
        ),
        temperature=None,
        max_output_tokens=None,
        tools=(
            LlmTool(
                type="provider_native",
                config={
                    "retrieval": {
                        "external_api": {
                            "api_spec": "ELASTIC_SEARCH",
                            "endpoint": "https://elastic.example.com",
                            "api_auth": {"api_key_config": {"api_key_string": "ApiKey test"}},
                            "elastic_search_params": {
                                "index": "feed-eval-alias",
                                "search_template": "feed_eval_hybrid_v1",
                                "num_hits": 20,
                            },
                        }
                    }
                },
            ),
        ),
    )

    await provider.invoke(request)

    config = captured["model_stream_call"]["config"]
    assert config.tools
    assert len(config.tools) == 2
    assert config.tools[0].google_search is not None
    retrieval = config.tools[1].retrieval
    assert retrieval is not None
    external_api = retrieval.external_api
    assert external_api is not None
    assert external_api.endpoint == "https://elastic.example.com"
    assert external_api.elastic_search_params is not None
    assert external_api.elastic_search_params.search_template == "feed_eval_hybrid_v1"


async def test_vertex_serializes_input_tool_result_as_function_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = LlmRequest(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-20b-maas",
        messages=(
            LlmMessage(
                role="user",
                content=(
                    LlmInputToolResultPart(
                        tool_call_id="call-1",
                        name="search_repo",
                        output_json=json.dumps({"data": [{"path": "README.md"}]}),
                    ),
                ),
            ),
        ),
        temperature=None,
        max_output_tokens=128,
        output_mode="text",
    )

    await provider.invoke(request)

    contents = captured["model_stream_call"]["contents"]
    assert contents
    part = contents[0].parts[0]
    function_response = part.function_response
    assert function_response is not None
    assert function_response.name == "search_repo"
    assert function_response.response["tool_call_id"] == "call-1"
    assert function_response.response["data"][0]["path"] == "README.md"


def test_vertex_verify_accepts_tool_call_only_choice() -> None:
    response = LlmResponse(
        id="resp-tool-call-only",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text=""),),
                    tool_calls=(
                        LlmMessageToolCall(
                            id="tc-1",
                            type="function",
                            name="search_repo",
                            arguments='{"query":"harnyx"}',
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ),
        usage=LlmUsage(),
        finish_reason="tool_calls",
    )

    ok, retryable, reason = VertexLlmProvider._verify_response(response)
    assert ok is True
    assert retryable is False
    assert reason is None
