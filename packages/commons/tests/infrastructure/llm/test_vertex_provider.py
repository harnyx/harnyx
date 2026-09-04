from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, cast

import httpx
import pytest
from anthropic import AsyncAnthropicVertex
from anthropic.types.raw_content_block_delta_event import RawContentBlockDeltaEvent
from anthropic.types.text_block import TextBlock
from anthropic.types.text_delta import TextDelta
from anthropic.types.thinking_block import ThinkingBlock
from google.genai import errors, types
from pydantic import BaseModel

from harnyx_commons.clients import CHUTES
from harnyx_commons.llm.provider import LlmRetryExhaustedError
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
from harnyx_commons.llm.providers.vertex.anthropic import build_anthropic_response
from harnyx_commons.llm.providers.vertex.codec import (
    _VertexMaasChatRequest,
    _VertexMaasChatResponse,
    normalize_messages,
    resolve_thinking_config,
    resolve_tool_config,
    vertex_maas_openai_chat_model_name,
)
from harnyx_commons.llm.providers.vertex.provider import (
    VertexLlmProvider,
    _vertex_stream_text_fragments,
    _VertexProviderProtocolError,
)
from harnyx_commons.llm.retry_utils import RetryPolicy
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
    LlmThinkingConfig,
    LlmTool,
    LlmUsage,
)
from harnyx_commons.llm.timeout import LlmAttemptTimeoutError
from harnyx_miner_sdk.llm import Timeout

pytestmark = pytest.mark.anyio("asyncio")


def _anthropic_sse(event_type: str, **payload: object) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps({'type': event_type, **payload})}\n\n".encode()


@pytest.mark.parametrize(
    ("output_kind", "outcome"),
    [(kind, outcome) for kind in ("json", "redacted") for outcome in ("complete", "total")]
    + [
        (kind, phase)
        for kind in ("empty-json", "empty-redacted", "signature", "ping")
        for phase in ("prefill", "inactivity")
    ],
)
async def test_claude_stream_output_controls_deadlines_through_anthropic_sdk(
    monkeypatch: pytest.MonkeyPatch, output_kind: str, outcome: str
) -> None:
    _patch_google_client(monkeypatch, {})
    loop = asyncio.get_running_loop()
    stream_time = loop.time()
    # Advance only at stream boundaries; SDK parsing and runner load consume no test time.
    monkeypatch.setattr(loop, "time", lambda: stream_time)

    class ClaudeStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal stream_time
            yield _anthropic_sse(
                "message_start",
                message={
                    "id": "msg-timeout",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5@20251001",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            )
            index = 0
            if outcome == "inactivity":
                yield _anthropic_sse("content_block_start", index=index, content_block={"type": "text", "text": ""})
                yield _anthropic_sse(
                    "content_block_delta", index=index, delta={"type": "text_delta", "text": "started"}
                )
                yield _anthropic_sse("content_block_stop", index=index)
                index += 1
            if output_kind in ("json", "empty-json"):
                yield _anthropic_sse(
                    "content_block_start",
                    index=index,
                    content_block={
                        "type": "server_tool_use",
                        "id": "srvtoolu_search",
                        "name": "web_search",
                        "input": {},
                    },
                )
                if output_kind == "json":
                    yield _anthropic_sse(
                        "content_block_delta",
                        index=index,
                        delta={"type": "input_json_delta", "partial_json": '{"query":"'},
                    )
            elif output_kind == "signature":
                yield _anthropic_sse(
                    "content_block_start",
                    index=index,
                    content_block={"type": "thinking", "thinking": "", "signature": ""},
                )

            for _ in range(12):
                if output_kind in ("json", "empty-json"):
                    yield _anthropic_sse(
                        "content_block_delta",
                        index=index,
                        delta={"type": "input_json_delta", "partial_json": "weather " if output_kind == "json" else ""},
                    )
                elif output_kind in ("redacted", "empty-redacted"):
                    yield _anthropic_sse(
                        "content_block_start",
                        index=index,
                        content_block={
                            "type": "redacted_thinking",
                            "data": "opaque" if output_kind == "redacted" else "",
                        },
                    )
                    yield _anthropic_sse("content_block_stop", index=index)
                    index += 1
                elif output_kind == "signature":
                    yield _anthropic_sse(
                        "content_block_delta",
                        index=index,
                        delta={"type": "signature_delta", "signature": "verification"},
                    )
                else:
                    yield _anthropic_sse("ping")
                stream_time += 0.02
                await asyncio.sleep(0)

            if outcome != "complete":
                raise AssertionError("The configured deadline should expire before the stream finishes")
            if output_kind == "json":
                yield _anthropic_sse(
                    "content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": '"}'}
                )
                yield _anthropic_sse("content_block_stop", index=index)
                index += 1
            yield _anthropic_sse("content_block_start", index=index, content_block={"type": "text", "text": ""})
            yield _anthropic_sse("content_block_delta", index=index, delta={"type": "text_delta", "text": "done"})
            yield _anthropic_sse("content_block_stop", index=index)
            yield _anthropic_sse(
                "message_delta",
                delta={"stop_reason": "end_turn", "stop_sequence": None},
                usage={"output_tokens": 12, "server_tool_use": {"web_search_requests": int(output_kind == "json")}},
            )
            yield _anthropic_sse("message_stop")

        async def aclose(self) -> None:
            self.closed = True

    body = ClaudeStream()

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    client = AsyncAnthropicVertex(
        project_id="demo-project",
        region="us-central1",
        access_token="test-token",  # noqa: S106
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle_request)),
        max_retries=0,
    )
    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.provider.AsyncAnthropicVertex", lambda **kwargs: client)
    provider = VertexLlmProvider(project="demo-project", location="us-central1")
    request = GroundedLlmRequest(
        provider="vertex",
        model="claude-haiku-4-5@20251001",
        messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Search the web."),)),),
        temperature=None,
        max_output_tokens=None,
        timeout=Timeout(0.2 if outcome == "total" else 1, prefill=0.1, inactivity=0.1),
        retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0),
    )
    try:
        if outcome == "complete":
            response = await provider.invoke(request)
            assert response.raw_text == "done"
            assert response.finish_reason == "end_turn"
            assert response.usage.completion_tokens == 12
            blocks = response.metadata["raw_response"]["content"]
            if output_kind == "redacted":
                assert [block["data"] for block in blocks if block["type"] == "redacted_thinking"] == ["opaque"] * 12
            else:
                assert blocks[0]["input"]["query"] == "weather " * 12
                assert response.usage.web_search_calls == 1
        else:
            with pytest.raises(LlmRetryExhaustedError) as exc:
                await provider.invoke(request)
            assert isinstance(exc.value.__cause__, LlmAttemptTimeoutError)
            assert exc.value.__cause__.phase == outcome
        assert body.closed
    finally:
        await provider.aclose()


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


@pytest.mark.parametrize(
    "code",
    [
        500,
        "500",
    ],
)
def test_vertex_classify_stream_error_preserves_server_retry_policy(code: int | str) -> None:
    exc = OpenAiStreamError(
        message="temporarily unavailable",
        error_type="server_error",
        code=code,
    )

    retryable, reason = VertexLlmProvider._classify_exception(exc)

    assert retryable is True
    assert reason == f"stream_error:{code}:server_error:temporarily unavailable"


def test_vertex_classify_google_api_error_retries_transient_codes() -> None:
    for code in (429, 500, 502, 503, 504, 529):
        exc = errors.APIError(
            code,
            {"error": {"code": code, "message": "temporary failure", "status": "TRANSIENT"}},
        )

        retryable, reason = VertexLlmProvider._classify_exception(exc)

        assert retryable is True
        assert reason.startswith(f"api_error:{code}:")


def test_vertex_classify_google_api_error_does_not_retry_client_errors() -> None:
    exc = errors.APIError(
        400,
        {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}},
    )

    retryable, reason = VertexLlmProvider._classify_exception(exc)

    assert retryable is False
    assert reason.startswith("api_error:400:")


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
            captured["model_stream_call_count"] = int(captured.get("model_stream_call_count", 0)) + 1

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


def _patch_vertex_maas_http_client_stream_sequence(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    stream_bodies: Sequence[str],
) -> dict[str, int]:
    state = {"next_index": 0, "stream_call_count": 0}
    http_calls: list[dict[str, Any]] = []
    captured["http_calls"] = http_calls

    class _FakeHttpResponse:
        def __init__(self, body: str) -> None:
            self._body = body
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in self._body.splitlines():
                yield line
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
            http_calls.append({"method": method, "url": url, **kwargs})
            captured["http_call"] = http_calls[-1]
            body = stream_bodies[state["next_index"]]
            state["next_index"] += 1
            state["stream_call_count"] += 1
            return _FakeStreamContext(_FakeHttpResponse(body))

        async def aclose(self) -> None:
            captured["http_closed"] = True

    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.provider.httpx.AsyncClient", _FakeAsyncHttpClient)
    return state


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
    )

    request = LlmRequest(
        provider="vertex",
        model="gemini-3-pro-preview",
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
    assert model_call["model"] == "gemini-3-pro-preview"
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
    assert isinstance(response.metadata["ttft_ms"], float)
    assert response.metadata["ttft_ms"] >= 0.0

    records = [record for record in caplog.records if record.message == "llm.vertex.stream.ttft"]
    assert records
    data = records[0].__dict__["data"]
    assert data["branch"] == "gemini"
    assert isinstance(data["ttft_ms"], float)
    assert data["ttft_ms"] >= 0.0


async def test_vertex_provider_retries_empty_gemini_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured, response_factory=lambda: [])

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
    )
    provider._retry_policy = RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0.0)

    request = LlmRequest(
        provider="vertex",
        model="gemini-3-pro-preview",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
        retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0),
    )

    with pytest.raises(
        LlmRetryExhaustedError,
        match="vertex streaming generation returned no response chunks",
    ):
        await provider.invoke(request)

    assert captured["model_stream_call_count"] == 2


def test_vertex_classify_empty_stream_protocol_error_retries() -> None:
    retryable, reason = VertexLlmProvider._classify_exception(
        _VertexProviderProtocolError("vertex streaming generation returned no response chunks")
    )

    assert retryable is True
    assert reason == "vertex streaming generation returned no response chunks"


async def test_vertex_provider_merges_request_http_headers_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)
    provider = VertexLlmProvider(
        project="demo-project",
        location="global",
        timeout=30.0,
    )
    headers = {
        "X-Vertex-AI-LLM-Request-Type": "shared",
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
    }
    request = LlmRequest(
        provider="vertex",
        model="gemini-3-pro-preview",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
        timeout=12.5,
        extra={"http_headers": headers},
    )

    await provider.invoke(request)

    config = captured["model_stream_call"]["config"]
    assert config.http_options.timeout == 12500
    assert config.http_options.headers == headers


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
            model="gemini-3-pro-preview",
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


def test_openai_choice_state_rejects_tool_call_without_function_name() -> None:
    state = OpenAiChoiceState(
        tool_calls={
            0: OpenAiToolCallState(
                id="tc-1",
                type="function",
                arguments_text='{"query":"harnyx"}',
            )
        }
    )

    with pytest.raises(OpenAiStreamError):
        state.tool_call_values()


def test_openai_tool_call_state_replaces_dict_argument_snapshots() -> None:
    state = OpenAiToolCallState(id="tc-1", type="function", name="lookup")

    assert state.merge_delta(
        _OpenAiToolCallDelta.model_validate({"index": 0, "function": {"arguments": {"query": "a"}}})
    )
    assert state.merge_delta(
        _OpenAiToolCallDelta.model_validate({"index": 0, "function": {"arguments": {"query": "ab"}}})
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
        provider="vertex",
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
    assert payload["stream_options"] == {
        "include_usage": True,
        "continuous_usage_stats": True,
    }
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 64
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert response.raw_text == "56"
    assert response.choices[0].message.reasoning == "I need to multiply 7 by 8."
    assert response.usage.reasoning_tokens == 5
    assert response.metadata is not None
    assert isinstance(response.metadata["ttft_ms"], float)
    assert response.metadata["ttft_ms"] >= 0.0
    assert response.metadata["actual_cost_provider"] == "vertex"
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.000001173)
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "static_pricing"

    records = [record for record in caplog.records if record.message == "llm.vertex.stream.ttft"]
    assert records
    data = records[0].__dict__["data"]
    assert data["branch"] == "vertex_maas_openai"
    assert isinstance(data["ttft_ms"], float)
    assert data["ttft_ms"] >= 0.0


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
        model="gemini-3-pro-preview",
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


def test_vertex_codec_rejects_assistant_replay_state_it_cannot_serialize() -> None:
    with pytest.raises(ValueError, match="do not support"):
        normalize_messages(
            (
                LlmMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        LlmMessageToolCall(
                            id="call-1",
                            type="function",
                            name="lookup",
                            arguments="{}",
                        ),
                    ),
                ),
            )
        )

    with pytest.raises(ValueError, match="do not support"):
        normalize_messages(
            (
                LlmMessage(
                    role="assistant",
                    content=(LlmMessageContentPart.input_text("answer"),),
                    reasoning_details=({"type": "reasoning.encrypted", "data": "opaque"},),
                ),
            )
        )


def test_vertex_tool_choice_none_disables_function_calling() -> None:
    config = resolve_tool_config("none", [types.Tool()])

    assert config is not None
    assert config.function_calling_config is not None
    assert config.function_calling_config.mode.name == "NONE"


def test_normalize_reasoning_effort_rejects_non_positive_budgets() -> None:
    assert normalize_reasoning_effort(None) is None
    assert normalize_reasoning_effort("  ") is None
    assert normalize_reasoning_effort("0") is None
    assert normalize_reasoning_effort("-1") is None
    assert normalize_reasoning_effort("high") == "high"
    assert normalize_reasoning_effort(" 256 ") == "256"


def test_vertex_resolve_thinking_config_rejects_numeric_budget() -> None:
    with pytest.raises(ValueError, match="numeric thinking budgets are not supported"):
        resolve_thinking_config(model="gemini-3-pro-preview", reasoning_effort="512")


class _StructuredPairwisePreference(BaseModel):
    preferred_position: str


def test_vertex_maas_chat_payload_supports_structured_output() -> None:
    request = LlmRequest(
        provider="vertex",
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


def _basic_vertex_maas_request(
    *,
    model: str = "deepseek-ai/deepseek-v3.2-maas",
    thinking: LlmThinkingConfig | None = None,
    reasoning_effort: str | None = None,
) -> LlmRequest:
    return LlmRequest(
        provider="vertex",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hi"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=32,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )


def test_vertex_maas_thinking_omitted_is_noop() -> None:
    payload = _VertexMaasChatRequest.from_request(_basic_vertex_maas_request()).model_dump(
        mode="python",
        exclude_none=True,
    )

    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload


def test_vertex_maas_deepseek_thinking_enabled_and_disabled_use_template_kwargs() -> None:
    enabled = _VertexMaasChatRequest.from_request(
        _basic_vertex_maas_request(thinking=LlmThinkingConfig(enabled=True))
    ).model_dump(mode="python", exclude_none=True)
    disabled = _VertexMaasChatRequest.from_request(
        _basic_vertex_maas_request(thinking=LlmThinkingConfig(enabled=False))
    ).model_dump(mode="python", exclude_none=True)

    assert enabled["chat_template_kwargs"] == {"thinking": True}
    assert disabled["chat_template_kwargs"] == {"thinking": False}
    assert "reasoning_effort" not in enabled
    assert "reasoning_effort" not in disabled


def test_vertex_maas_reasoning_effort_derives_template_thinking_and_suppresses_raw_effort() -> None:
    payload = _VertexMaasChatRequest.from_request(
        _basic_vertex_maas_request(
            model="zai-org/glm-5-maas",
            reasoning_effort="high",
        )
    ).model_dump(mode="python", exclude_none=True)

    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in payload


def test_vertex_maas_explicit_thinking_overrides_reasoning_effort() -> None:
    payload = _VertexMaasChatRequest.from_request(
        _basic_vertex_maas_request(
            model="zai-org/glm-5-maas",
            thinking=LlmThinkingConfig(enabled=False),
            reasoning_effort="high",
        )
    ).model_dump(mode="python", exclude_none=True)

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize("reasoning_effort", ("2048",))
def test_vertex_maas_template_model_suppresses_raw_numeric_and_blank_reasoning_effort(
    reasoning_effort: str,
) -> None:
    payload = _VertexMaasChatRequest.from_request(
        _basic_vertex_maas_request(
            model="zai-org/glm-5-maas",
            reasoning_effort=reasoning_effort,
        )
    ).model_dump(mode="python", exclude_none=True)

    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload


def test_vertex_maas_unsupported_thinking_capability_serializes_nothing() -> None:
    payload = _VertexMaasChatRequest.from_request(
        _basic_vertex_maas_request(
            model="publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas",
            thinking=LlmThinkingConfig(enabled=True, effort="high"),
        )
    ).model_dump(mode="python", exclude_none=True)

    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload


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
    assert response.usage.completion_tokens == 4
    assert response.usage.reasoning_tokens == 3
    assert response.usage.total_tokens == 24


def test_vertex_maas_deepseek_v31_splits_inline_reasoning_from_content() -> None:
    payload = {
        "id": "chatcmpl-v31",
        "model": "deepseek-ai/deepseek-v3.1-maas",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "private trace</think>final answer",
                    "reasoning_content": None,
                },
            }
        ],
    }

    response = _VertexMaasChatResponse.model_validate(payload).to_llm_response()

    assert response.raw_text == "final answer"
    assert response.choices[0].message.reasoning == "private trace"


def test_vertex_maas_deepseek_v31_alias_splits_inline_reasoning_from_content() -> None:
    payload = {
        "id": "chatcmpl-v31",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "<think>private trace</think>final answer",
                    "reasoning_content": None,
                },
            }
        ],
    }

    response = _VertexMaasChatResponse.model_validate(payload).to_llm_response(model="deepseek-ai/DeepSeek-V3.1-TEE")

    assert response.raw_text == "final answer"
    assert response.choices[0].message.reasoning == "private trace"


def test_vertex_maas_deepseek_v31_strips_truncated_inline_reasoning() -> None:
    payload = {
        "id": "chatcmpl-v31-truncated",
        "model": "deepseek-ai/deepseek-v3.1-maas",
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "<think>private trace",
                    "reasoning_content": None,
                },
            }
        ],
    }

    response = _VertexMaasChatResponse.model_validate(payload).to_llm_response()

    assert response.raw_text is None
    assert response.choices[0].message.reasoning == "private trace"
    assert VertexLlmProvider._verify_response(response) == (False, True, "empty_output")


def test_vertex_maas_deepseek_v31_keeps_delimiterless_length_content_as_answer() -> None:
    payload = {
        "id": "chatcmpl-v31-delimiterless-truncated",
        "model": "deepseek-ai/deepseek-v3.1-maas",
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "visible truncated answer",
                    "reasoning_content": None,
                },
            }
        ],
    }

    response = _VertexMaasChatResponse.model_validate(payload).to_llm_response()

    assert response.raw_text == "visible truncated answer"
    assert response.choices[0].message.reasoning is None
    assert VertexLlmProvider._verify_response(response) == (True, False, None)


def test_vertex_maas_deepseek_v32_length_content_is_not_inline_reasoning() -> None:
    payload = {
        "id": "chatcmpl-v32-truncated",
        "model": "deepseek-ai/deepseek-v3.2-maas",
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "visible truncated answer",
                    "reasoning_content": None,
                },
            }
        ],
    }

    response = _VertexMaasChatResponse.model_validate(payload).to_llm_response()

    assert response.raw_text == "visible truncated answer"
    assert response.choices[0].message.reasoning is None


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
                                "type": "function",
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
        vertex_maas_openai_chat_model_name("publishers/openai/models/gpt-oss-120b-maas") == "openai/gpt-oss-120b-maas"
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

        async def __aiter__(self) -> AsyncIterator[RawContentBlockDeltaEvent]:
            yield RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="ok")
            )

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
    assert captured_stream_kwargs["timeout"] == pytest.approx(300.0)
    assert response.raw_text == "ok"
    assert response.metadata is not None
    assert response.metadata["raw_response"] == {"id": "claude-stream-response", "mode": "json"}
    assert isinstance(response.metadata["ttft_ms"], float)
    assert response.metadata["ttft_ms"] >= 0.0

    records = [record for record in caplog.records if record.message == "llm.vertex.stream.ttft"]
    assert records
    data = records[0].__dict__["data"]
    assert data["branch"] == "claude"
    assert isinstance(data["ttft_ms"], float)
    assert data["ttft_ms"] >= 0.0


async def test_vertex_claude_thinking_forces_temperature_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=CHUTES.timeout_seconds,
    )

    class _FakeFinalAnthropicMessage:
        id = "claude-thinking-response"

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            return {"id": self.id, "mode": mode}

    class _FakeStreamManager:
        async def __aenter__(self) -> _FakeStreamManager:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def __aiter__(self) -> AsyncIterator[RawContentBlockDeltaEvent]:
            yield RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="ok")
            )

        async def get_final_message(self) -> _FakeFinalAnthropicMessage:
            return _FakeFinalAnthropicMessage()

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

    await provider.invoke(
        LlmRequest(
            provider="vertex",
            model="/anthropic/models/claude-sonnet-4-5@20250929",
            messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("hello"),)),),
            temperature=0.2,
            max_output_tokens=2048,
            reasoning_effort="1024",
        )
    )

    assert captured_stream_kwargs["temperature"] == pytest.approx(1.0)
    assert captured_stream_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_vertex_claude_response_maps_thinking_text_without_reasoning_tokens() -> None:
    class _Usage:
        input_tokens = 5
        output_tokens = 3
        server_tool_use = None

    class _Message:
        id = "claude-message"
        content = (
            ThinkingBlock(signature="sig-1", thinking="deliberation", type="thinking"),
            TextBlock(text="ok", type="text"),
        )
        stop_reason = "end_turn"
        usage = _Usage()

    response = build_anthropic_response(_Message())

    assert response.raw_text == "ok"
    assert response.choices[0].message.reasoning == "deliberation"
    assert response.usage.reasoning_tokens is None


async def test_vertex_maas_payload_forces_stream_even_when_extra_overrides() -> None:
    request = LlmRequest(
        provider="vertex",
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
    assert payload["stream_options"] == {
        "include_usage": True,
        "continuous_usage_stats": True,
    }


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
        model="gemini-3-pro-preview",
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
        model="gemini-3-pro-preview",
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
        model="gemini-3-pro-preview",
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


@pytest.mark.parametrize("failure", ["overdue_output", "cancelled"])
async def test_vertex_closes_gemini_stream_before_failed_attempt_returns(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from harnyx_commons.llm import timeout as deadline_module
    from harnyx_commons.llm.timeout import LlmAttemptTimeoutError, enforce_attempt_deadlines
    from harnyx_miner_sdk.llm import Timeout

    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)
    provider = VertexLlmProvider(project="test-project", location="us-central1")
    closed = False
    started = asyncio.Event()

    async def chunks() -> AsyncIterator[FakeResponse]:
        nonlocal closed
        try:
            started.set()
            if failure == "cancelled":
                await asyncio.Event().wait()
            else:
                attempt = deadline_module._current_attempt.get()
                assert attempt is not None
                monkeypatch.setattr(attempt, "now", lambda: attempt.started_at + 6)
            yield FakeResponse()
        finally:
            closed = True

    async def generate(**kwargs: Any) -> AsyncIterator[FakeResponse]:
        return chunks()

    monkeypatch.setattr(provider._genai_async_client.models, "generate_content_stream", generate)
    request = LlmRequest(
        provider="vertex", model="gemini-2.5-flash", messages=(), temperature=None, max_output_tokens=None
    )

    async def invoke() -> None:
        async with enforce_attempt_deadlines(Timeout(10, prefill=5)):
            await provider._call_vertex(request, [], None)

    try:
        if failure == "cancelled":
            task = asyncio.create_task(invoke())
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(LlmAttemptTimeoutError):
                await invoke()
        assert closed
    finally:
        await provider.aclose()
