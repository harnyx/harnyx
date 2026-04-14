from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from harnyx_commons.llm.providers.chutes import (
    ChutesLlmProvider,
    ChutesTextEmbeddingClient,
    _parse_chutes_response_payload,
    resolve_chutes_embedding_base_url,
)
from harnyx_commons.llm.providers.chutes_codec import _ChutesChatResponse, _ChutesReasoningStreamState
from harnyx_commons.llm.providers.openai_stream import (
    OpenAiStreamError,
    OpenAiStreamState,
    _OpenAiStreamEvent,
    iter_openai_sse_events,
)
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmResponse, LlmUsage


class _JudgeDecision(BaseModel):
    better: str


def test_parse_payload_skips_malformed_choice_and_keeps_valid_choice() -> None:
    payload = {
        "id": "resp_1",
        "choices": [
            {"message": {"content": "ok"}},
            None,
        ],
    }

    parsed = _parse_chutes_response_payload(payload)

    assert len(parsed.choices) == 1
    assert parsed.to_llm_response().choices[0].message.content[0].text == "ok"


def test_all_malformed_choices_fall_back_to_retryable_empty_choices_verifier() -> None:
    payload = {
        "id": "resp_2",
        "choices": [None],
    }

    parsed = _parse_chutes_response_payload(payload)
    ok, retryable, reason = ChutesLlmProvider._verify_response(
        LlmResponse(
            id="resp_2",
            choices=parsed.choices,
            usage=LlmUsage(),
        )
    )

    assert (ok, retryable, reason) == (False, True, "empty_choices")


def test_non_array_choices_fall_back_to_retryable_empty_choices_verifier() -> None:
    payload = {
        "id": "resp_3",
        "choices": {"unexpected": "object"},
    }

    parsed = _parse_chutes_response_payload(payload)
    ok, retryable, reason = ChutesLlmProvider._verify_response(
        LlmResponse(
            id="resp_3",
            choices=parsed.choices,
            usage=LlmUsage(),
        )
    )

    assert (ok, retryable, reason) == (False, True, "empty_choices")


def test_parse_payload_ignores_malformed_tool_call_and_keeps_valid_choice() -> None:
    payload = {
        "id": "resp_4",
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "tc-valid",
                            "type": "function",
                            "function": {
                                "name": "summarize",
                                "arguments": "{}",
                            },
                        },
                        {
                            "id": "tc-bad",
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "{}",
                            },
                        },
                    ],
                },
            },
        ],
    }

    parsed = _parse_chutes_response_payload(payload)

    assert len(parsed.choices) == 1
    assert parsed.to_llm_response().choices[0].message.content[0].text == "ok"
    tool_calls = parsed.choices[0].message.tool_calls or ()
    assert tuple(call.id for call in tool_calls) == ("tc-valid",)


def test_parse_payload_skips_malformed_content_fragment_and_keeps_valid_text() -> None:
    payload = {
        "id": "resp_5",
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "ok"},
                        None,
                    ]
                },
            },
        ],
    }

    parsed = _parse_chutes_response_payload(payload)

    assert len(parsed.choices) == 1
    parts = parsed.choices[0].message.content
    assert len(parts) == 1
    assert parts[0].text == "ok"


def test_parse_payload_normalizes_string_reasoning_field() -> None:
    payload = {
        "id": "resp_reasoning",
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "reasoning": "  model supplied unsupported reasoning shape  ",
                },
            },
        ],
    }

    parsed = _parse_chutes_response_payload(payload)

    assert parsed.choices[0].message.reasoning == "model supplied unsupported reasoning shape"


def test_resolve_chutes_embedding_base_url_returns_expected_live_base_url() -> None:
    base_url = resolve_chutes_embedding_base_url("Qwen/Qwen3-Embedding-0.6B")

    assert base_url == "https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai"


def test_resolve_chutes_embedding_base_url_fails_for_unmapped_model() -> None:
    with pytest.raises(RuntimeError, match="no chutes embedding base_url configured"):
        resolve_chutes_embedding_base_url("Unknown/Embedding-Model")


@pytest.mark.anyio("asyncio")
async def test_chutes_text_embedding_client_posts_openai_compatible_embeddings_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["json"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "embedding": [0.25, 0.5, 0.75],
                        "index": 0,
                        "object": "embedding",
                    }
                ]
            },
        )

    client = ChutesTextEmbeddingClient(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="https://example.com",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
        api_key="test-key",
        dimensions=3,
    )

    vector = await client.embed("hello world")

    assert vector == (0.25, 0.5, 0.75)
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/embeddings"
    assert '"model":"Qwen/Qwen3-Embedding-0.6B"' in str(captured["json"])
    assert '"input":"hello world"' in str(captured["json"])


def test_classify_http_status_includes_upstream_detail() -> None:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={"detail": "response_format.json_schema is invalid"},
    )
    exc = httpx.HTTPStatusError("bad request", request=request, response=response)

    retryable, reason = ChutesLlmProvider._classify_exception(exc)

    assert retryable is False
    assert reason == "http_400: response_format.json_schema is invalid"


def test_classify_http_status_handles_closed_stream_response() -> None:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(
        503,
        request=request,
        stream=httpx.ByteStream(b'{"detail":"temporarily unavailable"}'),
    )
    response.close()
    exc = httpx.HTTPStatusError("upstream failure", request=request, response=response)

    retryable, reason = ChutesLlmProvider._classify_exception(exc)

    assert retryable is True
    assert reason == "http_503"


@pytest.mark.parametrize("code", [500, 502, 503, 504, "500"])
def test_classify_stream_error_preserves_server_retry_policy(code: int | str) -> None:
    exc = OpenAiStreamError(
        message="temporarily unavailable",
        error_type="server_error",
        code=code,
    )

    retryable, reason = ChutesLlmProvider._classify_exception(exc)

    assert retryable is True
    assert reason == f"stream_error:{code}:server_error:temporarily unavailable"


@pytest.mark.anyio("asyncio")
async def test_iter_openai_sse_events_raises_upstream_in_band_error() -> None:
    response = httpx.Response(
        200,
        text='data: {"error":{"message":"temporarily unavailable","type":"server_error"}}\n\n',
    )

    with pytest.raises(OpenAiStreamError, match="temporarily unavailable"):
        async for _ in iter_openai_sse_events(
            response,
            invalid_data_message="invalid_data",
            invalid_event_message="invalid_event",
        ):
            pass


@pytest.mark.anyio("asyncio")
async def test_iter_openai_sse_events_rejects_wrapped_event_envelope() -> None:
    response = httpx.Response(
        200,
        text='data: {"event":{"choices":[{"message":{"content":"ok"}}]}}\n\n',
    )

    with pytest.raises(RuntimeError, match="invalid_data"):
        async for _ in iter_openai_sse_events(
            response,
            invalid_data_message="invalid_data",
            invalid_event_message="invalid_event",
        ):
            pass


def test_streamed_fallback_reasoning_preserves_exact_chunk_text() -> None:
    state = OpenAiStreamState()
    reasoning_state = _ChutesReasoningStreamState()

    first_event = _OpenAiStreamEvent.model_validate(
        {"choices": [{"index": 0, "message": {"content": "ok", "reasoning": "step"}}]}
    )
    second_event = _OpenAiStreamEvent.model_validate(
        {"choices": [{"index": 0, "message": {"reasoning": " "}}]}
    )
    third_event = _OpenAiStreamEvent.model_validate(
        {"choices": [{"index": 0, "message": {"reasoning": "two"}, "finish_reason": "stop"}]}
    )

    reasoning_state.merge_event(first_event)
    assert state.merge_event(first_event, reasoning_keys=()) is True
    reasoning_state.merge_event(second_event)
    assert state.merge_event(second_event, reasoning_keys=()) is False
    reasoning_state.merge_event(third_event)
    assert state.merge_event(third_event, reasoning_keys=()) is False

    response = _ChutesChatResponse.from_stream_state(state, reasoning_state=reasoning_state)

    assert response.to_llm_response().choices[0].message.reasoning == "step two"


def test_build_payload_accepts_json_object_output_mode() -> None:
    provider = ChutesLlmProvider(base_url="https://example.com", api_key="test-key")

    payload = provider._build_request(
        LlmRequest(
            provider="chutes",
            model="deepseek-ai/DeepSeek-V3.1",
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text("Return JSON"),),
                ),
            ),
            temperature=0.0,
            max_output_tokens=64,
            output_mode="json_object",
        )
    ).model_dump(mode="python", exclude_none=True)

    assert payload["response_format"] == {"type": "json_object"}


def test_build_payload_accepts_structured_output_mode() -> None:
    provider = ChutesLlmProvider(base_url="https://example.com", api_key="test-key")

    payload = provider._build_request(
        LlmRequest(
            provider="chutes",
            model="deepseek-ai/DeepSeek-V3.1",
            messages=(
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text("Choose the better answer"),),
                ),
            ),
            temperature=0.0,
            max_output_tokens=64,
            output_mode="structured",
            output_schema=_JudgeDecision,
        )
    ).model_dump(mode="python", exclude_none=True)

    response_format = payload["response_format"]
    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "_JudgeDecision",
            "schema": _JudgeDecision.model_json_schema(),
        },
    }
