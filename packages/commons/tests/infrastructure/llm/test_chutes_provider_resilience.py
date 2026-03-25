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
    assert parsed.choices[0].message.content[0].text == "ok"


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
    assert parsed.choices[0].message.content[0].text == "ok"
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
                    "reasoning": "model supplied unsupported reasoning shape",
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


def test_build_payload_accepts_json_object_output_mode() -> None:
    provider = ChutesLlmProvider(base_url="https://example.com", api_key="test-key")

    payload = provider._build_payload(
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
    )

    assert payload["response_format"] == {"type": "json_object"}


def test_build_payload_accepts_structured_output_mode() -> None:
    provider = ChutesLlmProvider(base_url="https://example.com", api_key="test-key")

    payload = provider._build_payload(
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
    )

    response_format = payload["response_format"]
    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "_JudgeDecision",
            "schema": _JudgeDecision.model_json_schema(),
        },
    }
