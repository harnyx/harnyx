from __future__ import annotations

import json
import logging

import httpx
import pytest

from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


def _request() -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model="openai/gpt-oss-20b-TEE",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Explain retry telemetry briefly."),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
    )


def _sse_payload(*events: dict[str, object]) -> str:
    chunks = [f"data: {json.dumps(event)}\n\n" for event in events]
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


def _transport_backed_provider(payload: dict[str, object]) -> tuple[ChutesLlmProvider, httpx.AsyncClient]:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=_sse_payload(payload), request=request)

    client = httpx.AsyncClient(
        base_url="https://example.com",
        transport=httpx.MockTransport(handler),
    )
    provider = ChutesLlmProvider(
        base_url="https://example.com",
        api_key="test-key",
        client=client,
    )
    return provider, client


async def test_chutes_reasoning_payload_is_preserved_through_invoke() -> None:
    provider, client = _transport_backed_provider(
        {
            "id": "resp_reasoning_ok",
            "choices": [
                {
                    "message": {
                        "content": "final answer",
                        "reasoning": {
                            "thought_text_parts": ["step one", "step two"],
                            "has_thought_signature": True,
                        },
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await client.aclose()

    assert response.raw_text == "final answer"
    assert response.usage.total_tokens == 3
    assert response.choices[0].message.reasoning == "step one\n\nstep two"
    assert response.metadata is not None
    assert response.metadata["raw_response"]["choices"][0]["message"]["reasoning"] == {
        "thought_text_parts": ["step one", "step two"],
        "has_thought_signature": True,
    }
    assert "ttft_ms" not in response.metadata


async def test_chutes_reasoning_object_is_preserved_across_multiple_stream_events() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            text=_sse_payload(
                {
                    "id": "resp_reasoning_split",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning": {
                                    "thought_text_parts": ["step one"],
                                    "has_thought_signature": True,
                                },
                            },
                        }
                    ],
                },
                {
                    "id": "resp_reasoning_split",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning": {
                                    "thought_text_parts": ["step two"],
                                },
                                "content": "final answer",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                },
            ),
            request=request,
        )

    client = httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler))
    provider = ChutesLlmProvider(base_url="https://example.com", api_key="test-key", client=client)

    try:
        response = await provider.invoke(_request())
    finally:
        await client.aclose()

    assert response.raw_text == "final answer"
    assert response.choices[0].message.reasoning == "step one\n\nstep two"
    assert response.metadata["raw_response"]["choices"][0]["message"]["reasoning"] == {
        "thought_text_parts": ["step one", "step two"],
        "has_thought_signature": True,
    }


async def test_chutes_reasoning_string_is_normalized_through_invoke() -> None:
    provider, client = _transport_backed_provider(
        {
            "id": "resp_reasoning_bad",
            "choices": [
                {
                    "message": {
                        "content": "final answer",
                        "reasoning": "bad-shape",
                    }
                }
            ],
        }
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await client.aclose()

    assert response.choices[0].message.reasoning == "bad-shape"


async def test_chutes_stream_default_emits_provider_local_ttft_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.providers.chutes")
    provider, client = _transport_backed_provider(
        {
            "id": "resp_reasoning_ok",
            "choices": [
                {
                    "message": {
                        "content": "final answer",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }
    )

    try:
        await provider.invoke(_request())
    finally:
        await client.aclose()

    records = [record for record in caplog.records if record.message == "chutes.stream.ttft"]
    assert records
    data = records[0].__dict__["data"]
    assert data["provider"] == "chutes"
    assert data["model"] == "openai/gpt-oss-20b-TEE"
    assert isinstance(data["ttft_ms"], float)
    assert data["ttft_ms"] >= 0.0


async def test_chutes_payload_forces_stream_even_when_extra_overrides() -> None:
    provider = ChutesLlmProvider(
        base_url="https://example.invalid",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="https://example.invalid"),
    )
    request = LlmRequest(
        provider="chutes",
        model="deepseek-ai/DeepSeek-R1",
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

    try:
        payload = provider._build_request(request).model_dump(mode="python", exclude_none=True)
    finally:
        await provider.aclose()

    assert payload["stream"] is True
