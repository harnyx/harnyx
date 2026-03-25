from __future__ import annotations

import httpx
import pytest

from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


def _request() -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model="openai/gpt-oss-20b",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Explain retry telemetry briefly."),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
    )


def _transport_backed_provider(payload: dict[str, object]) -> tuple[ChutesLlmProvider, httpx.AsyncClient]:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=payload, request=request)

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
