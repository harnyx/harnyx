from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.llm.provider import LlmProviderConfigurationError, LlmProviderError
from harnyx_commons.llm.providers import ai_gateway
from harnyx_commons.llm.providers.ai_gateway import (
    AI_GATEWAY_BASE_URL,
    AI_GATEWAY_SUPPORTED_MODELS,
    AiGatewayLlmProvider,
)
from harnyx_commons.llm.schema import (
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmThinkingConfig,
    LlmTool,
)
from harnyx_commons.llm.tool_models import MINER_SELECTED_LLM_PROVIDER_MODELS

pytestmark = pytest.mark.anyio("asyncio")


def test_ai_gateway_supported_models_come_from_miner_selected_provider_contract() -> None:
    assert AI_GATEWAY_SUPPORTED_MODELS == MINER_SELECTED_LLM_PROVIDER_MODELS["ai_gateway"]


def test_ai_gateway_provider_rejects_blank_key() -> None:
    with pytest.raises(LlmProviderConfigurationError, match="AI_GATEWAY_API_KEY must be configured"):
        AiGatewayLlmProvider(ai_gateway_api_key=SecretStr(" "))


async def test_ai_gateway_provider_rejects_unsupported_model_before_http_request() -> None:
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return _streaming_response(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AiGatewayLlmProvider(ai_gateway_api_key=SecretStr("test-ai-gateway-key"), client=client)

    try:
        with pytest.raises(ValueError, match="AI Gateway provider does not support model"):
            await provider.invoke(_request(model="unsupported/model"))
    finally:
        await provider.aclose()
        await client.aclose()

    assert request_count == 0


@pytest.mark.parametrize(
    "extra",
    (
        {"provider": {"only": ["cerebras"]}},
        {"providerOptions": {"gateway": {"only": ["cerebras"]}}},
    ),
)
async def test_ai_gateway_provider_serializes_request_extra_and_provider_metadata(extra: dict[str, Any]) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return _streaming_response(request)

    client = httpx.AsyncClient(
        base_url=AI_GATEWAY_BASE_URL,
        headers={"Authorization": "Bearer test-ai-gateway-key"},
        transport=httpx.MockTransport(handler),
    )
    provider = AiGatewayLlmProvider(ai_gateway_api_key=SecretStr("test-ai-gateway-key"), client=client)

    try:
        response = await provider.invoke(_request(model="zai/glm-5.2-fast", extra=extra))
    finally:
        await provider.aclose()
        await client.aclose()

    assert captured["url"] == f"{AI_GATEWAY_BASE_URL}/chat/completions"
    assert captured["json"]["model"] == "zai/glm-5.2-fast"
    for key, value in extra.items():
        assert captured["json"][key] == value
    assert response.raw_text == "ok"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 5
    assert response.metadata is not None
    assert response.metadata["raw_response"]["providerMetadata"] == {"gateway": {"cost": "0.0042"}}
    assert response.metadata["actual_cost_provider"] == "ai_gateway"
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.0042)
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "provider_returned"


async def test_ai_gateway_provider_preserves_nested_reasoning_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = "\n\n".join(
            (
                'data: {"id":"resp-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
                'data: {"id":"resp-1","choices":[{"index":0,"delta":{},'
                '"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":3,"completion_tokens":6,'
                '"completion_tokens_details":{"reasoning_tokens":4},"total_tokens":9}}',
                "data: [DONE]",
                "",
            )
        )
        return httpx.Response(200, text=payload, request=request, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(
        base_url=AI_GATEWAY_BASE_URL,
        headers={"Authorization": "Bearer test-ai-gateway-key"},
        transport=httpx.MockTransport(handler),
    )
    provider = AiGatewayLlmProvider(ai_gateway_api_key=SecretStr("test-ai-gateway-key"), client=client)

    try:
        response = await provider.invoke(_request(model="openai/gpt-oss-20b"))
    finally:
        await provider.aclose()
        await client.aclose()

    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    assert response.usage.reasoning_tokens == 4
    assert response.usage.total_tokens == 9
    assert response.metadata is not None
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "static_pricing"


@pytest.mark.parametrize(
    ("thinking", "expected_reasoning"),
    (
        (LlmThinkingConfig(enabled=False), {"effort": "none"}),
        (LlmThinkingConfig(enabled=True), {"enabled": True}),
        (LlmThinkingConfig(enabled=True, effort="low"), {"enabled": True, "effort": "low"}),
        (LlmThinkingConfig(enabled=True, budget=256), {"enabled": True, "max_tokens": 256}),
    ),
)
async def test_ai_gateway_provider_serializes_typed_thinking_to_reasoning(
    thinking: LlmThinkingConfig,
    expected_reasoning: dict[str, Any],
) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return _streaming_response(request)

    client = httpx.AsyncClient(
        base_url=AI_GATEWAY_BASE_URL,
        headers={"Authorization": "Bearer test-ai-gateway-key"},
        transport=httpx.MockTransport(handler),
    )
    provider = AiGatewayLlmProvider(ai_gateway_api_key=SecretStr("test-ai-gateway-key"), client=client)

    try:
        await provider.invoke(_request(model="openai/gpt-oss-120b", thinking=thinking))
    finally:
        await provider.aclose()
        await client.aclose()

    assert captured["json"]["reasoning"] == expected_reasoning


async def test_ai_gateway_provider_merges_typed_thinking_into_internal_reasoning_extra() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return _streaming_response(request)

    client = httpx.AsyncClient(
        base_url=AI_GATEWAY_BASE_URL,
        headers={"Authorization": "Bearer test-ai-gateway-key"},
        transport=httpx.MockTransport(handler),
    )
    provider = AiGatewayLlmProvider(ai_gateway_api_key=SecretStr("test-ai-gateway-key"), client=client)

    try:
        await provider.invoke(
            _request(
                model="openai/gpt-oss-120b",
                thinking=LlmThinkingConfig(enabled=True, effort="high"),
                extra={"reasoning": {"exclude": True}},
            )
        )
    finally:
        await provider.aclose()
        await client.aclose()

    assert captured["json"]["reasoning"] == {"exclude": True, "enabled": True, "effort": "high"}


async def test_ai_gateway_provider_rejects_non_object_internal_reasoning_extra() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: _streaming_response(request)))
    provider = AiGatewayLlmProvider(ai_gateway_api_key=SecretStr("test-ai-gateway-key"), client=client)

    try:
        with pytest.raises(LlmProviderError, match="AI Gateway request extra.reasoning must be an object"):
            await provider.invoke(_request(model="openai/gpt-oss-120b", extra={"reasoning": "low"}))
    finally:
        await provider.aclose()
        await client.aclose()


def _request(
    *,
    model: str,
    extra: dict[str, Any] | None = None,
    thinking: LlmThinkingConfig | None = None,
) -> LlmRequest:
    return LlmRequest(
        provider="ai_gateway",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=32,
        extra=extra,
        thinking=thinking,
    )


def _streaming_response(request: httpx.Request) -> httpx.Response:
    payload = "\n\n".join(
        (
            'data: {"id":"resp-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"id":"resp-1","choices":[{"index":0,'
            '"delta":{"provider_metadata":{"gateway":{"cost":"0.0042"}}},'
            '"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, text=payload, request=request, headers={"content-type": "text/event-stream"})


def test_ai_gateway_request_serializes_complete_tool_loop() -> None:
    request = LlmRequest(
        provider="ai_gateway",
        model="openai/gpt-oss-20b",
        messages=(
            LlmMessage(
                role="assistant",
                content=(),
                tool_calls=(
                    LlmMessageToolCall(
                        id="call-1",
                        type="function",
                        name="lookup_weather",
                        arguments='{"city":"Paris"}',
                    ),
                ),
                reasoning_details=({"type": "reasoning.encrypted", "data": "opaque"},),
            ),
            LlmMessage(
                role="tool",
                content=(
                    LlmInputToolResultPart(
                        tool_call_id="call-1",
                        name=None,
                        output_json='{"temperature":19}',
                    ),
                ),
            ),
        ),
        temperature=0.0,
        max_output_tokens=32,
        tools=(LlmTool(type="function", function={"name": "lookup_weather", "strict": True}),),
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
        parallel_tool_calls=True,
    )

    payload = ai_gateway._AiGatewayChatRequest.from_request(request).model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )

    assert payload["messages"][0]["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "opaque"}
    ]
    assert payload["messages"][1] == {
        "role": "tool",
        "content": '{"temperature":19}',
        "tool_call_id": "call-1",
    }
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup_weather"}}
    assert payload["parallel_tool_calls"] is True
