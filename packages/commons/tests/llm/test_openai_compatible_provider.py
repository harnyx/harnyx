from __future__ import annotations

import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from harnyx_commons.config.llm import LlmSettings, parse_openai_compatible_endpoints_json
from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.providers import openai_compatible
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmThinkingConfig,
    LlmTool,
)

pytestmark = pytest.mark.anyio("asyncio")


def test_endpoint_config_rejects_duplicate_ids() -> None:
    raw = json.dumps(
        [
            {"id": "duplicate", "base_url": "https://example.com/v1", "auth": {"type": "none"}},
            {"id": "duplicate", "base_url": "https://example.org/v1", "auth": {"type": "none"}},
        ]
    )

    with pytest.raises(ValueError, match="duplicated"):
        parse_openai_compatible_endpoints_json(raw)


def test_endpoint_config_rejects_raw_bearer_token_field() -> None:
    raw = json.dumps(
        [
            {
                "id": "local",
                "base_url": "https://example.com/v1",
                "auth": {"type": "bearer_token_env", "token_env": "LOCAL_TOKEN", "token": "secret"},
            }
        ]
    )

    with pytest.raises(ValueError, match="extra_forbidden"):
        parse_openai_compatible_endpoints_json(raw)


def test_openai_compatible_provider_defaults_client_timeout_to_300(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(openai_compatible.httpx, "AsyncClient", _FakeAsyncClient)

    OpenAiCompatibleLlmProvider(endpoint=_endpoint(auth={"type": "none"}))

    assert captured["timeout"] == pytest.approx(300.0)


async def test_bearer_token_env_auth_adds_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPENAI_COMPATIBLE_TOKEN", "test-token")
    endpoint = _endpoint(
        auth={"type": "bearer_token_env", "token_env": "LOCAL_OPENAI_COMPATIBLE_TOKEN"},
    )
    seen_headers: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers["authorization"]
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=endpoint,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert seen_headers["authorization"] == "Bearer test-token"
    assert response.raw_text == "ok"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2


async def test_google_id_token_service_account_b64_auth_refreshes_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_account_info = {
        "type": "service_account",
        "client_email": "test@example.iam.gserviceaccount.com",
        "private_key": "unused",
    }
    encoded = base64.b64encode(json.dumps(service_account_info).encode("utf-8")).decode("ascii")
    monkeypatch.setenv("GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64", encoded)
    captured: dict[str, object] = {"credential_count": 0, "refresh_count": 0}

    class _FakeCredentials:
        token: str | None = None

        def refresh(self, request: object) -> None:
            captured["refresh_request_type"] = type(request).__name__
            captured["refresh_count"] = int(captured["refresh_count"]) + 1
            self.token = f"google-id-token-{captured['refresh_count']}"

    def fake_from_service_account_info(info: dict[str, object], *, target_audience: str) -> _FakeCredentials:
        captured["credential_count"] = int(captured["credential_count"]) + 1
        captured["info"] = info
        captured["target_audience"] = target_audience
        return _FakeCredentials()

    monkeypatch.setattr(
        openai_compatible.service_account,
        "IDTokenCredentials",
        SimpleNamespace(from_service_account_info=fake_from_service_account_info),
    )
    endpoint = _endpoint(
        auth={
            "type": "google_id_token",
            "audience": "https://gemma.example.run.app",
            "credential_source": "service_account_json_b64_env",
            "credential_env": "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
        }
    )
    seen_headers: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers["authorization"])
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=endpoint,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        response = await provider.invoke(_request())
        second_response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert seen_headers == ["Bearer google-id-token-1", "Bearer google-id-token-2"]
    assert captured["credential_count"] == 2
    assert captured["refresh_count"] == 2
    assert captured["info"] == service_account_info
    assert captured["target_audience"] == "https://gemma.example.run.app"
    assert response.raw_text == "ok"
    assert second_response.raw_text == "ok"


async def test_openai_compatible_provider_normalizes_streamed_chat_response() -> None:
    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _streaming_response())),
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert response.id == "chatcmpl-1"
    assert response.raw_text == "ok"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 5
    assert not hasattr(response.usage, "cost")
    assert response.metadata is not None
    assert response.metadata["raw_response"]["id"] == "chatcmpl-1"
    assert response.metadata["raw_response"]["usage"]["cost"] == pytest.approx(0.00123)
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.00000115)
    assert response.metadata["actual_cost_provider"] == "custom-openai-compatible:local"
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "static_pricing"
    assert isinstance(response.metadata["ttft_ms"], float)
    assert response.metadata["ttft_ms"] >= 0.0


async def test_openai_compatible_provider_attaches_openrouter_usage_cost() -> None:
    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(endpoint_id="openrouter", auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _streaming_response())),
    )

    try:
        response = await provider.invoke(_request(provider="openrouter", model="deepseek/deepseek-v3.2"))
    finally:
        await provider.aclose()

    assert response.metadata is not None
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.00123)
    assert response.metadata["actual_cost_provider"] == "openrouter"
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "provider_returned"


async def test_openai_compatible_provider_attaches_openrouter_static_cost_when_usage_cost_missing() -> None:
    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(endpoint_id="openrouter", auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _streaming_response_without_usage_cost())),
    )

    try:
        response = await provider.invoke(_request(provider="openrouter", model="deepseek/deepseek-v3.2"))
    finally:
        await provider.aclose()

    assert response.metadata is not None
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.00000168)
    assert response.metadata["actual_cost_provider"] == "openrouter"
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "static_pricing"


async def test_openai_compatible_thinking_omitted_is_noop() -> None:
    seen_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert "chat_template_kwargs" not in seen_payload


@pytest.mark.parametrize("enabled", (True, False))
async def test_openai_compatible_gemma_thinking_uses_enable_thinking_template_kwarg(enabled: bool) -> None:
    seen_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await provider.invoke(_request(thinking=LlmThinkingConfig(enabled=enabled, effort="high")))
    finally:
        await provider.aclose()

    assert seen_payload["chat_template_kwargs"] == {"enable_thinking": enabled}
    assert "reasoning_effort" not in seen_payload


async def test_openai_compatible_reasoning_effort_derives_template_thinking() -> None:
    seen_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await provider.invoke(_request(reasoning_effort="high"))
    finally:
        await provider.aclose()

    assert seen_payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in seen_payload


async def test_openai_compatible_explicit_thinking_overrides_reasoning_effort() -> None:
    seen_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await provider.invoke(_request(thinking=LlmThinkingConfig(enabled=False), reasoning_effort="high"))
    finally:
        await provider.aclose()

    assert seen_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in seen_payload


@pytest.mark.parametrize("enabled", (True, False))
async def test_openai_compatible_gemma_thinking_survives_custom_route_model_alias(enabled: bool) -> None:
    route_target = "custom-openai-compatible:gemma4-cloud-run-turbo"
    seen_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return _streaming_response()

    provider = LlmProviderAdapter(
        provider_name=route_target,
        delegate=OpenAiCompatibleLlmProvider(
            endpoint=_endpoint(endpoint_id="gemma4-cloud-run-turbo", auth={"type": "none"}),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )

    try:
        await provider.invoke(
            _request(
                provider=route_target,
                thinking=LlmThinkingConfig(enabled=enabled),
            )
        )
    finally:
        await provider.aclose()

    assert seen_payload["model"] == "nvidia/Gemma-4-31B-IT-NVFP4"
    assert seen_payload["chat_template_kwargs"] == {"enable_thinking": enabled}


@pytest.mark.parametrize("enabled", (True, False))
async def test_openai_compatible_qwen36_thinking_survives_custom_route_model_alias(enabled: bool) -> None:
    route_target = "custom-openai-compatible:qwen36-cloud-run"
    seen_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return _streaming_response()

    provider = LlmProviderAdapter(
        provider_name=route_target,
        delegate=OpenAiCompatibleLlmProvider(
            endpoint=_endpoint(endpoint_id="qwen36-cloud-run", auth={"type": "none"}),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )

    try:
        await provider.invoke(
            _request(
                provider=route_target,
                model="Qwen/Qwen3.6-27B-TEE",
                thinking=LlmThinkingConfig(enabled=enabled),
            )
        )
    finally:
        await provider.aclose()

    assert seen_payload["model"] == "Qwen/Qwen3.6-27B-FP8"
    assert seen_payload["chat_template_kwargs"] == {"enable_thinking": enabled}


async def test_openai_compatible_qwen36_reasoning_effort_survives_custom_route_model_alias() -> None:
    route_target = "custom-openai-compatible:qwen36-cloud-run"
    seen_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return _streaming_response()

    provider = LlmProviderAdapter(
        provider_name=route_target,
        delegate=OpenAiCompatibleLlmProvider(
            endpoint=_endpoint(endpoint_id="qwen36-cloud-run", auth={"type": "none"}),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )

    try:
        await provider.invoke(
            _request(
                provider=route_target,
                model="Qwen/Qwen3.6-27B-TEE",
                reasoning_effort="high",
            )
        )
    finally:
        await provider.aclose()

    assert seen_payload["model"] == "Qwen/Qwen3.6-27B-FP8"
    assert seen_payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in seen_payload


async def test_openai_compatible_unsupported_thinking_capability_serializes_nothing() -> None:
    seen_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await provider.invoke(
            _request(
                model="unsupported/model",
                thinking=LlmThinkingConfig(enabled=True, budget=1024),
            )
        )
    finally:
        await provider.aclose()

    assert "chat_template_kwargs" not in seen_payload
    assert "reasoning_effort" not in seen_payload


async def test_openai_compatible_provider_preserves_streamed_reasoning_and_usage() -> None:
    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _streaming_reasoning_response())),
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert response.raw_text == "ok"
    assert response.choices[0].message.reasoning == "think trace"
    assert response.usage.completion_tokens == 2
    assert response.usage.reasoning_tokens == 4
    assert response.metadata is not None
    assert response.metadata["raw_response"]["choices"][0]["reasoning"] == "think trace"
    assert response.metadata["raw_response"]["usage"]["completion_tokens"] == 6


async def test_openai_compatible_provider_keeps_reasoning_tokens_unavailable_when_usage_omits_them() -> None:
    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: _streaming_reasoning_without_usage_response())
        ),
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert response.raw_text == "ok"
    assert response.choices[0].message.reasoning == "think trace"
    assert response.usage.completion_tokens == 6
    assert response.usage.reasoning_tokens is None


async def test_openai_compatible_provider_normalizes_nested_reasoning_usage() -> None:
    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _streaming_reasoning_details_response())),
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert response.raw_text == "ok"
    assert response.usage.completion_tokens == 2
    assert response.usage.reasoning_tokens == 4
    assert response.usage.total_tokens == 9


async def test_openai_compatible_provider_translates_streamed_reasoning_details() -> None:
    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _streaming_reasoning_details_text_response())),
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert response.raw_text == "ok"
    assert response.choices[0].message.reasoning == "first thoughtsummary thought"
    assert response.metadata is not None
    assert response.metadata["raw_response"]["choices"][0]["reasoning"] == "first thoughtsummary thought"


async def test_openai_compatible_provider_uses_request_retry_policy_over_default() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "capacity"})
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider._retry_policy = RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0.0)
    request = replace(_request(), retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0))

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert calls == 2
    assert response.raw_text == "ok"


async def test_openai_compatible_provider_retries_malformed_tool_call_response() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _streaming_response_with_malformed_tool_call()
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    request = replace(_request(), retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0))

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert calls == 2
    assert response.raw_text == "ok"
    assert response.metadata is not None
    assert response.metadata["attempts"] == 2


def _endpoint(
    *,
    auth: dict[str, object],
    endpoint_id: str = "local",
):
    return LlmSettings(
        LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON=json.dumps(
            [{"id": endpoint_id, "base_url": "https://example.com/v1", "auth": auth}]
        )
    ).openai_compatible_endpoints[endpoint_id]


def _request(
    *,
    provider: str = "custom-openai-compatible:local",
    model: str = "google/gemma-4-31B-turbo-TEE",
    thinking: LlmThinkingConfig | None = None,
    reasoning_effort: str | None = None,
) -> LlmRequest:
    return LlmRequest(
        provider=provider,
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )


def _streaming_response() -> httpx.Response:
    payload = "\n\n".join(
        (
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"cost":0.00123}}',
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, content=payload.encode("utf-8"))


def _streaming_response_with_malformed_tool_call() -> httpx.Response:
    tool_call_delta = {
        "id": "chatcmpl-malformed",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup_weather", "arguments": '{"city":'},
                        }
                    ]
                },
            }
        ],
    }
    completion_delta = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    payload = "\n\n".join(
        (
            f"data: {json.dumps(tool_call_delta)}",
            f"data: {json.dumps(completion_delta)}",
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, content=payload.encode("utf-8"))


def _streaming_response_without_usage_cost() -> httpx.Response:
    payload = "\n\n".join(
        (
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, content=payload.encode("utf-8"))


def test_openai_compatible_request_serializes_complete_tool_loop() -> None:
    request = LlmRequest(
        provider="openrouter",
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
        tools=(
            LlmTool(
                type="function",
                function={"name": "lookup_weather", "strict": True},
            ),
        ),
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
        parallel_tool_calls=True,
    )

    payload = openai_compatible._OpenAiCompatibleChatRequest.from_request(
        request,
        provider_name="openrouter",
    ).model_dump(mode="json", exclude_none=True)

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
        },
        {"role": "tool", "content": '{"temperature":19}', "tool_call_id": "call-1"},
    ]
    assert payload["tools"] == [
        {"type": "function", "function": {"name": "lookup_weather", "strict": True}}
    ]
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup_weather"}}
    assert payload["parallel_tool_calls"] is True


def test_openai_compatible_request_preserves_assistant_reasoning_without_tool_calls() -> None:
    request = LlmRequest(
        provider="openrouter",
        model="openai/gpt-oss-20b",
        messages=(
            LlmMessage(
                role="assistant",
                content=(LlmMessageContentPart.input_text("Prior answer."),),
                reasoning_details=({"type": "reasoning.encrypted", "data": "opaque"},),
            ),
        ),
        temperature=0.0,
        max_output_tokens=32,
    )

    payload = openai_compatible._OpenAiCompatibleChatRequest.from_request(
        request,
        provider_name="openrouter",
    ).model_dump(mode="json", exclude_none=True)

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": "Prior answer.",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
        }
    ]


def test_openai_stream_preserves_opaque_reasoning_details_in_order() -> None:
    from harnyx_commons.llm.providers.openai_stream import OpenAiStreamState, _OpenAiStreamEvent

    details = [
        {"type": "reasoning.encrypted", "data": "opaque", "index": 0},
        {"type": "reasoning.text", "text": "thinking", "signature": "sig"},
    ]
    state = OpenAiStreamState()
    state.merge_event(
        _OpenAiStreamEvent.model_validate(
            {"choices": [{"index": 0, "delta": {"reasoning_details": details}}]}
        ),
        reasoning_keys=("reasoning_details",),
    )

    assert state.choice(0).reasoning_details == details


@pytest.mark.parametrize(
    "tool_call",
    (
        {"type": "function", "name": "lookup", "arguments": "{}"},
        {"id": "call-1", "name": "lookup", "arguments": "{}"},
        {"id": "call-1", "type": "function", "name": "", "arguments": "{}"},
        {"id": " ", "type": "function", "name": "lookup", "arguments": "{}"},
        {"id": "call-1", "type": "function", "name": " ", "arguments": "{}"},
        {"id": "call-1", "type": "function", "name": "lookup", "arguments": "[]"},
        {"id": "call-1", "type": "function", "name": "lookup", "arguments": '{"x":NaN}'},
        {"id": "call-1", "type": "function", "name": "lookup", "arguments": '{"x":Infinity}'},
    ),
)
def test_openai_stream_rejects_malformed_completed_tool_calls(tool_call: dict[str, object]) -> None:
    from harnyx_commons.llm.providers.openai_stream import OpenAiToolCallState

    state = OpenAiToolCallState(
        id=tool_call.get("id"),
        type=tool_call.get("type"),
        name=tool_call.get("name"),
        arguments_text=str(tool_call["arguments"]),
    )

    with pytest.raises(ValueError):
        state.to_tool_call(index=0)


def test_openai_stream_rejects_tool_call_delta_without_index() -> None:
    from harnyx_commons.llm.providers.openai_stream import OpenAiStreamState, _OpenAiStreamEvent

    state = OpenAiStreamState()
    event = _OpenAiStreamEvent.model_validate(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="require an index"):
        state.merge_event(event, reasoning_keys=())


def test_openai_stream_accumulates_fragmented_tool_call_identity() -> None:
    from harnyx_commons.llm.providers.openai_stream import OpenAiStreamState, _OpenAiStreamEvent

    state = OpenAiStreamState()
    for payload in (
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_",
                                "type": "function",
                                "function": {"name": "look", "arguments": '{"city":"'},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "123",
                                "function": {"name": "up", "arguments": 'Paris"}'},
                            }
                        ]
                    },
                }
            ]
        },
    ):
        state.merge_event(_OpenAiStreamEvent.model_validate(payload), reasoning_keys=())

    tool_calls = state.choice(0).tool_call_values()
    assert tool_calls is not None
    assert tool_calls[0].id == "call_123"
    assert tool_calls[0].name == "lookup"
    assert tool_calls[0].arguments == '{"city":"Paris"}'


def test_openai_stream_accepts_complete_message_tool_calls_without_indices() -> None:
    from harnyx_commons.llm.providers.openai_stream import OpenAiStreamState, _OpenAiStreamEvent

    state = OpenAiStreamState()
    state.merge_event(
        _OpenAiStreamEvent.model_validate(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                ]
            }
        ),
        reasoning_keys=(),
    )

    tool_calls = state.choice(0).tool_call_values()
    assert tool_calls is not None
    assert tool_calls[0].id == "call-1"
    assert tool_calls[0].name == "lookup"


def test_openai_stream_rejects_duplicate_tool_call_ids_within_one_block() -> None:
    from harnyx_commons.llm.providers.openai_stream import (
        OpenAiChoiceState,
        OpenAiStreamError,
        OpenAiToolCallState,
    )

    state = OpenAiChoiceState(
        tool_calls={
            0: OpenAiToolCallState(
                id="call-1",
                type="function",
                name="lookup_a",
                arguments_text="{}",
            ),
            1: OpenAiToolCallState(
                id="call-1",
                type="function",
                name="lookup_b",
                arguments_text="{}",
            ),
        }
    )

    with pytest.raises(OpenAiStreamError, match="unique"):
        state.tool_call_values()


@pytest.mark.parametrize(
    "changed_delta",
    (
        {"index": 0, "id": "call-2"},
        {"index": 0, "type": "custom"},
        {"index": 0, "function": {"name": "other"}},
    ),
)
def test_openai_stream_rejects_tool_call_identity_changes(
    changed_delta: dict[str, object],
) -> None:
    from harnyx_commons.llm.providers.openai_stream import OpenAiToolCallState, _OpenAiToolCallDelta

    state = OpenAiToolCallState(id="call-1", type="function", name="lookup")

    with pytest.raises(ValueError, match="changed"):
        state.merge_delta(
            _OpenAiToolCallDelta.model_validate(changed_delta),
            complete_snapshot=True,
        )


def _streaming_reasoning_response() -> httpx.Response:
    payload = "\n\n".join(
        (
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning":"think "}}]}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning_content":"trace"}}]}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":6,"reasoning_tokens":4,"total_tokens":9}}',
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, content=payload.encode("utf-8"))


def _streaming_reasoning_without_usage_response() -> httpx.Response:
    payload = "\n\n".join(
        (
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning":"think "}}]}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning_content":"trace"}}]}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":6,"total_tokens":9}}',
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, content=payload.encode("utf-8"))


def _streaming_reasoning_details_response() -> httpx.Response:
    payload = "\n\n".join(
        (
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":6,'
            '"completion_tokens_details":{"reasoning_tokens":4},"total_tokens":9}}',
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, content=payload.encode("utf-8"))


def _streaming_reasoning_details_text_response() -> httpx.Response:
    payload = "\n\n".join(
        (
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning_details":['
            '{"type":"reasoning.text","text":"first thought"},'
            '{"type":"reasoning.summary","summary":"summary thought"},'
            '{"type":"reasoning.encrypted","data":"opaque"}]}}]}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":6,'
            '"completion_tokens_details":{"reasoning_tokens":4},"total_tokens":9}}',
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, content=payload.encode("utf-8"))
