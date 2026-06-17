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
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmThinkingConfig

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
    assert isinstance(response.metadata["ttft_ms"], float)
    assert response.metadata["ttft_ms"] >= 0.0


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
