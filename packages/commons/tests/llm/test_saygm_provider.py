"""Protect SayGM credential routing, streaming tools, and settled accounting."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm import provider_factory
from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.cost_settlement import settled_response_cost
from harnyx_commons.llm.provider import LlmProviderError, LlmRetryExhaustedError
from harnyx_commons.llm.providers.saygm import SayGmLlmProvider
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmThinkingConfig,
    LlmTool,
)
from harnyx_commons.llm.tool_models import parse_miner_selected_llm_provider_model
from harnyx_miner_sdk.llm import Timeout

pytestmark = pytest.mark.anyio("asyncio")


async def test_cached_internal_and_explicit_miner_credentials_stay_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    sent_keys = []
    clients = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_keys.append(request.headers["Authorization"])
        return httpx.Response(200, content=_stream(delta={"content": "ok"}, usage={"cost_nano_usd": 1}))

    def build_provider(*, api_key: SecretStr) -> SayGmLlmProvider:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return SayGmLlmProvider(api_key=api_key, client=client)

    monkeypatch.setattr(provider_factory, "SayGmLlmProvider", build_provider)
    settings = LlmSettings(SAYGM_API_KEY="platform-key", _env_file=None)
    registry = provider_factory.build_cached_llm_provider_registry(
        llm_settings=settings,
        bedrock_settings=BedrockSettings(_env_file=None),
        vertex_settings=VertexSettings(_env_file=None),
    )
    internal = registry.resolve("saygm")
    assert registry.resolve("saygm") is internal
    miner = provider_factory.build_miner_paid_llm_provider(provider="saygm", api_key="miner-key", llm_settings=settings)
    request = LlmRequest(
        provider="saygm",
        model="gpt-oss-20b",
        temperature=None,
        max_output_tokens=None,
        messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Hello"),)),),
    )
    try:
        await internal.invoke(request)
        await miner.invoke(request)
        await internal.invoke(request)
    finally:
        await registry.aclose()
        await miner.aclose()
        for client in clients:
            await client.aclose()
    assert sent_keys == ["Bearer platform-key", "Bearer miner-key", "Bearer platform-key"]


def _stream(*, delta: dict, usage: dict) -> bytes:
    events = [
        {"id": "gm-test", "choices": [{"index": 0, "delta": delta}]},
        {"id": "gm-test", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "gm-test", "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 9, **usage}},
    ]
    return ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode()


async def test_internal_alias_and_miner_native_routes_stream_and_settle_with_explicit_key() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_stream(delta={"content": "ok"}, usage={"cost": 0.00014, "cost_nano_usd": 140000}),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = LlmProviderAdapter(
            provider_name="saygm", delegate=SayGmLlmProvider(api_key=SecretStr("miner-key"), client=client)
        )
        internal = LlmRequest(
            provider="saygm",
            temperature=None,
            max_output_tokens=None,
            model="google/gemma-4-31B-turbo-TEE",
            messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Hello"),)),),
        )
        for request in (
            internal,
            replace(internal, model="gpt-oss-20b", thinking=LlmThinkingConfig(enabled=True, effort="low")),
            replace(internal, model="zai-org/GLM-5.1-TEE"),
        ):
            response = await provider.invoke(request)
            assert response.raw_text == "ok"
            assert response.metadata["actual_cost_provider"] == "saygm"
            assert response.metadata["actual_cost_usd"] == pytest.approx(0.00014)
            assert response.metadata["actual_cost_evidence"]["cost_nano_usd"] == 140000
        await provider.aclose()
        assert not client.is_closed

    assert all(str(request.url) == "https://api.saygm.com/v1/chat/completions" for request in requests)
    assert all(request.headers["Authorization"] == "Bearer miner-key" for request in requests)
    first, second, third = [json.loads(request.content) for request in requests]
    assert first["model"] == "gemma-4-31b-turbo-tee"
    assert second["model"] == "gpt-oss-20b"
    assert third["model"] == "glm-5.1-tee"
    assert second["reasoning_effort"] == "low"
    assert second["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in second


async def test_function_tool_loop_keeps_call_ids_arguments_and_integer_zero_settlement() -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        delta = (
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"id":1}'},
                    }
                ]
            }
            if len(bodies) == 1
            else {"content": "Found"}
        )
        return httpx.Response(200, content=_stream(delta=delta, usage={"cost_nano_usd": 0}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        request = LlmRequest(
            provider="saygm",
            temperature=None,
            max_output_tokens=None,
            model="gpt-oss-20b",
            messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Look up 1"),)),),
            tools=(
                LlmTool(
                    type="function",
                    function={
                        "name": "lookup",
                        "description": "Look up an ID",
                        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    },
                ),
            ),
        )
        first = await provider.invoke(request)
        call = first.choices[0].message.tool_calls[0]
        assert call.id == "call-1"
        assert json.loads(call.arguments) == {"id": 1}
        second = await provider.invoke(
            replace(
                request,
                messages=(
                    *request.messages,
                    LlmMessage(
                        role="assistant",
                        content=(LlmMessageContentPart.input_text(""),),
                        tool_calls=(
                            LlmMessageToolCall(id=call.id, type="function", name=call.name, arguments=call.arguments),
                        ),
                    ),
                    LlmMessage(
                        role="tool",
                        content=(
                            LlmInputToolResultPart(tool_call_id=call.id, name=call.name, output_json='{"found":true}'),
                        ),
                    ),
                ),
            )
        )
        assert second.raw_text == "Found"
        assert second.metadata["actual_cost_usd"] == 0.0
    assert bodies[1]["messages"][-1]["tool_call_id"] == "call-1"


@pytest.mark.parametrize("usage", [{}, {"cost": -1.0}, {"cost_nano_usd": True}, {"cost": 0.1, "cost_nano_usd": 1}])
def test_missing_or_invalid_settlement_never_uses_catalog_estimates(usage: dict) -> None:
    response = LlmResponse(
        id="gm-test", choices=(), usage=None, metadata={"raw_response": {"id": "gm-test", "usage": usage}}
    )
    assert settled_response_cost(response, provider="saygm", model="gpt-oss-20b") is None


async def test_billable_empty_output_retry_accumulates_all_settled_costs() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200, content=_stream(delta={"content": "" if attempts == 1 else "ok"}, usage={"cost_nano_usd": 100000})
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        response = await provider.invoke(
            LlmRequest(
                provider="saygm",
                model="gpt-oss-20b",
                temperature=None,
                max_output_tokens=None,
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Hello"),)),),
                retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
            )
        )
    assert attempts == 2
    assert response.metadata["actual_cost_usd_total"] == pytest.approx(0.0002)


def test_miner_selection_accepts_native_ids_and_rejects_other_api_surfaces_and_internal_aliases() -> None:
    assert parse_miner_selected_llm_provider_model(provider="saygm", model="gpt-oss-20b").model == "gpt-oss-20b"
    for model in (
        "gpt-6-luna",
        "glm-5.1-tee",
        "kimi-k3-tee",
        "glm-5.3",
        "ornith-1.5-397b",
        "qwen3.6-35b-a3b",
        "claude-sonnet-5",
        "gemini-3.8-flash",
        "google/gemma-4-31B-turbo-TEE",
    ):
        with pytest.raises(ValueError, match="not supported"):
            parse_miner_selected_llm_provider_model(provider="saygm", model=model)


def _retry_request(**updates: object) -> LlmRequest:
    return replace(
        LlmRequest(
            provider="saygm",
            model="gpt-oss-20b",
            temperature=None,
            max_output_tokens=None,
            messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Hello"),)),),
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
            timeout=Timeout(0.5, inactivity=0.02),
        ),
        **updates,
    )


class _InterruptedStream(httpx.AsyncByteStream):
    def __init__(self, delta: dict | None, failure: str) -> None:
        self.delta = delta
        self.failure = failure
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.delta is not None:
            yield f'data: {json.dumps({"choices": [{"index": 0, "delta": self.delta}]})}\n\n'.encode()
        if self.failure == "inactivity":
            await asyncio.sleep(1)
        raise httpx.ReadError("stream disconnected")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "delta",
    [
        {"content": "partial"},
        {"reasoning_content": "thinking"},
        {"tool_calls": [{"index": 0, "id": "call-1", "type": "function", "function": {"name": "lookup"}}]},
    ],
)
@pytest.mark.parametrize("failure", ["transport", "inactivity"])
async def test_interrupted_output_is_terminal_and_closes_stream(delta: dict, failure: str) -> None:
    streams = []

    def handler(request: httpx.Request) -> httpx.Response:
        stream = _InterruptedStream(delta, failure)
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        with pytest.raises(LlmProviderError, match="stream_interrupted_after_output"):
            await provider.invoke(_retry_request())
    assert len(streams) == 1
    assert streams[0].closed


async def test_output_in_one_call_does_not_disable_another_calls_prefill_retry() -> None:
    attempts: dict[str, int] = {}
    both_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        name = json.loads(request.content)["messages"][0]["content"]
        attempts[name] = attempts.get(name, 0) + 1
        if len(attempts) == 2:
            both_started.set()
        await both_started.wait()
        if name == "output":
            return httpx.Response(200, stream=_InterruptedStream({"content": "partial"}, "inactivity"))
        if attempts[name] == 1:
            return httpx.Response(200, stream=_InterruptedStream(None, "transport"))
        return httpx.Response(200, content=_stream(delta={"content": "ok"}, usage={"cost_nano_usd": 1}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        requests = [
            _retry_request(messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text(name),)),))
            for name in ("output", "prefill")
        ]
        failed, succeeded = await asyncio.gather(
            *(provider.invoke(request) for request in requests), return_exceptions=True
        )
    assert isinstance(failed, LlmProviderError)
    assert isinstance(succeeded, LlmResponse)
    assert succeeded.raw_text == "ok"
    assert attempts == {"output": 1, "prefill": 2}


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize(
    "header_kind, expected_delay", [("seconds", 30), ("date", 29.5), ("invalid", 0), ("absent", 0)]
)
async def test_retry_after_controls_delay_before_retry(
    monkeypatch: pytest.MonkeyPatch, status: int, header_kind: str, expected_delay: float
) -> None:
    delays = []
    attempts = 0
    now = datetime(2026, 10, 13, 12, 0, 0, 500_000, tzinfo=UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr("harnyx_commons.llm.providers.saygm.datetime", FixedDatetime)
    retry_at = now + timedelta(seconds=30)
    headers = {
        "seconds": {"Retry-After": "30"},
        "date": {"Retry-After": format_datetime(retry_at, usegmt=True)},
        "invalid": {"Retry-After": "invalid"},
        "absent": {},
    }[header_kind]

    async def sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status, headers=headers)
        return httpx.Response(200, content=_stream(delta={"content": "ok"}, usage={"cost_nano_usd": 1}))

    monkeypatch.setattr("harnyx_commons.llm.provider.asyncio.sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        response = await provider.invoke(_retry_request())
    assert response.raw_text == "ok"
    assert attempts == 2
    assert len(delays) == 1
    assert delays[0] == expected_delay


async def test_retry_after_does_not_consume_next_attempt_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    delays = []
    real_sleep = asyncio.sleep

    async def sleep(delay: float) -> None:
        delays.append(delay)
        # Model a provider-directed delay longer than a complete attempt budget.
        await real_sleep(0.1)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "60"})
        return httpx.Response(200, content=_stream(delta={"content": "ok"}, usage={"cost_nano_usd": 1}))

    monkeypatch.setattr("harnyx_commons.llm.provider.asyncio.sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        response = await provider.invoke(_retry_request(timeout=Timeout(0.05, prefill=0.02)))
    assert response.raw_text == "ok"
    assert attempts == 2
    assert delays == [60]


@pytest.mark.parametrize("phase", ["prefill", "total"])
async def test_timeout_before_output_gives_retry_fresh_deadlines(phase: str) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(1)
        return httpx.Response(200, content=_stream(delta={"content": "ok"}, usage={"cost_nano_usd": 1}))

    timeout = Timeout(0.2, prefill=0.02) if phase == "prefill" else Timeout(0.05)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        response = await provider.invoke(_retry_request(timeout=timeout))
    assert response.raw_text == "ok"
    assert attempts == 2
    assert response.metadata["retry_reasons"] == (f"{phase}_timeout",)


@pytest.mark.parametrize("failure", ["backoff", "before_output", "after_output"])
async def test_retry_failure_preserves_previous_settled_charge_and_attempt_count(failure: str) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, content=_stream(delta={"content": ""}, usage={"cost_nano_usd": 100000}))
        if failure == "backoff":
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            stream=_InterruptedStream({"content": "partial"} if failure == "after_output" else None, "inactivity"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        exception_type = LlmProviderError if failure == "after_output" else LlmRetryExhaustedError
        with pytest.raises(exception_type) as raised:
            await provider.invoke(
                _retry_request(
                    timeout=0.05,
                    retry_policy=RetryPolicy(attempts=3, initial_ms=0, max_ms=0, jitter=0),
                )
            )
    assert attempts == (2 if failure == "after_output" else 3)
    response = raised.value.response
    assert response is not None
    assert response.usage.prompt_tokens == 12
    assert response.metadata["actual_cost_usd_total"] == pytest.approx(0.0001)
    assert response.metadata["billable_response_count"] == 1
    assert response.metadata["attempts"] == attempts
    assert response.metadata["retry_reasons"][0] == "empty_output"


async def test_unsupported_internal_thinking_budget_fails_before_http_dispatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("unsupported thinking budget reached the provider")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        with pytest.raises(LlmProviderError, match="does not support thinking.budget"):
            await provider.invoke(_retry_request(thinking=LlmThinkingConfig(enabled=True, budget=2048)))


async def test_prefill_timeout_can_retry_before_output() -> None:
    stream = _InterruptedStream(None, "inactivity")
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, stream=stream)
        return httpx.Response(200, content=_stream(delta={"content": "ok"}, usage={"cost_nano_usd": 1}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        response = await provider.invoke(_retry_request(timeout=Timeout(0.5, prefill=0.02)))
    assert response.raw_text == "ok"
    assert attempts == 2
    assert stream.closed


async def test_caller_cancellation_closes_stream_without_retry() -> None:
    output_started = asyncio.Event()
    attempts = 0

    class WaitingStream(_InterruptedStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
            output_started.set()
            await asyncio.Event().wait()

    stream = WaitingStream(None, "transport")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SayGmLlmProvider(api_key=SecretStr("key"), client=client)
        call = asyncio.create_task(provider.invoke(_retry_request()))
        await output_started.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
    assert attempts == 1
    assert stream.closed
