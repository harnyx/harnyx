from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pytest
from pydantic import BaseModel

from harnyx_commons.llm.pricing import ModelPricing
from harnyx_commons.llm.providers.chutes import (
    ChutesLlmProvider,
    ChutesTextEmbeddingClient,
    _parse_chutes_response_payload,
    resolve_chutes_embedding_base_url,
)
from harnyx_commons.llm.providers.chutes_codec import (
    _ChutesChatRequest,
    _ChutesChatResponse,
    _ChutesReasoningStreamState,
    _ChutesToolCallPayload,
)
from harnyx_commons.llm.providers.chutes_pricing import ChutesModelPricingCache
from harnyx_commons.llm.providers.openai_stream import (
    OpenAiStreamError,
    OpenAiStreamState,
    _OpenAiStreamEvent,
    iter_openai_sse_events,
)
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmThinkingConfig,
    LlmUsage,
    PostprocessResult,
)


class _JudgeDecision(BaseModel):
    better: str


def test_chutes_provider_defaults_client_timeout_to_300(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("harnyx_commons.llm.providers.chutes.httpx.AsyncClient", _FakeAsyncClient)

    ChutesLlmProvider(base_url="https://llm.chutes.ai", api_key="key")

    assert captured["timeout"] == pytest.approx(300.0)


def _basic_chutes_request(
    *,
    model: str = "deepseek-ai/DeepSeek-V3.2-TEE",
    thinking: LlmThinkingConfig | None = None,
    reasoning_effort: str | None = None,
    extra: dict[str, object] | None = None,
) -> LlmRequest:
    return LlmRequest(
        provider="chutes",
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
        extra=extra,
    )


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


def test_parse_payload_rejects_response_containing_malformed_tool_call() -> None:
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

    with pytest.raises(RuntimeError, match="tool_calls"):
        _parse_chutes_response_payload(payload)


@pytest.mark.parametrize(
    "tool_call",
    (
        {"id": " ", "type": "function", "function": {"name": "summarize", "arguments": "{}"}},
        {"id": "tc-1", "type": "function", "function": {"name": " ", "arguments": "{}"}},
    ),
)
def test_parse_payload_rejects_whitespace_only_tool_call_identifiers(
    tool_call: dict[str, object],
) -> None:
    payload = {
        "id": "resp-whitespace",
        "choices": [{"message": {"content": None, "tool_calls": [tool_call]}}],
    }

    with pytest.raises(RuntimeError, match="tool_calls"):
        _parse_chutes_response_payload(payload)


def test_parse_payload_rejects_duplicate_tool_call_ids_within_one_block() -> None:
    tool_call = {
        "id": "tc-1",
        "type": "function",
        "function": {"name": "summarize", "arguments": "{}"},
    }
    payload = {
        "id": "resp-duplicate",
        "choices": [{"message": {"content": None, "tool_calls": [tool_call, tool_call]}}],
    }

    with pytest.raises(RuntimeError, match="unique"):
        _parse_chutes_response_payload(payload)


@pytest.mark.parametrize("arguments", ('{"x":NaN}', '{"x":Infinity}'))
def test_parse_payload_rejects_non_standard_json_constants_in_tool_arguments(
    arguments: str,
) -> None:
    payload = {
        "id": "resp-non-finite",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc-1",
                            "type": "function",
                            "function": {"name": "summarize", "arguments": arguments},
                        }
                    ],
                }
            }
        ],
    }

    parsed = _parse_chutes_response_payload(payload)
    with pytest.raises(ValueError, match="non-finite"):
        parsed.to_llm_response()


def test_chutes_tool_call_rejects_non_finite_values_in_object_arguments() -> None:
    tool_call = _ChutesToolCallPayload.model_validate(
        {
            "id": "tc-1",
            "type": "function",
            "function": {"name": "summarize", "arguments": {"x": float("nan")}},
        }
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        tool_call.to_tool_call(index=0)


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


def test_chutes_thinking_omitted_is_noop() -> None:
    payload = _ChutesChatRequest.from_request(_basic_chutes_request()).model_dump(
        mode="python",
        exclude_none=True,
    )

    assert payload["stream_options"] == {
        "include_usage": True,
        "continuous_usage_stats": True,
    }
    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload


def test_chutes_request_forces_stream_even_when_extra_overrides() -> None:
    payload = _ChutesChatRequest.from_request(
        _basic_chutes_request(extra={"stream": False})
    ).model_dump(mode="python", exclude_none=True)

    assert payload["stream"] is True


def test_chutes_request_serializes_complete_tool_loop() -> None:
    from harnyx_commons.llm.schema import LlmInputToolResultPart, LlmMessageToolCall, LlmTool

    request = replace(
        _basic_chutes_request(),
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
        tools=(LlmTool(type="function", function={"name": "lookup_weather", "strict": True}),),
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
        parallel_tool_calls=True,
    )

    payload = _ChutesChatRequest.from_request(request).model_dump(mode="json", exclude_none=True)

    assert payload["messages"][0]["tool_calls"][0]["id"] == "call-1"
    assert payload["messages"][1] == {
        "role": "tool",
        "content": '{"temperature":19}',
        "tool_call_id": "call-1",
    }
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup_weather"}}
    assert payload["parallel_tool_calls"] is True


def test_chutes_deepseek_thinking_enabled_and_disabled_use_template_kwargs() -> None:
    enabled = _ChutesChatRequest.from_request(
        _basic_chutes_request(thinking=LlmThinkingConfig(enabled=True))
    ).model_dump(mode="python", exclude_none=True)
    disabled = _ChutesChatRequest.from_request(
        _basic_chutes_request(thinking=LlmThinkingConfig(enabled=False))
    ).model_dump(mode="python", exclude_none=True)

    assert enabled["chat_template_kwargs"] == {"thinking": True}
    assert disabled["chat_template_kwargs"] == {"thinking": False}
    assert "reasoning_effort" not in enabled
    assert "reasoning_effort" not in disabled


def test_chutes_glm_thinking_disabled_uses_enable_thinking_template_kwarg() -> None:
    payload = _ChutesChatRequest.from_request(
        _basic_chutes_request(
            model="zai-org/GLM-5-TEE",
            thinking=LlmThinkingConfig(enabled=False),
        )
    ).model_dump(mode="python", exclude_none=True)

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize("model", ("Qwen/Qwen3.6-27B-TEE", "google/gemma-4-31B-turbo-TEE"))
@pytest.mark.parametrize(
    "thinking",
    (
        LlmThinkingConfig(enabled=True, effort="high"),
        LlmThinkingConfig(enabled=False, budget=1024),
    ),
)
def test_chutes_qwen_and_gemma_thinking_use_only_enable_thinking_template_kwarg(
    model: str,
    thinking: LlmThinkingConfig,
) -> None:
    payload = _ChutesChatRequest.from_request(
        _basic_chutes_request(model=model, thinking=thinking)
    ).model_dump(mode="python", exclude_none=True)

    assert payload["chat_template_kwargs"] == {"enable_thinking": thinking.enabled}
    assert "reasoning_effort" not in payload
    assert "budget" not in payload


def test_chutes_reasoning_effort_derives_template_thinking_for_capable_model() -> None:
    payload = _ChutesChatRequest.from_request(
        _basic_chutes_request(
            model="google/gemma-4-31B-turbo-TEE",
            reasoning_effort="high",
        )
    ).model_dump(mode="python", exclude_none=True)

    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in payload


def test_chutes_explicit_thinking_overrides_reasoning_effort() -> None:
    payload = _ChutesChatRequest.from_request(
        _basic_chutes_request(
            model="google/gemma-4-31B-turbo-TEE",
            thinking=LlmThinkingConfig(enabled=False),
            reasoning_effort="high",
        )
    ).model_dump(mode="python", exclude_none=True)

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in payload


def test_chutes_unsupported_reasoning_effort_capability_serializes_nothing() -> None:
    payload = _ChutesChatRequest.from_request(
        _basic_chutes_request(
            model="unsupported/model",
            reasoning_effort="high",
        )
    ).model_dump(mode="python", exclude_none=True)

    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload


def test_parse_payload_preserves_reasoning_usage_without_double_counting_completion_tokens() -> None:
    payload = {
        "id": "resp_reasoning_usage",
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "reasoning": "thinking trace",
                },
            },
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "reasoning_tokens": 2,
            "total_tokens": 7,
        },
    }

    response = _parse_chutes_response_payload(payload).to_llm_response()

    assert response.choices[0].message.reasoning == "thinking trace"
    assert response.usage.completion_tokens == 1
    assert response.usage.reasoning_tokens == 2
    assert response.usage.total_tokens == 7


def test_chutes_stream_preserves_reasoning_tokens_when_usage_progresses_with_reasoning() -> None:
    response = _merge_chutes_stream_events(
        (
            {
                "id": "resp-progress",
                "choices": [{"index": 0, "delta": {"reasoning_content": "think "}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "reasoning_tokens": 1, "total_tokens": 4},
            },
            {
                "id": "resp-progress",
                "choices": [{"index": 0, "delta": {"reasoning_content": "more"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "reasoning_tokens": 4, "total_tokens": 8},
            },
            {
                "id": "resp-progress",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 6, "reasoning_tokens": 4, "total_tokens": 9},
            },
        )
    )

    assert response.raw_text == "ok"
    assert response.choices[0].message.reasoning == "think more"
    assert response.usage.completion_tokens == 2
    assert response.usage.reasoning_tokens == 4


def test_chutes_stream_preserves_single_reasoning_usage_sample() -> None:
    response = _merge_chutes_stream_events(
        (
            {
                "id": "resp-single-usage",
                "choices": [{"index": 0, "delta": {"reasoning_content": "think"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "reasoning_tokens": 1, "total_tokens": 4},
            },
            {
                "id": "resp-single-usage",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "reasoning_tokens": 1, "total_tokens": 5},
            },
        )
    )

    assert response.raw_text == "ok"
    assert response.choices[0].message.reasoning == "think"
    assert response.usage.completion_tokens == 1
    assert response.usage.reasoning_tokens == 1


def test_chutes_stream_keeps_reasoning_tokens_unavailable_when_usage_does_not_progress() -> None:
    response = _merge_chutes_stream_events(
        (
            {
                "id": "resp-stalled",
                "choices": [{"index": 0, "delta": {"reasoning_content": "think "}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "reasoning_tokens": 1, "total_tokens": 4},
            },
            {
                "id": "resp-stalled",
                "choices": [{"index": 0, "delta": {"reasoning_content": "more"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "reasoning_tokens": 1, "total_tokens": 8},
            },
            {
                "id": "resp-stalled",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 6, "reasoning_tokens": 1, "total_tokens": 9},
            },
        )
    )

    assert response.raw_text == "ok"
    assert response.choices[0].message.reasoning == "think more"
    assert response.usage.completion_tokens == 6
    assert response.usage.reasoning_tokens is None


def test_chutes_stream_keeps_reasoning_tokens_unavailable_with_final_only_usage() -> None:
    response = _merge_chutes_stream_events(
        (
            {
                "id": "resp-final-only",
                "choices": [{"index": 0, "delta": {"reasoning_content": "think "}}],
            },
            {
                "id": "resp-final-only",
                "choices": [{"index": 0, "delta": {"reasoning_content": "more"}}],
            },
            {
                "id": "resp-final-only",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 6, "reasoning_tokens": 1, "total_tokens": 9},
            },
        )
    )

    assert response.raw_text == "ok"
    assert response.choices[0].message.reasoning == "think more"
    assert response.usage.completion_tokens == 6
    assert response.usage.reasoning_tokens is None


def _merge_chutes_stream_events(events: tuple[dict[str, object], ...]) -> LlmResponse:
    openai_state = OpenAiStreamState()
    reasoning_state = _ChutesReasoningStreamState()
    for event_payload in events:
        event = _OpenAiStreamEvent.model_validate(event_payload)
        openai_state.merge_event(event, reasoning_keys=("reasoning", "reasoning_content"))
        reasoning_state.merge_event(event)
    return _ChutesChatResponse.from_stream_state(openai_state, reasoning_state=reasoning_state).to_llm_response()


@pytest.mark.anyio("asyncio")
async def test_chutes_provider_persists_stream_ttft_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = {
            "id": "resp-ttft",
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, text=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n")

    provider = ChutesLlmProvider(
        base_url="https://example.com",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
    )

    try:
        response = await provider.invoke(_basic_chutes_request())
    finally:
        await provider.aclose()

    assert response.metadata is not None
    assert isinstance(response.metadata["ttft_ms"], float)
    assert response.metadata["ttft_ms"] >= 0.0


@pytest.mark.anyio("asyncio")
async def test_chutes_provider_preserves_structured_reasoning_raw_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        reasoning_payload = {
            "id": "resp-reasoning-raw",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning": {
                            "thought_text_parts": ["step one", "step two"],
                            "has_thought_signature": True,
                        }
                    },
                }
            ],
        }
        content_payload = {
            "id": "resp-reasoning-raw",
            "choices": [{"delta": {"content": "final answer"}, "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
        }
        return httpx.Response(
            200,
            text=(
                f"data: {json.dumps(reasoning_payload)}\n\n"
                f"data: {json.dumps(content_payload)}\n\n"
                "data: [DONE]\n\n"
            ),
        )

    provider = ChutesLlmProvider(
        base_url="https://example.com",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
    )

    try:
        response = await provider.invoke(_basic_chutes_request())
    finally:
        await provider.aclose()

    assert response.raw_text == "final answer"
    assert response.choices[0].message.reasoning == "step one\n\nstep two"
    assert response.metadata is not None
    assert response.metadata["raw_response"]["choices"][0]["message"]["reasoning"] == {
        "thought_text_parts": ["step one", "step two"],
        "has_thought_signature": True,
    }


@pytest.mark.anyio("asyncio")
async def test_chutes_provider_attaches_actual_cost_from_cached_pricing_without_live_fetch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = {
            "id": "resp-cost",
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 2_000, "total_tokens": 3_000},
        }
        return httpx.Response(200, text=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n")

    provider = ChutesLlmProvider(
        base_url="https://example.com",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
        pricing_cache=ChutesModelPricingCache(
            cached_pricing={"deepseek-ai/DeepSeek-V3.2-TEE": ModelPricing(0.10, 0.20, 0.0)}
        ),
    )
    try:
        response = await provider.invoke(_basic_chutes_request())
    finally:
        await provider.aclose()

    assert response.metadata is not None
    assert response.metadata["actual_cost_provider"] == "chutes"
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.0005)
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "cached_provider_pricing"
    assert response.metadata["actual_cost_evidence"]["pricing_origin"] == "chutes_live_snapshot"


@pytest.mark.anyio("asyncio")
async def test_chutes_provider_attaches_actual_cost_from_static_pricing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = {
            "id": "resp-cost-fallback",
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 2_000, "total_tokens": 3_000},
        }
        return httpx.Response(200, text=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n")

    provider = ChutesLlmProvider(
        base_url="https://example.com",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
    )

    try:
        response = await provider.invoke(_basic_chutes_request(model="Qwen/Qwen3.6-27B-TEE"))
    finally:
        await provider.aclose()

    assert response.metadata is not None
    assert response.metadata["actual_cost_provider"] == "chutes"
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.0045)
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "static_pricing"
    assert response.metadata["actual_cost_evidence"]["pricing_origin"] == "chutes_repo_rates"


@pytest.mark.anyio("asyncio")
async def test_chutes_provider_accumulates_per_response_cost_after_provider_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/v1/chat/completions"
        text = "bad" if calls == 1 else "ok"
        payload = {
            "id": f"resp-cost-retry-{calls}",
            "choices": [{"delta": {"content": text}, "finish_reason": "stop", "index": 0}],
            "usage": {
                "prompt_tokens": calls * 1_000,
                "completion_tokens": calls * 1_000,
                "total_tokens": calls * 2_000,
            },
        }
        return httpx.Response(200, text=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n")

    def postprocessor(response: LlmResponse) -> PostprocessResult:
        if response.raw_text == "ok":
            return PostprocessResult(ok=True, retryable=False)
        return PostprocessResult(ok=False, retryable=True, reason="bad response")

    provider = ChutesLlmProvider(
        base_url="https://example.com",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
        pricing_cache=ChutesModelPricingCache(
            cached_pricing={"deepseek-ai/DeepSeek-V3.2-TEE": ModelPricing(0.10, 0.20, 0.0)}
        ),
    )
    provider._retry_policy = RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0)

    try:
        response = await provider.invoke(replace(_basic_chutes_request(), postprocessor=postprocessor))
    finally:
        await provider.aclose()

    assert calls == 2
    assert response.usage.prompt_tokens == 3_000
    assert response.usage.completion_tokens == 3_000
    assert response.metadata is not None
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.0006)
    assert response.metadata["actual_cost_usd_total"] == pytest.approx(0.0009)
    assert response.metadata["actual_cost_evidence"]["prompt_tokens"] == 2_000
    assert response.metadata["actual_cost_evidence"]["completion_tokens"] == 2_000


@pytest.mark.anyio("asyncio")
async def test_chutes_provider_uses_request_retry_policy_over_default() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/v1/chat/completions"
        if calls == 1:
            return httpx.Response(429, json={"error": "capacity"})
        payload = {
            "id": "resp-retry-success",
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, text=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n")

    provider = ChutesLlmProvider(
        base_url="https://example.com",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
        pricing_cache=ChutesModelPricingCache(
            cached_pricing={"deepseek-ai/DeepSeek-V3.2-TEE": ModelPricing(0.10, 0.20, 0.0)}
        ),
    )
    provider._retry_policy = RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0.0)
    request = replace(
        _basic_chutes_request(),
        retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0),
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert calls == 2
    assert response.raw_text == "ok"


@pytest.mark.anyio("asyncio")
async def test_chutes_provider_records_ttft_on_reasoning_only_first_stream_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_seconds = 10.0

    def fake_perf_counter() -> float:
        return clock_seconds

    monkeypatch.setattr("harnyx_commons.llm.providers.chutes.time.perf_counter", fake_perf_counter)

    class _DelayedReasoningThenContentStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal clock_seconds
            reasoning_payload = {
                "id": "resp-reasoning-first",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning": {
                                "thought_text_parts": ["reasoning trace"],
                                "has_thought_signature": True,
                            }
                        },
                    }
                ],
            }
            clock_seconds = 10.05
            yield f"data: {json.dumps(reasoning_payload)}\n\n".encode()
            await asyncio.sleep(0)
            content_payload = {
                "id": "resp-reasoning-first",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "final answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
            clock_seconds = 10.2
            yield f"data: {json.dumps(content_payload)}\n\ndata: [DONE]\n\n".encode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, stream=_DelayedReasoningThenContentStream(), request=request)

    provider = ChutesLlmProvider(
        base_url="https://example.com",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
    )

    try:
        response = await provider.invoke(_basic_chutes_request())
    finally:
        await provider.aclose()

    assert response.raw_text == "final answer"
    assert response.choices[0].message.reasoning == "reasoning trace"
    assert response.metadata is not None
    assert response.metadata["ttft_ms"] == pytest.approx(50.0)


def test_resolve_chutes_embedding_base_url_returns_expected_live_base_url() -> None:
    base_url = resolve_chutes_embedding_base_url("Qwen/Qwen3-Embedding-8B-TEE")

    assert base_url == "https://chutes-qwen-qwen3-embedding-8b-tee.chutes.ai"


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
        model="Qwen/Qwen3-Embedding-8B-TEE",
        base_url="https://example.com",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
        api_key="test-key",
        dimensions=3,
    )

    vector = await client.embed("hello world")

    assert vector == (0.25, 0.5, 0.75)
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/embeddings"
    assert '"model":"Qwen/Qwen3-Embedding-8B-TEE"' in str(captured["json"])
    assert '"input":"hello world"' in str(captured["json"])


async def test_chutes_text_embedding_client_splits_multi_text_requests() -> None:
    requests: list[dict[str, object]] = []
    timeouts: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode())
        requests.append(payload)
        timeouts.append(request.extensions["timeout"])
        index = len(requests)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "embedding": [float(index), float(index + 1)],
                        "index": 0,
                        "object": "embedding",
                    }
                ],
                "usage": {
                    "prompt_tokens": index,
                    "total_tokens": index + 1,
                },
            },
        )

    client = ChutesTextEmbeddingClient(
        model="Qwen/Qwen3-Embedding-8B-TEE",
        base_url="https://example.com",
        client=httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(handler)),
        api_key="test-key",
        dimensions=2,
    )

    response = await client.embed_many(("first", "second"), timeout_seconds=190.0)

    assert [request["input"] for request in requests] == ["first", "second"]
    assert all(isinstance(request["input"], str) for request in requests)
    assert timeouts == [
        {"connect": 190.0, "read": 190.0, "write": 190.0, "pool": 190.0},
        {"connect": 190.0, "read": 190.0, "write": 190.0, "pool": 190.0},
    ]
    assert response.vectors == ((1.0, 2.0), (2.0, 3.0))
    assert response.usage is not None
    assert response.usage.prompt_tokens == 3
    assert response.usage.total_tokens == 5


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
async def test_iter_openai_sse_events_classifies_truncated_json_as_retryable_stream_error() -> None:
    response = httpx.Response(
        200,
        text='data: {"choices":[{"delta":{"content":"unterminated',
    )

    with pytest.raises(OpenAiStreamError) as exc_info:
        async for _ in iter_openai_sse_events(
            response,
            invalid_data_message="invalid_data",
            invalid_event_message="invalid_event",
        ):
            pass

    exc = exc_info.value
    assert exc.message == "invalid_data"
    assert exc.code == 502
    assert exc.error_type == "server_error"
    assert exc.retryable is True


@pytest.mark.anyio("asyncio")
async def test_iter_openai_sse_events_rejects_wrapped_event_envelope() -> None:
    response = httpx.Response(
        200,
        text='data: {"event":{"choices":[{"message":{"content":"ok"}}]}}\n\n',
    )

    with pytest.raises(OpenAiStreamError, match="invalid_data") as exc_info:
        async for _ in iter_openai_sse_events(
            response,
            invalid_data_message="invalid_data",
            invalid_event_message="invalid_event",
        ):
            pass

    exc = exc_info.value
    assert exc.message == "invalid_data"
    assert exc.code == 502
    assert exc.error_type == "server_error"
    assert exc.retryable is True


@pytest.mark.anyio("asyncio")
async def test_iter_openai_sse_events_rejects_non_object_event_as_stream_error() -> None:
    response = httpx.Response(
        200,
        text="data: []\n\n",
    )

    with pytest.raises(OpenAiStreamError, match="invalid_event") as exc_info:
        async for _ in iter_openai_sse_events(
            response,
            invalid_data_message="invalid_data",
            invalid_event_message="invalid_event",
        ):
            pass

    exc = exc_info.value
    assert exc.message == "invalid_event"
    assert exc.code == 502
    assert exc.error_type == "server_error"
    assert exc.retryable is True


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


def test_streamed_reasoning_content_is_preserved_as_reasoning_clue() -> None:
    state = OpenAiStreamState()
    reasoning_state = _ChutesReasoningStreamState()

    first_event = _OpenAiStreamEvent.model_validate(
        {"choices": [{"index": 0, "delta": {"content": "ok", "reasoning_content": "step"}}]}
    )
    second_event = _OpenAiStreamEvent.model_validate(
        {"choices": [{"index": 0, "delta": {"reasoning_content": " two"}, "finish_reason": "stop"}]}
    )

    reasoning_state.merge_event(first_event)
    assert state.merge_event(first_event, reasoning_keys=()) is True
    reasoning_state.merge_event(second_event)
    assert state.merge_event(second_event, reasoning_keys=()) is False

    response = _ChutesChatResponse.from_stream_state(state, reasoning_state=reasoning_state).to_llm_response()

    assert response.choices[0].message.reasoning == "step two"


def test_streamed_multipart_reasoning_content_is_preserved_as_reasoning_clue() -> None:
    state = OpenAiStreamState()
    reasoning_state = _ChutesReasoningStreamState()

    event = _OpenAiStreamEvent.model_validate(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "ok",
                        "reasoning_content": [
                            {"type": "text", "text": "step"},
                            {"type": "text", "text": " two"},
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    reasoning_state.merge_event(event)
    assert state.merge_event(event, reasoning_keys=()) is True

    response = _ChutesChatResponse.from_stream_state(state, reasoning_state=reasoning_state).to_llm_response()

    assert response.choices[0].message.reasoning == "step two"


def test_streamed_mirrored_reasoning_keys_are_deduplicated_per_event() -> None:
    state = OpenAiStreamState()
    reasoning_state = _ChutesReasoningStreamState()

    event = _OpenAiStreamEvent.model_validate(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "ok",
                        "reasoning": "think",
                        "reasoning_content": "think",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    reasoning_state.merge_event(event)
    assert state.merge_event(event, reasoning_keys=()) is True

    response = _ChutesChatResponse.from_stream_state(state, reasoning_state=reasoning_state).to_llm_response()

    assert response.choices[0].message.reasoning == "think"


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
