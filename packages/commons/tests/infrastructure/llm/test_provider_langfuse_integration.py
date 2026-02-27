from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType

import pytest

import caster_commons.llm.provider as provider_module
from caster_commons.llm.provider import BaseLlmProvider
from caster_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

pytestmark = pytest.mark.anyio("asyncio")


@dataclass
class _Scope:
    generation: object | None
    entered: int = 0
    exited: int = 0

    def __enter__(self) -> object | None:
        self.entered += 1
        return self.generation

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.exited += 1
        return False


@dataclass(frozen=True)
class _UpdateCall:
    generation: object | None
    input_payload: object | None
    output: object | None
    usage: LlmUsage | None
    metadata: Mapping[str, object] | None


class _StubProvider(BaseLlmProvider):
    def __init__(
        self,
        *,
        response: LlmResponse | None = None,
        error: Exception | None = None,
        provider_label: str = "openai",
    ) -> None:
        super().__init__(provider_label=provider_label)
        self._response = response
        self._error = error
        self.requests: list[AbstractLlmRequest] = []

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._response is None:
            raise RuntimeError("stub response must be configured")
        return self._response


class _VerifierFailureProvider(BaseLlmProvider):
    def __init__(self, *, response: LlmResponse) -> None:
        super().__init__(provider_label="vertex")
        self._response = response
        self.requests: list[AbstractLlmRequest] = []

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)

        async def _call() -> LlmResponse:
            return self._response

        def _always_fail_verifier(_: LlmResponse) -> tuple[bool, bool, str | None]:
            return False, False, "empty_output"

        return await self._call_with_retry(
            request,
            call_coro=_call,
            verifier=_always_fail_verifier,
        )


def _request(
    *,
    provider: str = "openai",
    model: str = "gpt-5-mini",
    reasoning_effort: str | None = None,
    internal_metadata: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> LlmRequest:
    return LlmRequest(
        provider=provider,
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        reasoning_effort=reasoning_effort,
        output_mode="text",
        internal_metadata=internal_metadata,
        extra=extra,
    )


def _response(
    *,
    metadata: Mapping[str, object] | None = None,
    usage: LlmUsage | None = None,
) -> LlmResponse:
    return LlmResponse(
        id="response-id",
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
        usage=usage or LlmUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        metadata=metadata,
        finish_reason="stop",
    )


async def test_invoke_success_updates_generation_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "test-server")
    request = _request(
        internal_metadata={
            "use_case": "claim_generation",
            "feed_run_id": "feed-run-123",
        }
    )
    response = _response(metadata={"source": "stub", "raw_response": {"response_id": "provider-raw"}})
    provider = _StubProvider(response=response)
    scope = _Scope(generation=object())
    start_calls: list[dict[str, object]] = []
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        start_calls.append(
            {
                "provider_label": provider_label,
                "request": request,
            }
        )
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(
            _UpdateCall(
                generation=generation,
                input_payload=input_payload,
                output=output,
                usage=usage,
                metadata=metadata,
            )
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    result = await provider.invoke(request)

    assert result == response
    assert provider.requests == [request]
    assert scope.entered == 1
    assert scope.exited == 1

    assert len(start_calls) == 1
    assert start_calls[0]["provider_label"] == "openai"
    assert start_calls[0]["request"] is request

    assert len(update_calls) == 1
    update_call = update_calls[0]
    assert update_call.generation is scope.generation
    assert update_call.input_payload is None
    assert update_call.output == {
        "assistant": {"role": "assistant", "text": "ok"},
        "finish_reason": "stop",
    }
    assert update_call.usage == response.usage
    assert update_call.metadata is not None
    assert update_call.metadata["provider"] == "openai"
    assert update_call.metadata["server"] == "test-server"
    assert update_call.metadata["use_case"] == "claim_generation"
    assert update_call.metadata["feed_run_id"] == "feed-run-123"
    assert update_call.metadata["finish_reason"] == "stop"
    assert update_call.metadata["response_metadata"] == {
        "source": "stub",
        "raw_response": {"response_id": "provider-raw"},
    }
    assert update_call.metadata["raw"] == {
        "request": provider_module._request_snapshot(request),
        "response_payload": response.payload,
        "response_metadata": {"source": "stub", "raw_response": {"response_id": "provider-raw"}},
        "provider_response": {"response_id": "provider-raw"},
    }

    elapsed_ms = update_call.metadata["elapsed_ms"]
    wait_ms = update_call.metadata["wait_ms"]
    assert isinstance(elapsed_ms, float)
    assert isinstance(wait_ms, float)


async def test_invoke_success_handles_json_safe_vertex_thought_signature_in_raw_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(provider="vertex", model="gemini-2.5-pro", reasoning_effort="high")
    response = _response(
        metadata={
            "source": "stub",
            "raw_response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thought_signature": "ZmFrZS10aG91Z2h0LXNpZw==",
                                }
                            ]
                        }
                    }
                ]
            },
        }
    )
    provider = _StubProvider(response=response, provider_label="vertex")
    scope = _Scope(generation=object())
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(
            _UpdateCall(
                generation=generation,
                input_payload=input_payload,
                output=output,
                usage=usage,
                metadata=metadata,
            )
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    result = await provider.invoke(request)

    assert result == response
    assert len(update_calls) == 1
    update_call = update_calls[0]
    assert update_call.metadata is not None
    raw = update_call.metadata["raw"]
    assert isinstance(raw, Mapping)
    assert raw["response_payload"] == response.payload
    assert raw["provider_response"] == {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "thought_signature": "ZmFrZS10aG91Z2h0LXNpZw==",
                        }
                    ]
                }
            }
        ]
    }


async def test_invoke_skips_child_observation_recording_when_generation_scope_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    response = _response(metadata={"source": "stub"})
    provider = _StubProvider(response=response)
    scope = _Scope(generation=None)
    child_record_calls: list[dict[str, object]] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        return None

    def fake_record_child_observations(
        *,
        provider_label: str,
        model: str,
        response: LlmResponse,
        response_metadata: Mapping[str, object],
    ) -> None:
        child_record_calls.append(
            {
                "provider_label": provider_label,
                "model": model,
                "response": response,
                "response_metadata": dict(response_metadata),
            }
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)
    monkeypatch.setattr(provider_module, "_record_child_observations", fake_record_child_observations)

    result = await provider.invoke(request)

    assert result == response
    assert provider.requests == [request]
    assert scope.entered == 1
    assert scope.exited == 1
    assert child_record_calls == []


async def test_invoke_error_updates_generation_error_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    provider = _StubProvider(error=ValueError("invoke failed"))
    scope = _Scope(generation=object())
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(
            _UpdateCall(
                generation=generation,
                input_payload=input_payload,
                output=output,
                usage=usage,
                metadata=metadata,
            )
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    with pytest.raises(ValueError, match="invoke failed"):
        await provider.invoke(request)

    assert provider.requests == [request]
    assert scope.entered == 1
    assert scope.exited == 1
    assert len(update_calls) == 1

    update_call = update_calls[0]
    assert update_call.generation is scope.generation
    assert update_call.input_payload is None
    assert update_call.output is None
    assert update_call.usage is None
    assert update_call.metadata is not None

    error_value = update_call.metadata["error"]
    assert isinstance(error_value, str)
    assert "ValueError" in error_value
    assert "invoke failed" in error_value
    assert isinstance(update_call.metadata["elapsed_ms"], float)
    assert isinstance(update_call.metadata["wait_ms"], float)


async def test_invoke_verifier_failure_includes_raw_payload_in_error_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(provider="vertex", model="gemini-2.5-pro")
    response = _response(metadata={"source": "stub", "raw_response": {"response_id": "provider-raw"}})
    provider = _VerifierFailureProvider(response=response)
    scope = _Scope(generation=object())
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(
            _UpdateCall(
                generation=generation,
                input_payload=input_payload,
                output=output,
                usage=usage,
                metadata=metadata,
            )
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    with pytest.raises(RuntimeError, match="empty_output"):
        await provider.invoke(request)

    assert provider.requests == [request]
    assert len(update_calls) == 1
    update_call = update_calls[0]
    assert update_call.metadata is not None
    raw = update_call.metadata.get("raw")
    assert isinstance(raw, Mapping)
    assert raw["request"] == provider_module._request_snapshot(request)
    assert raw["response_payload"] == response.payload
    assert raw["response_metadata"] == {"source": "stub", "raw_response": {"response_id": "provider-raw"}}
    assert raw["provider_response"] == {"response_id": "provider-raw"}


async def test_invoke_with_none_generation_still_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    response = _response(metadata={"source": "stub"})
    provider = _StubProvider(response=response)
    scope = _Scope(generation=None)
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(
            _UpdateCall(
                generation=generation,
                input_payload=input_payload,
                output=output,
                usage=usage,
                metadata=metadata,
            )
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    result = await provider.invoke(request)

    assert result == response
    assert provider.requests == [request]
    assert scope.entered == 1
    assert scope.exited == 1
    assert len(update_calls) == 1
    assert update_calls[0].generation is None


@pytest.mark.parametrize(
    ("provider_label", "expected_retriever_name"),
    (
        ("vertex", "vertex.grounding.search"),
        ("openai", "openai.search.query"),
    ),
)
async def test_invoke_records_retriever_and_tool_child_observations(
    monkeypatch: pytest.MonkeyPatch,
    provider_label: str,
    expected_retriever_name: str,
) -> None:
    request = _request()
    response = LlmResponse(
        id="response-id",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text="ok"),),
                    tool_calls=(
                        LlmMessageToolCall(
                            id="call-1",
                            type="function",
                            name="search_repo",
                            arguments='{"query":"caster"}',
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ),
        usage=LlmUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7, web_search_calls=1),
        metadata={"web_search_queries": ("caster subnet",), "source": "stub"},
        finish_reason="tool_calls",
    )
    provider = _StubProvider(response=response, provider_label=provider_label)
    scope = _Scope(generation=object())
    child_calls: list[dict[str, object]] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        return None

    def fake_record_child_observation(
        *,
        name: str,
        as_type: str,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        child_calls.append(
            {
                "name": name,
                "as_type": as_type,
                "input_payload": input_payload,
                "output": output,
                "metadata": metadata,
            }
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)
    monkeypatch.setattr(provider_module, "record_child_observation_best_effort", fake_record_child_observation)

    await provider.invoke(request)

    assert len(child_calls) == 2
    assert child_calls[0]["as_type"] == "retriever"
    assert child_calls[0]["name"] == expected_retriever_name
    assert child_calls[1]["as_type"] == "tool"
    assert child_calls[1]["name"] == "search_repo"
    assert all(call["as_type"] != "agent" for call in child_calls)


async def test_invoke_preserves_provider_facing_extra_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(
        internal_metadata={"use_case": "tool_runtime_invoker"},
        extra={"web_search_options": {"mode": "auto"}},
    )
    response = _response(metadata={"source": "stub"})
    provider = _StubProvider(response=response)
    scope = _Scope(generation=object())

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        return None

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    await provider.invoke(request)

    assert provider.requests == [request]
    assert provider.requests[0].extra == {"web_search_options": {"mode": "auto"}}
    assert provider.requests[0].internal_metadata == {"use_case": "tool_runtime_invoker"}


async def test_invoke_vertex_gemini_reasoning_marks_include_thoughts_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        provider="vertex",
        model="gemini-2.5-pro",
        reasoning_effort="high",
    )
    response = _response(metadata={"source": "stub"})
    provider = _StubProvider(response=response, provider_label="vertex")
    scope = _Scope(generation=object())
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(
            _UpdateCall(
                generation=generation,
                input_payload=input_payload,
                output=output,
                usage=usage,
                metadata=metadata,
            )
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    await provider.invoke(request)

    assert len(update_calls) == 1
    update_call = update_calls[0]
    assert update_call.metadata is not None
    reasoning = update_call.metadata.get("reasoning")
    assert isinstance(reasoning, Mapping)
    assert reasoning["include_thoughts_requested"] is True
    assert reasoning["reasoning_effort"] == "high"


async def test_invoke_vertex_claude_reasoning_does_not_mark_include_thoughts_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        provider="vertex",
        model="publishers/anthropic/models/claude-3-7-sonnet",
        reasoning_effort="high",
    )
    response = _response(
        metadata={"source": "stub"},
        usage=LlmUsage(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            reasoning_tokens=4,
        ),
    )
    provider = _StubProvider(response=response, provider_label="vertex")
    scope = _Scope(generation=object())
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(
            _UpdateCall(
                generation=generation,
                input_payload=input_payload,
                output=output,
                usage=usage,
                metadata=metadata,
            )
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    await provider.invoke(request)

    assert len(update_calls) == 1
    update_call = update_calls[0]
    assert update_call.metadata is not None
    reasoning = update_call.metadata.get("reasoning")
    assert isinstance(reasoning, Mapping)
    assert reasoning["include_thoughts_requested"] is False
    assert reasoning["reasoning_tokens"] == 4


async def test_invoke_vertex_maas_openai_reasoning_does_not_mark_include_thoughts_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-20b-maas",
        reasoning_effort="high",
    )
    response = _response(
        metadata={"source": "stub"},
        usage=LlmUsage(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            reasoning_tokens=3,
        ),
    )
    provider = _StubProvider(response=response, provider_label="vertex")
    scope = _Scope(generation=object())
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        return scope

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(
            _UpdateCall(
                generation=generation,
                input_payload=input_payload,
                output=output,
                usage=usage,
                metadata=metadata,
            )
        )

    monkeypatch.setattr(provider_module, "start_llm_generation", fake_start)
    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)

    await provider.invoke(request)

    assert len(update_calls) == 1
    update_call = update_calls[0]
    assert update_call.metadata is not None
    reasoning = update_call.metadata.get("reasoning")
    assert isinstance(reasoning, Mapping)
    assert reasoning["include_thoughts_requested"] is False
    assert reasoning["reasoning_effort"] == "high"
    assert reasoning["reasoning_tokens"] == 3
