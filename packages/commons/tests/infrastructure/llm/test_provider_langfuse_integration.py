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
    ) -> None:
        super().__init__(provider_label="openai")
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


def _request() -> LlmRequest:
    return LlmRequest(
        provider="openai",
        model="gpt-5-mini",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
    )


def _response(*, metadata: Mapping[str, object] | None = None) -> LlmResponse:
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
        usage=LlmUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        metadata=metadata,
        finish_reason="stop",
    )


async def test_invoke_success_updates_generation_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    response = _response(metadata={"source": "stub"})
    provider = _StubProvider(response=response)
    scope = _Scope(generation=object())
    start_calls: list[dict[str, object]] = []
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        trace_id: str | None,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> _Scope:
        start_calls.append(
            {
                "trace_id": trace_id,
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
    assert update_call.output == {"raw_text": "ok", "payload": response.payload}
    assert update_call.usage == response.usage
    assert update_call.metadata is not None
    assert update_call.metadata["provider"] == "openai"
    assert update_call.metadata["finish_reason"] == "stop"
    assert update_call.metadata["response_metadata"] == {"source": "stub"}

    elapsed_ms = update_call.metadata["elapsed_ms"]
    wait_ms = update_call.metadata["wait_ms"]
    assert isinstance(elapsed_ms, float)
    assert isinstance(wait_ms, float)


async def test_invoke_error_updates_generation_error_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    provider = _StubProvider(error=ValueError("invoke failed"))
    scope = _Scope(generation=object())
    update_calls: list[_UpdateCall] = []

    def fake_start(
        *,
        trace_id: str | None,
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
        trace_id: str | None,
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
