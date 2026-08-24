from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType

import pytest

import harnyx_commons.llm.provider as provider_module
from harnyx_commons.llm.provider import BaseLlmProvider
from harnyx_commons.llm.schema import (
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
from harnyx_commons.observability import langfuse

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


class _TraceAttributeScope:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
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

        async def _call(_: AbstractLlmRequest) -> LlmResponse:
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
    use_case: str | None = None,
    internal_metadata: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
    include_payloads_in_observability: bool = True,
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
        use_case=use_case,
        internal_metadata=internal_metadata,
        extra=extra,
        include_payloads_in_observability=include_payloads_in_observability,
    )


def _response(
    *,
    metadata: Mapping[str, object] | None = None,
    usage: LlmUsage | None = None,
    reasoning: str | None = None,
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
                    reasoning=reasoning,
                ),
                finish_reason="stop",
            ),
        ),
        usage=usage or LlmUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        metadata=metadata,
        finish_reason="stop",
    )


async def test_metadata_only_invoke_exports_usage_without_request_or_response_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = _request(
        internal_metadata={"private": "metadata-sentinel"},
        extra={"private": "extra-sentinel"},
        include_payloads_in_observability=False,
    )
    response = _response(
        metadata={"raw_response": {"private": "response-sentinel"}},
    )
    provider = _StubProvider(response=response)
    scope = _Scope(generation=object())
    update_calls: list[_UpdateCall] = []

    monkeypatch.setattr(provider_module, "start_llm_generation", lambda **_: scope)

    def fake_update(
        generation: object | None,
        *,
        input_payload: object | None = None,
        output: object | None = None,
        usage: LlmUsage | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        update_calls.append(_UpdateCall(generation, input_payload, output, usage, metadata))

    monkeypatch.setattr(provider_module, "update_generation_best_effort", fake_update)
    caplog.set_level("DEBUG", logger="harnyx_commons.llm.calls")

    assert await provider.invoke(request) == response

    assert len(update_calls) == 1
    assert update_calls[0].output is None
    assert update_calls[0].usage == response.usage
    assert update_calls[0].metadata is not None
    assert "raw" not in update_calls[0].metadata
    assert "response_metadata" not in update_calls[0].metadata
    serialized_logs = repr([record.__dict__ for record in caplog.records])
    for sentinel in ("metadata-sentinel", "extra-sentinel", "response-sentinel"):
        assert sentinel not in serialized_logs


async def test_metadata_only_invoke_disables_automatic_otel_exception_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(include_payloads_in_observability=False)
    provider = _StubProvider(error=RuntimeError("raw-provider-exception-sentinel"))
    start_kwargs: list[dict[str, object]] = []

    class Span:
        def set_attributes(self, attributes: Mapping[str, object]) -> None:
            del attributes

        def set_attribute(self, key: str, value: object) -> None:
            del key, value

    class SpanScope:
        def __enter__(self) -> Span:
            return Span()

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

    class Tracer:
        def start_as_current_span(self, name: str, **kwargs: object) -> SpanScope:
            start_kwargs.append({"name": name, **kwargs})
            return SpanScope()

    monkeypatch.setattr(provider_module.trace, "get_tracer", lambda _: Tracer())
    monkeypatch.setattr(provider_module, "start_llm_generation", lambda **_: _Scope(generation=None))

    with pytest.raises(RuntimeError, match="raw-provider-exception-sentinel"):
        await provider.invoke(request)

    assert start_kwargs[0]["record_exception"] is False
    assert start_kwargs[0]["set_status_on_exception"] is False


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
        trace_name: str | None = None,
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
                            arguments='{"query":"harnyx"}',
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ),
        usage=LlmUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7, web_search_calls=1),
        metadata={"web_search_queries": ("harnyx subnet",), "source": "stub"},
        finish_reason="tool_calls",
    )
    provider = _StubProvider(response=response, provider_label=provider_label)
    scope = _Scope(generation=object())
    child_calls: list[dict[str, object]] = []

    def fake_start(
        *,
        provider_label: str,
        request: AbstractLlmRequest,
        trace_name: str | None = None,
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


def test_usage_observability_keeps_unavailable_reasoning_tokens_out_of_usage_details() -> None:
    usage = LlmUsage(prompt_tokens=3, completion_tokens=6, reasoning_tokens=None, total_tokens=9)

    provider_usage = provider_module._usage_snapshot(usage)
    langfuse_usage = langfuse._usage_details(usage)

    assert "reasoning" not in provider_usage
    assert "reasoning" not in langfuse_usage
