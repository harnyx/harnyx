from __future__ import annotations

from collections.abc import Mapping

import pytest

import harnyx_commons.llm.provider as provider_module
from harnyx_commons.llm.provider import BaseLlmProvider
from harnyx_commons.llm.schema import (
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

    with pytest.raises(RuntimeError, match="raw-provider-exception-sentinel"):
        await provider.invoke(request)

    assert start_kwargs[0]["record_exception"] is False
    assert start_kwargs[0]["set_status_on_exception"] is False


async def test_metadata_only_invoke_keeps_private_payloads_out_of_logs(caplog: pytest.LogCaptureFixture) -> None:
    request = _request(
        internal_metadata={"private": "metadata-sentinel"},
        extra={"private": "extra-sentinel"},
        include_payloads_in_observability=False,
    )
    response = _response(metadata={"raw_response": {"private": "response-sentinel"}})
    caplog.set_level("DEBUG", logger="harnyx_commons.llm.calls")
    assert await _StubProvider(response=response).invoke(request) == response
    serialized_logs = repr([record.__dict__ for record in caplog.records])
    for sentinel in ("metadata-sentinel", "extra-sentinel", "response-sentinel"):
        assert sentinel not in serialized_logs


def test_usage_observability_keeps_unavailable_reasoning_tokens_out_of_usage_details() -> None:
    usage = LlmUsage(prompt_tokens=3, completion_tokens=6, reasoning_tokens=None, total_tokens=9)

    assert "reasoning" not in provider_module._usage_snapshot(usage)
