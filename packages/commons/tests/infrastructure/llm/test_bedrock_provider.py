from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace

import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError

from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.providers.bedrock import BedrockLlmProvider
from harnyx_commons.llm.providers.bedrock_codec import (
    BEDROCK_STREAM_EVENT_ADAPTER,
    BedrockStreamAccumulator,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InternalServerExceptionEvent,
    MessageStartEvent,
    MessageStopEvent,
    MetadataEvent,
    ModelStreamErrorExceptionEvent,
    ReasoningDelta,
    ServiceUnavailableExceptionEvent,
    TextDelta,
    ThrottlingExceptionEvent,
    ValidationExceptionEvent,
)
from harnyx_commons.llm.schema import GroundedLlmRequest, LlmMessage, LlmMessageContentPart, LlmRequest, LlmTool

pytestmark = pytest.mark.anyio("asyncio")


class _StructuredAnswer(BaseModel):
    answer: str


class _FakeEventStream:
    def __init__(self, events: Sequence[dict[str, object]]) -> None:
        self._events = tuple(events)

    def __aiter__(self) -> AsyncIterator[dict[str, object]]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[dict[str, object]]:
        for event in self._events:
            yield event


@dataclass
class _FakeClient:
    events: Sequence[dict[str, object]]
    calls: list[dict[str, object]]

    async def converse_stream(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "ResponseMetadata": {
                "RequestId": "request-123",
                "HTTPStatusCode": 200,
            },
            "stream": _FakeEventStream(self.events),
        }


class _FakeClientContext(AbstractAsyncContextManager[_FakeClient]):
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    async def __aenter__(self) -> _FakeClient:
        return self._client

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSession:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client
        self.calls: list[dict[str, object]] = []

    def create_client(self, service_name: str, **kwargs: object) -> _FakeClientContext:
        self.calls.append({"service_name": service_name, **kwargs})
        return _FakeClientContext(self._client)


def _patch_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: Sequence[dict[str, object]],
) -> tuple[_FakeSession, list[dict[str, object]]]:
    captured_calls: list[dict[str, object]] = []
    client = _FakeClient(events=events, calls=captured_calls)
    session = _FakeSession(client)
    monkeypatch.setattr("harnyx_commons.llm.providers.bedrock.get_session", lambda: session)
    return session, captured_calls


def _provider() -> BedrockLlmProvider:
    return BedrockLlmProvider(
        region="us-east-1",
        connect_timeout_seconds=5.0,
        read_timeout_seconds=60.0,
    )


def _base_request(*, output_mode: str = "text", output_schema: type[BaseModel] | None = None) -> LlmRequest:
    return LlmRequest(
        provider="bedrock",
        model="openai.gpt-oss-20b-1:0",
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("You are terse."),),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("What is 7 times 8?"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=128,
        output_mode=output_mode,
        output_schema=output_schema,
        reasoning_effort="high",
    )


async def test_bedrock_provider_maps_stream_response_and_logs_ttft(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Thinking. "}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "56"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}}},
        ),
    )
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    provider = _provider()

    response = await provider.invoke(_base_request())

    assert response.id == "request-123"
    assert response.raw_text == "56"
    assert response.choices[0].message.reasoning == "Thinking."
    assert response.finish_reason == "end_turn"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 18
    assert "ttft_ms" not in dict(response.metadata or {})
    raw_response = dict(response.metadata or {})["raw_response"]
    assert isinstance(raw_response, dict)
    assert len(raw_response["events"]) == 6
    ttft_records = [record for record in caplog.records if record.message == "llm.bedrock.stream.ttft"]
    assert ttft_records
    assert ttft_records[0].__dict__["data"]["ttft_ms"] >= 0


async def test_bedrock_provider_ttft_uses_first_reasoning_delta(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Thinking. "}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "56"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}}},
        ),
    )
    produced_output_events: list[tuple[object, dict[str, object], bool]] = []
    original_apply = BedrockStreamAccumulator.apply

    def _record_apply(self: BedrockStreamAccumulator, event, *, raw_event: dict[str, object]) -> bool:
        result = original_apply(self, event, raw_event=raw_event)
        produced_output_events.append((event, dict(raw_event), result))
        return result

    monkeypatch.setattr("harnyx_commons.llm.providers.bedrock.BedrockStreamAccumulator.apply", _record_apply)
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    provider = _provider()

    await provider.invoke(_base_request())

    ttft_records = [record for record in caplog.records if record.message == "llm.bedrock.stream.ttft"]
    assert ttft_records
    assert any(result for _, _, result in produced_output_events)
    first_output_event, first_output_raw_event, first_output_result = next(
        (event, raw_event, result)
        for event, raw_event, result in produced_output_events
        if result
    )
    assert first_output_result is True
    assert isinstance(first_output_event, ContentBlockDeltaEvent)
    assert isinstance(first_output_event.content_block_delta.delta, ReasoningDelta)
    assert first_output_raw_event == {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"text": "Thinking. "}},
        }
    }


async def test_bedrock_provider_ttft_uses_first_text_delta(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "56"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Thinking. "}}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}}},
        ),
    )
    produced_output_events: list[tuple[object, bool]] = []
    original_apply = BedrockStreamAccumulator.apply

    def _record_apply(self: BedrockStreamAccumulator, event, *, raw_event: dict[str, object]) -> bool:
        result = original_apply(self, event, raw_event=raw_event)
        produced_output_events.append((event, result))
        return result

    monkeypatch.setattr("harnyx_commons.llm.providers.bedrock.BedrockStreamAccumulator.apply", _record_apply)
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    provider = _provider()

    await provider.invoke(_base_request())

    ttft_records = [record for record in caplog.records if record.message == "llm.bedrock.stream.ttft"]
    assert ttft_records
    first_output_event, first_output_result = next(
        (event, result) for event, result in produced_output_events if result
    )
    assert first_output_result is True
    assert isinstance(first_output_event, ContentBlockDeltaEvent)
    assert isinstance(first_output_event.content_block_delta.delta, TextDelta)


async def test_bedrock_provider_builds_structured_output_config(monkeypatch: pytest.MonkeyPatch) -> None:
    session, client_calls = _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": '{"answer":"pong"}'}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 4, "totalTokens": 9}}},
        ),
    )
    provider = _provider()

    response = await provider.invoke(_base_request(output_mode="structured", output_schema=_StructuredAnswer))

    assert response.raw_text == '{"answer":"pong"}'
    assert session.calls[0]["service_name"] == "bedrock-runtime"
    assert session.calls[0]["config"].retries["total_max_attempts"] == 1
    request_payload = client_calls[0]
    assert request_payload["modelId"] == "openai.gpt-oss-20b-1:0"
    assert request_payload["additionalModelRequestFields"] == {"reasoning_effort": "high"}
    output_config = request_payload["outputConfig"]
    assert output_config["textFormat"]["type"] == "json_schema"
    json_schema = output_config["textFormat"]["structure"]["jsonSchema"]
    assert json_schema["name"] == "_StructuredAnswer"
    assert "description" not in json_schema


async def test_bedrock_provider_preserves_explicit_zero_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "56"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}}},
        ),
    )
    provider = _provider()
    request = replace(_base_request(), timeout_seconds=0.0)

    await provider.invoke(request)

    assert session.calls[0]["config"].read_timeout == 0.0


async def test_bedrock_provider_omits_empty_inference_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _, client_calls = _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "56"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}}},
        ),
    )
    provider = _provider()
    request = replace(_base_request(), max_output_tokens=None, temperature=None)

    await provider.invoke(request)

    assert "inferenceConfig" not in client_calls[0]


def test_bedrock_stream_event_adapter_parses_message_start_variant() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python({"messageStart": {"role": "assistant"}})

    assert isinstance(event, MessageStartEvent)
    assert event.message_start.role == "assistant"


def test_bedrock_stream_event_adapter_parses_content_block_stop_variant() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python({"contentBlockStop": {"contentBlockIndex": 2}})

    assert isinstance(event, ContentBlockStopEvent)
    assert event.content_block_stop.content_block_index == 2


def test_bedrock_stream_event_adapter_parses_reasoning_delta_variant() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python(
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Thinking. "}}}}
    )

    assert isinstance(event, ContentBlockDeltaEvent)
    assert event.content_block_delta.content_block_index == 0
    assert isinstance(event.content_block_delta.delta, ReasoningDelta)
    assert event.content_block_delta.delta.reasoning_content.text == "Thinking. "


def test_bedrock_stream_event_adapter_parses_reasoning_content_signature() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "Thinking. ", "signature": "sig-123"}},
            }
        }
    )

    assert isinstance(event, ContentBlockDeltaEvent)
    assert isinstance(event.content_block_delta.delta, ReasoningDelta)
    assert event.content_block_delta.delta.reasoning_content.signature == "sig-123"


def test_bedrock_stream_event_adapter_rejects_invalid_multi_key_shape() -> None:
    with pytest.raises(ValidationError):
        BEDROCK_STREAM_EVENT_ADAPTER.validate_python(
            {
                "messageStart": {"role": "assistant"},
                "metadata": {"usage": {"totalTokens": 1}},
            }
        )


def test_bedrock_stream_event_adapter_rejects_unknown_message_start_keys() -> None:
    with pytest.raises(ValidationError):
        BEDROCK_STREAM_EVENT_ADAPTER.validate_python(
            {"messageStart": {"role": "assistant", "unexpected": "x"}}
        )


def test_bedrock_stream_event_adapter_rejects_unknown_error_payload_keys() -> None:
    with pytest.raises(ValidationError):
        BEDROCK_STREAM_EVENT_ADAPTER.validate_python(
            {"validationException": {"message": "schema mismatch", "unexpected": "x"}}
        )


def test_bedrock_stream_event_dispatch_delegates_message_start_behavior() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python({"messageStart": {"role": "assistant"}})
    accumulator = BedrockStreamAccumulator()

    result = accumulator.apply(event, raw_event={"messageStart": {"role": "assistant"}})

    assert isinstance(event, MessageStartEvent)
    assert result is False
    assert accumulator.response_role == "assistant"


def test_bedrock_stream_event_dispatch_delegates_content_block_start_behavior() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python({"contentBlockStart": {"contentBlockIndex": 0, "start": {}}})
    accumulator = BedrockStreamAccumulator()

    result = accumulator.apply(event, raw_event={"contentBlockStart": {"contentBlockIndex": 0, "start": {}}})

    assert isinstance(event, ContentBlockStartEvent)
    assert result is False


def test_bedrock_stream_event_dispatch_delegates_message_stop_behavior() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python(
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": {"foo": "bar"}}}
    )
    accumulator = BedrockStreamAccumulator()

    result = accumulator.apply(
        event,
        raw_event={"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": {"foo": "bar"}}},
    )

    assert isinstance(event, MessageStopEvent)
    assert result is False
    assert accumulator.finish_reason == "end_turn"
    assert accumulator.additional_model_response_fields == {"foo": "bar"}


def test_bedrock_stream_event_dispatch_delegates_metadata_behavior() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python({"metadata": {"usage": {"totalTokens": 1}}})
    accumulator = BedrockStreamAccumulator()

    result = accumulator.apply(event, raw_event={"metadata": {"usage": {"totalTokens": 1}}})

    assert isinstance(event, MetadataEvent)
    assert result is False
    assert accumulator.metadata_event is not None
    assert accumulator.metadata_event.usage is not None
    assert accumulator.metadata_event.usage.total_tokens == 1


def test_bedrock_stream_event_adapter_preserves_documented_metadata_fields() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python(
        {
            "metadata": {
                "usage": {"totalTokens": 1, "cacheWriteInputTokens": 2},
                "metrics": {"latencyMs": 123},
                "trace": {"guardrail": {}},
                "performanceConfig": {"latency": "optimized"},
                "serviceTier": {"type": "priority"},
                "ignoredFutureField": {"still": "allowed"},
            }
        }
    )

    assert isinstance(event, MetadataEvent)
    assert event.metadata.model_dump(mode="python", by_alias=True, exclude_none=True) == {
        "usage": {"totalTokens": 1, "cacheWriteInputTokens": 2},
        "metrics": {"latencyMs": 123},
        "trace": {"guardrail": {}},
        "performanceConfig": {"latency": "optimized"},
        "serviceTier": {"type": "priority"},
    }


def test_bedrock_stream_event_adapter_keeps_documented_start_tool_shape_opaque_until_rejection() -> None:
    raw_event = {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {"toolUse": {"toolUseId": "tool-1", "name": "lookup", "type": "server_tool_use"}},
        }
    }
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python(raw_event)
    accumulator = BedrockStreamAccumulator()

    with pytest.raises(ValueError, match="does not support tool use responses"):
        accumulator.apply(event, raw_event=raw_event)


def test_bedrock_stream_event_adapter_keeps_opaque_additional_model_response_fields() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python(
        {
            "messageStop": {
                "stopReason": "end_turn",
                "additionalModelResponseFields": {"foo": {"bar": [1]}},
            }
        }
    )

    assert isinstance(event, MessageStopEvent)
    assert event.message_stop.additional_model_response_fields == {"foo": {"bar": [1]}}


def test_bedrock_stream_event_dispatch_delegates_stream_error_behavior() -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python({"validationException": {"message": "schema mismatch"}})
    accumulator = BedrockStreamAccumulator()

    with pytest.raises(ClientError, match="schema mismatch"):
        accumulator.apply(event, raw_event={"validationException": {"message": "schema mismatch"}})


@pytest.mark.parametrize(
    ("raw_event", "expected_type"),
    [
        ({"validationException": {"message": "schema mismatch"}}, ValidationExceptionEvent),
        ({"throttlingException": {"message": "slow down"}}, ThrottlingExceptionEvent),
        ({"serviceUnavailableException": {"message": "outage"}}, ServiceUnavailableExceptionEvent),
        ({"modelStreamErrorException": {"message": "stream exploded"}}, ModelStreamErrorExceptionEvent),
        ({"internalServerException": {"message": "internal"}} , InternalServerExceptionEvent),
    ],
)
def test_bedrock_stream_event_adapter_parses_stream_error_variants(
    raw_event: dict[str, object],
    expected_type: type[object],
) -> None:
    event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python(raw_event)

    assert isinstance(event, expected_type)


@pytest.mark.parametrize(
    ("raw_event", "expected_reason"),
    [
        (
            {"validationException": {"message": "schema mismatch"}},
            "client_error:ValidationException:400:schema mismatch",
        ),
        (
            {"throttlingException": {"message": "slow down"}},
            "client_error:ThrottlingException:429:slow down",
        ),
        (
            {"serviceUnavailableException": {"message": "outage"}},
            "client_error:ServiceUnavailableException:503:outage",
        ),
        (
            {"modelStreamErrorException": {"message": "stream exploded"}},
            "client_error:ModelStreamErrorException:424:stream exploded",
        ),
        (
            {"internalServerException": {"message": "internal"}},
            "client_error:InternalServerException:500:internal",
        ),
    ],
)
async def test_bedrock_provider_preserves_all_stream_error_mappings(
    monkeypatch: pytest.MonkeyPatch,
    raw_event: dict[str, object],
    expected_reason: str,
) -> None:
    _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            raw_event,
        ),
    )
    provider = _provider()

    with pytest.raises(LlmRetryExhaustedError, match=expected_reason):
        await provider.invoke(_base_request())


@pytest.mark.parametrize(
    ("delta", "message"),
    [
        (
            {"toolUse": {"name": "lookup", "toolUseId": "tool-1", "input": {}}},
            "does not support tool use deltas",
        ),
        (
            {"toolResult": [{"toolUseId": "tool-1", "status": "success", "content": []}]},
            "does not support tool result deltas",
        ),
        (
            {"image": {"format": "png", "source": {"bytes": "abcd"}}},
            "does not support image deltas",
        ),
    ],
)
async def test_bedrock_provider_rejects_unsupported_delta_variants(
    monkeypatch: pytest.MonkeyPatch,
    delta: dict[str, object],
    message: str,
) -> None:
    _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": delta}},
        ),
    )
    provider = _provider()

    with pytest.raises(LlmRetryExhaustedError, match=message):
        await provider.invoke(_base_request())


@pytest.mark.parametrize(
    ("invalid_request", "message"),
    [
        (
            LlmRequest(
                provider="bedrock",
                model="openai/gpt-oss-20b-TEE",
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("hello"),)),),
                temperature=None,
                max_output_tokens=None,
                output_mode="text",
            ),
            "unsupported Bedrock model",
        ),
        (
            LlmRequest(
                provider="bedrock",
                model="moonshotai/Kimi-K2.5-TEE",
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("hello"),)),),
                temperature=None,
                max_output_tokens=None,
                output_mode="text",
            ),
            "unsupported Bedrock model",
        ),
        (
            LlmRequest(
                provider="bedrock",
                model="openai.gpt-oss-20b-1:0",
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("hello"),)),),
                temperature=None,
                max_output_tokens=None,
                output_mode="json_object",
            ),
            "does not support json_object",
        ),
        (
            LlmRequest(
                provider="bedrock",
                model="openai.gpt-oss-20b-1:0",
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("hello"),)),),
                temperature=None,
                max_output_tokens=None,
                output_mode="text",
                tools=(LlmTool(type="function", function={"name": "lookup"}),),
            ),
            "does not support tool definitions",
        ),
        (
            GroundedLlmRequest(
                provider="vertex",
                model="gemini-2.5-flash",
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("hello"),)),),
                temperature=None,
                max_output_tokens=None,
            ),
            "supports only ungrounded",
        ),
    ],
)
async def test_bedrock_provider_rejects_unsupported_requests(
    monkeypatch: pytest.MonkeyPatch,
    invalid_request: object,
    message: str,
) -> None:
    _patch_session(monkeypatch, events=())
    provider = _provider()

    with pytest.raises(ValueError, match=message):
        await provider.invoke(invalid_request)


async def test_bedrock_provider_accepts_native_kimi_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(
        monkeypatch,
        events=(
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "56"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}}},
        ),
    )
    provider = _provider()
    request = replace(_base_request(), model="moonshotai.kimi-k2.5")

    response = await provider.invoke(request)

    assert response.raw_text == "56"


def test_bedrock_provider_classifies_client_errors() -> None:
    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ThrottlingException",
                "Message": "slow down",
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 429,
            },
        },
        operation_name="ConverseStream",
    )

    retryable, reason = BedrockLlmProvider._classify_exception(exc)

    assert retryable is True
    assert reason == "client_error:ThrottlingException:429:slow down"
