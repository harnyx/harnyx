"""Typed Bedrock request/stream codec helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn

from botocore.exceptions import ClientError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)
from pydantic.alias_generators import to_camel

from harnyx_commons.llm.provider_types import normalize_reasoning_effort
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmInputTextPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

_STRICT_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    strict=True,
    populate_by_name=True,
    alias_generator=to_camel,
)

_IGNORE_EXTRA_MODEL_CONFIG = ConfigDict(
    extra="ignore",
    strict=True,
    populate_by_name=True,
    alias_generator=to_camel,
)


class BedrockTextBlock(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    text: str


class BedrockRequestMessage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    role: str
    content: list[BedrockTextBlock]


class BedrockInferenceConfig(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    max_tokens: int | None = None
    temperature: float | None = None


class BedrockJsonSchemaConfig(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    json_schema: str = Field(serialization_alias="schema", validation_alias="schema")
    name: str
    description: str | None = None


class BedrockStructuredOutputConfig(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    text_format: dict[str, object]

    @classmethod
    def from_request(cls, request: LlmRequest) -> BedrockStructuredOutputConfig | None:
        if request.output_mode != "structured":
            return None
        schema_type = request.output_schema
        if schema_type is None:
            raise ValueError("Bedrock structured output requires output_schema")
        schema = schema_type.model_json_schema()
        json_schema = BedrockJsonSchemaConfig(
            json_schema=json.dumps(schema),
            name=_schema_name(schema_type),
            description=_schema_description(schema),
        )
        return cls(
            text_format={
                "type": "json_schema",
                "structure": {
                    "jsonSchema": json_schema.model_dump(mode="python", by_alias=True, exclude_none=True),
                },
            }
        )


class BedrockConverseStreamRequest(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    model_id: str
    messages: list[BedrockRequestMessage]
    system: list[BedrockTextBlock] | None = None
    inference_config: BedrockInferenceConfig | None = None
    additional_model_request_fields: dict[str, object] | None = None
    output_config: BedrockStructuredOutputConfig | None = None

    @classmethod
    def from_llm_request(cls, request: LlmRequest) -> BedrockConverseStreamRequest:
        messages, system = _serialize_messages(request.messages)
        return cls(
            model_id=request.model,
            messages=messages,
            system=system or None,
            inference_config=_build_inference_config(request),
            additional_model_request_fields=_build_additional_model_request_fields(request),
            output_config=BedrockStructuredOutputConfig.from_request(request),
        )

    @model_serializer(mode="wrap")
    def _serialize_request(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload = handler(self)
        if payload.get("inferenceConfig") == {}:
            payload.pop("inferenceConfig", None)
        return payload

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(mode="python", by_alias=True, exclude_none=True)


class _BedrockReasoningContentPayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    text: str | None = None
    redacted_content: bytes | None = None
    signature: str | None = None


class _BedrockMessageStartPayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    role: str


class _BedrockStartPayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    tool_use: object | None = None
    tool_result: object | None = None
    image: object | None = None


class _BedrockContentBlockStartEventPayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    content_block_index: int
    start: _BedrockStartPayload | None = None


class _BedrockContentBlockStopEventPayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    content_block_index: int


class _BedrockMessageStopPayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    stop_reason: str | None = None
    additional_model_response_fields: dict[str, object] | None = None


class _BedrockUsagePayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    cache_details: list[dict[str, object]] | None = None


class BedrockMetadataPayload(BaseModel):
    model_config = _IGNORE_EXTRA_MODEL_CONFIG

    usage: _BedrockUsagePayload | None = None
    metrics: dict[str, object] | None = None
    trace: dict[str, object] | None = None
    performance_config: dict[str, object] | None = None
    service_tier: dict[str, object] | None = None


class _BedrockErrorPayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    message: str | None = None
    original_message: str | None = None
    original_status_code: int | None = None


class TextDelta(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    text: str

    def apply_to(self, accumulator: BedrockStreamAccumulator, *, content_block_index: int) -> bool:
        accumulator.append_text(content_block_index, self.text)
        return True


class ReasoningDelta(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    reasoning_content: _BedrockReasoningContentPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator, *, content_block_index: int) -> bool:
        return accumulator.append_reasoning(self.reasoning_content.text)


class CitationDelta(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    citation: dict[str, object]

    def apply_to(self, accumulator: BedrockStreamAccumulator, *, content_block_index: int) -> bool:
        accumulator.append_citation(self.citation)
        return False


class ToolUseDelta(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    tool_use: object

    def apply_to(self, accumulator: BedrockStreamAccumulator, *, content_block_index: int) -> bool:
        raise ValueError("Bedrock first cut does not support tool use deltas")


class ToolResultDelta(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    tool_result: object

    def apply_to(self, accumulator: BedrockStreamAccumulator, *, content_block_index: int) -> bool:
        raise ValueError("Bedrock first cut does not support tool result deltas")


class ImageDelta(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    image: object

    def apply_to(self, accumulator: BedrockStreamAccumulator, *, content_block_index: int) -> bool:
        raise ValueError("Bedrock first cut does not support image deltas")


BedrockDelta = TextDelta | ReasoningDelta | CitationDelta | ToolUseDelta | ToolResultDelta | ImageDelta


class BedrockContentBlockDeltaEventPayload(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    content_block_index: int
    delta: BedrockDelta

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        return self.delta.apply_to(accumulator, content_block_index=self.content_block_index)


class MessageStartEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    message_start: _BedrockMessageStartPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        return accumulator.apply_message_start(self.message_start)


class ContentBlockStartEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    content_block_start: _BedrockContentBlockStartEventPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        return accumulator.apply_content_block_start(self.content_block_start)


class ContentBlockDeltaEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    content_block_delta: BedrockContentBlockDeltaEventPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        return self.content_block_delta.apply_to(accumulator)


class ContentBlockStopEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    content_block_stop: _BedrockContentBlockStopEventPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        return False


class MessageStopEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    message_stop: _BedrockMessageStopPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        return accumulator.apply_message_stop(self.message_stop)


class MetadataEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    metadata: BedrockMetadataPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        return accumulator.apply_metadata(self.metadata)


class ValidationExceptionEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    validation_exception: _BedrockErrorPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        accumulator.raise_stream_error(code="ValidationException", http_status=400, error=self.validation_exception)


class ThrottlingExceptionEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    throttling_exception: _BedrockErrorPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        accumulator.raise_stream_error(code="ThrottlingException", http_status=429, error=self.throttling_exception)


class ServiceUnavailableExceptionEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    service_unavailable_exception: _BedrockErrorPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        accumulator.raise_stream_error(
            code="ServiceUnavailableException",
            http_status=503,
            error=self.service_unavailable_exception,
        )


class ModelStreamErrorExceptionEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    model_stream_error_exception: _BedrockErrorPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        accumulator.raise_stream_error(
            code="ModelStreamErrorException",
            http_status=424,
            error=self.model_stream_error_exception,
        )


class InternalServerExceptionEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    internal_server_exception: _BedrockErrorPayload

    def apply_to(self, accumulator: BedrockStreamAccumulator) -> bool:
        accumulator.raise_stream_error(
            code="InternalServerException",
            http_status=500,
            error=self.internal_server_exception,
        )


BedrockStreamEvent = (
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageStopEvent
    | MetadataEvent
    | ValidationExceptionEvent
    | ThrottlingExceptionEvent
    | ServiceUnavailableExceptionEvent
    | ModelStreamErrorExceptionEvent
    | InternalServerExceptionEvent
)


BEDROCK_STREAM_EVENT_ADAPTER = TypeAdapter(BedrockStreamEvent)


@dataclass(slots=True)
class BedrockStreamAccumulator:
    text_parts: dict[int, list[str]] = field(default_factory=dict)
    reasoning_parts: list[str] = field(default_factory=list)
    citations: list[dict[str, object]] = field(default_factory=list)
    response_events: list[dict[str, object]] = field(default_factory=list)
    response_metadata: Mapping[str, Any] | None = None
    metadata_event: BedrockMetadataPayload | None = None
    finish_reason: str | None = None
    additional_model_response_fields: dict[str, object] | None = None
    response_role: str | None = None

    def set_response_metadata(self, response_metadata: Mapping[str, Any] | None) -> None:
        self.response_metadata = response_metadata

    def apply(self, event: BedrockStreamEvent, *, raw_event: Mapping[str, object]) -> bool:
        self.response_events.append(dict(raw_event))
        return event.apply_to(self)

    def response_id(self) -> str:
        if self.response_metadata is None:
            return ""
        request_id = self.response_metadata.get("RequestId")
        return "" if request_id is None else str(request_id)

    def to_llm_response(self) -> LlmResponse:
        choice = LlmChoice(
            index=0,
            message=LlmChoiceMessage(
                role="assistant",
                content=self._text_content(),
                tool_calls=None,
                reasoning=self._reasoning_text(),
            ),
            finish_reason=self.finish_reason,
        )
        metadata: dict[str, object] = {
            "raw_response": {
                "events": self.response_events,
                "response_metadata": self.response_metadata,
            }
        }
        if self.additional_model_response_fields is not None:
            metadata["additional_model_response_fields"] = self.additional_model_response_fields
        if self.citations:
            metadata["citations"] = tuple(self.citations)
        if self.metadata_event is not None:
            metadata["bedrock_metadata"] = self.metadata_event.model_dump(
                mode="python",
                by_alias=True,
                exclude_none=True,
            )
        return LlmResponse(
            id=self.response_id(),
            choices=(choice,),
            usage=_extract_usage(self.metadata_event),
            metadata=metadata,
            finish_reason=self.finish_reason,
        )

    def apply_message_start(self, payload: _BedrockMessageStartPayload) -> bool:
        role = payload.role
        if role != "assistant":
            raise RuntimeError(f"Bedrock returned unexpected response role: {role!r}")
        self.response_role = role
        return False

    def apply_content_block_start(self, payload: _BedrockContentBlockStartEventPayload) -> bool:
        start = payload.start
        if start is None:
            return False
        if start.tool_use is not None:
            raise ValueError("Bedrock first cut does not support tool use responses")
        if start.tool_result is not None:
            raise ValueError("Bedrock first cut does not support tool result responses")
        if start.image is not None:
            raise ValueError("Bedrock first cut does not support image output")
        return False

    def apply_message_stop(self, payload: _BedrockMessageStopPayload) -> bool:
        self.finish_reason = payload.stop_reason
        self.additional_model_response_fields = payload.additional_model_response_fields
        return False

    def apply_metadata(self, payload: BedrockMetadataPayload) -> bool:
        self.metadata_event = payload
        return False

    def append_text(self, content_block_index: int, text: str) -> None:
        self.text_parts.setdefault(content_block_index, []).append(text)

    def append_reasoning(self, text: str | None) -> bool:
        if isinstance(text, str) and text:
            self.reasoning_parts.append(text)
            return True
        return False

    def append_citation(self, citation: Mapping[str, object]) -> None:
        self.citations.append(dict(citation))

    def _text_content(self) -> tuple[LlmMessageContentPart, ...]:
        parts: list[LlmMessageContentPart] = []
        for index in sorted(self.text_parts):
            text = "".join(self.text_parts[index]).strip()
            if text:
                parts.append(LlmMessageContentPart(type="text", text=text))
        return tuple(parts)

    def _reasoning_text(self) -> str | None:
        combined = "".join(self.reasoning_parts).strip()
        return combined or None

    def raise_stream_error(self, *, code: str, http_status: int, error: _BedrockErrorPayload) -> NoReturn:
        message = str(error.message or error.original_message or code)
        raise ClientError(
            error_response={
                "Error": {
                    "Code": code,
                    "Message": message,
                },
                "ResponseMetadata": {
                    "HTTPStatusCode": error.original_status_code or http_status,
                },
            },
            operation_name="ConverseStream",
        )


def _serialize_messages(
    messages: Sequence[LlmMessage],
) -> tuple[list[BedrockRequestMessage], list[BedrockTextBlock]]:
    serialized_messages: list[BedrockRequestMessage] = []
    system_blocks: list[BedrockTextBlock] = []
    for message in messages:
        serialized_parts = [_serialize_text_part(message=message, part=part) for part in message.content]
        if message.role == "system":
            system_blocks.extend(serialized_parts)
            continue
        if message.role not in {"user", "assistant"}:
            raise ValueError(f"Bedrock first cut does not support message role '{message.role}'")
        serialized_messages.append(BedrockRequestMessage(role=message.role, content=serialized_parts))
    if not serialized_messages:
        raise ValueError("Bedrock requests must include at least one non-system message")
    return serialized_messages, system_blocks


def _serialize_text_part(*, message: LlmMessage, part: object) -> BedrockTextBlock:
    if not isinstance(part, LlmInputTextPart):
        raise ValueError(
            f"Bedrock first cut supports only input_text parts; found {type(part).__name__} in {message.role} message"
        )
    return BedrockTextBlock(text=part.text)


def _build_inference_config(request: LlmRequest) -> BedrockInferenceConfig | None:
    return BedrockInferenceConfig(
        max_tokens=request.max_output_tokens,
        temperature=request.temperature,
    )


def _build_additional_model_request_fields(request: LlmRequest) -> dict[str, object] | None:
    reasoning_effort = normalize_reasoning_effort(request.reasoning_effort)
    if reasoning_effort is None:
        return None
    return {"reasoning_effort": reasoning_effort}


def _schema_name(schema_type: type[BaseModel]) -> str:
    title = (schema_type.model_json_schema().get("title") or schema_type.__name__).strip()
    return title[:64] or schema_type.__name__


def _schema_description(schema: dict[str, object]) -> str | None:
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        return description
    return None


def _extract_usage(metadata_event: BedrockMetadataPayload | None) -> LlmUsage:
    usage = metadata_event.usage if metadata_event is not None else None
    if usage is None:
        return LlmUsage()
    return LlmUsage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        prompt_cached_tokens=usage.cache_read_input_tokens,
    )


__all__ = [
    "BEDROCK_STREAM_EVENT_ADAPTER",
    "BedrockConverseStreamRequest",
    "BedrockStreamAccumulator",
]
