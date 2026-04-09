"""LLM adapter backed by the Chutes HTTP API."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harnyx_commons.llm.provider import BaseLlmProvider
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmTool,
    LlmUsage,
)

logger = logging.getLogger(__name__)

_CHUTES_EMBEDDING_BASE_URL_BY_MODEL = {
    "Qwen/Qwen3-Embedding-0.6B": "https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai",
}


class ChutesLlmProvider(BaseLlmProvider):
    """Wraps the Chutes chat completions endpoint as an LLM provider."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        auth_header: str = "Authorization",
        max_concurrent: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Chutes API key must be provided for LLM usage")
        super().__init__(provider_label="chutes", max_concurrent=max_concurrent)
        normalized_base = base_url.rstrip("/")
        self._owns_client = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            base_url=normalized_base,
            timeout=timeout,
        )
        self._api_key = api_key
        self._auth_header = auth_header

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        headers = self._auth_headers()

        return await self._call_with_retry(
            request,
            call_coro=lambda current_request: self._request_chutes(
                self._build_payload(current_request),
                headers,
                timeout_seconds=current_request.timeout_seconds,
            ),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
        )

    def _build_payload(self, request: AbstractLlmRequest) -> dict[str, Any]:
        if request.grounded:
            raise ValueError("grounded mode is not supported for chutes provider")
        payload: dict[str, Any] = {
            "provider": request.provider or "chutes",
            "model": request.model,
            "messages": [_serialize_message(message) for message in request.messages],
        }

        optional_fields = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "tools": [_serialize_tool(spec) for spec in request.tools] if request.tools else None,
            "tool_choice": request.tool_choice,
            "include": list(request.include) if request.include else None,
        }
        payload |= {k: v for k, v in optional_fields.items() if v is not None}
        response_format = _build_response_format(request)
        if response_format is not None:
            payload["response_format"] = response_format
        if request.extra:
            payload.update(dict(request.extra))
        return payload

    def _auth_headers(self) -> dict[str, str]:
        return {self._auth_header: f"Bearer {self._api_key}"}

    async def _request_chutes(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        *,
        timeout_seconds: float | None,
    ) -> LlmResponse:
        request_kwargs: dict[str, Any] = {
            "json": payload,
            "headers": headers,
        }
        if timeout_seconds is not None:
            request_kwargs["timeout"] = timeout_seconds
        response = await self._client.post("v1/chat/completions", **request_kwargs)
        response.raise_for_status()
        body = await self._parse_body(response)
        llm_response = self._payload_to_response(body)
        metadata = dict(llm_response.metadata or {})
        metadata.setdefault("raw_response", body.raw)
        return LlmResponse(
            id=llm_response.id,
            choices=llm_response.choices,
            usage=llm_response.usage,
            metadata=metadata,
            finish_reason=llm_response.finish_reason,
        )

    async def _parse_body(self, response: httpx.Response) -> _ChutesResponsePayload:
        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - network dependent
            raise RuntimeError("chutes chat completions returned non-JSON payload") from exc
        return _parse_chutes_response_payload(payload)

    @staticmethod
    def _payload_to_response(payload: _ChutesResponsePayload) -> LlmResponse:
        choices = payload.choices
        usage = payload.usage or LlmUsage()
        response_id = payload.response_id
        return LlmResponse(
            id=response_id,
            choices=choices,
            usage=usage,
        )

    @staticmethod
    def _verify_response(resp: LlmResponse) -> tuple[bool, bool, str | None]:
        if not resp.choices:
            return False, True, "empty_choices"
        if not resp.raw_text and not resp.tool_calls:
            return False, True, "empty_output"
        for call in _iter_tool_calls(resp):
            if not _is_valid_json(call.arguments):
                return False, True, "tool_call_args_invalid_json"
        return True, False, None

    @staticmethod
    def _classify_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code if exc.response else None
            retryable = status is not None and (status == 429 or status >= 500)
            detail = _summarize_response(exc.response) if exc.response is not None else ""
            if detail:
                return retryable, f"http_{status}: {detail}"
            return retryable, f"http_{status}"
        if isinstance(exc, httpx.HTTPError):
            return True, exc.__class__.__name__
        if classify_exception is not None:
            return classify_exception(exc)
        return False, str(exc)

    async def aclose(self) -> None:
        """Close the underlying HTTP client when this provider owns it."""
        if self._owns_client:
            await self._client.aclose()


class _EmbeddingDatum(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    embedding: list[float] = Field(min_length=1)


class _EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    data: list[_EmbeddingDatum] = Field(min_length=1)


@dataclass(slots=True)
class ChutesTextEmbeddingClient:
    model: str
    api_key: str
    base_url: str | None = None
    timeout_seconds: float = 30.0
    dimensions: int | None = None
    client: httpx.AsyncClient | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        normalized_model = self.model.strip()
        if not normalized_model:
            raise ValueError("chutes embedding model must be configured")
        if not self.api_key:
            raise ValueError("Chutes API key must be provided for embeddings")
        resolved_base_url = _normalize_embedding_base_url(self.base_url, normalized_model)
        self.model = normalized_model
        self.base_url = resolved_base_url
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=resolved_base_url,
                timeout=self.timeout_seconds,
            )
            self._owns_client = True
        else:
            self._owns_client = False

    async def embed(self, text: str) -> tuple[float, ...]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("embedding input text must not be empty")
        client = self._require_client()
        response = await client.post(
            "v1/embeddings",
            json=self._request_body(normalized),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        payload = _EmbeddingResponse.model_validate(response.json())
        vector = tuple(float(value) for value in payload.data[0].embedding)
        if self.dimensions is not None and len(vector) != self.dimensions:
            raise RuntimeError(
                f"embedding dimensions mismatch: expected={self.dimensions} actual={len(vector)}"
            )
        return vector

    async def aclose(self) -> None:
        if self._owns_client:
            await self._require_client().aclose()

    def _request_body(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload

    def _require_client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("chutes embedding client is not initialized")
        return self.client


def resolve_chutes_embedding_base_url(model: str) -> str:
    normalized_model = model.strip()
    if not normalized_model:
        raise RuntimeError("chutes embedding model must be configured")
    try:
        return _CHUTES_EMBEDDING_BASE_URL_BY_MODEL[normalized_model]
    except KeyError as exc:
        raise RuntimeError(f"no chutes embedding base_url configured for model: {normalized_model}") from exc


def _normalize_embedding_base_url(base_url: str | None, model: str) -> str:
    if base_url is None:
        return resolve_chutes_embedding_base_url(model)
    normalized_base_url = base_url.rstrip("/")
    if not normalized_base_url:
        raise ValueError("chutes embedding base_url must not be empty")
    return normalized_base_url


def _serialize_tool(spec: LlmTool) -> dict[str, Any]:
    if spec.type == "function" and spec.function is not None:
        return {
            "type": "function",
            "function": dict(spec.function),
        }
    tool_payload: dict[str, Any] = {"type": spec.type}
    if spec.config:
        tool_payload.update(dict(spec.config))
    return tool_payload


def _build_response_format(request: AbstractLlmRequest) -> dict[str, Any] | None:
    match request.output_mode:
        case "text":
            return None
        case "json_object":
            return {"type": "json_object"}
        case "structured":
            schema_type = request.output_schema
            if schema_type is None:
                raise ValueError("structured output requires output_schema")
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_type.__name__,
                    "schema": _json_schema_from_model(schema_type),
                },
            }
        case _:
            raise ValueError(f"unsupported chutes output_mode: {request.output_mode!r}")


def _json_schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    return dict(model.model_json_schema())


def _serialize_message(message: LlmMessage) -> dict[str, Any]:
    fragments: list[str] = []
    tool_results: list[LlmInputToolResultPart] = []
    for part in message.content:
        match part:
            case LlmInputTextPart(text=text):
                fragments.append(text)
            case LlmInputImagePart():
                raise ValueError("chutes provider does not support image content parts")
            case LlmInputToolResultPart() as tool_result:
                tool_results.append(tool_result)
            case _:
                raise ValueError(f"unsupported Chutes request content part type: {part!r}")

    if tool_results:
        if fragments:
            raise ValueError("chutes input_tool_result messages cannot mix text parts")
        if len(tool_results) != 1:
            raise ValueError("chutes input_tool_result messages must include exactly one part")
        tool_result = tool_results[0]
        return {
            "role": "tool",
            "tool_call_id": tool_result.tool_call_id,
            "name": tool_result.name,
            "content": tool_result.output_json,
        }

    return {
        "role": message.role,
        "content": "\n".join(fragments),
    }


@dataclass(frozen=True, slots=True)
class _ChutesResponsePayload:
    raw: dict[str, Any]
    response_id: str
    choices: tuple[LlmChoice, ...]
    usage: LlmUsage | None


class _ChutesReasoningObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    thought_text_parts: list[str] = Field(default_factory=list)
    text: str | None = None
    summary: str | None = None
    content: str | None = None

    @property
    def reasoning_text(self) -> str | None:
        normalized_parts = tuple(part.strip() for part in self.thought_text_parts if part.strip())
        if normalized_parts:
            return "\n\n".join(normalized_parts)

        for candidate in (self.text, self.summary, self.content):
            normalized_text = _normalize_reasoning_text(candidate)
            if normalized_text is not None:
                return normalized_text
        return None


def _parse_chutes_response_payload(value: object) -> _ChutesResponsePayload:
    payload = _require_object_mapping(value, label="chutes chat completions payload must be a JSON object")
    return _ChutesResponsePayload(
        raw=payload,
        response_id=_as_str(payload.get("id")),
        choices=_build_choices(payload.get("choices")),
        usage=_extract_usage(payload.get("usage")),
    )


def _build_choices(choices_payload: object) -> tuple[LlmChoice, ...]:
    if choices_payload is None:
        return ()
    if not isinstance(choices_payload, list):
        return ()

    choices: list[LlmChoice] = []
    for idx, choice_payload in enumerate(choices_payload):
        choice = _choice_from_payload(choice_payload, index=idx)
        if choice is not None:
            choices.append(choice)
    return tuple(choices)


def _choice_from_payload(choice_payload: object, *, index: int) -> LlmChoice | None:
    choice = _mapping_with_string_keys(choice_payload)
    if choice is None:
        return None
    message = _mapping_with_string_keys(choice.get("message"))
    if message is None:
        return None

    parts = _message_parts(message)
    tool_calls = _message_tool_calls(message)
    reasoning = _message_reasoning(message)
    if not parts:
        parts = (LlmMessageContentPart(type="text", text=""),)

    return LlmChoice(
        index=index,
        message=LlmChoiceMessage(
            role="assistant",
            content=parts,
            tool_calls=tool_calls,
            reasoning=reasoning,
        ),
        finish_reason="stop",
    )


def _message_parts(message: Mapping[str, object]) -> tuple[LlmMessageContentPart, ...]:
    content_value = message.get("content")
    if isinstance(content_value, str):
        return (LlmMessageContentPart(type="text", text=content_value),)
    if isinstance(content_value, list):
        parts: list[LlmMessageContentPart] = []
        for part in content_value:
            part_mapping = _mapping_with_string_keys(part)
            if part_mapping is None:
                continue
            text_raw = part_mapping.get("text")
            if not isinstance(text_raw, str):
                continue
            type_raw = part_mapping.get("type")
            if type_raw is not None and not isinstance(type_raw, str):
                continue
            parts.append(
                LlmMessageContentPart(
                    type=type_raw or "text",
                    text=text_raw,
                )
            )
        return tuple(parts)
    return ()


def _message_tool_calls(message: Mapping[str, object]) -> tuple[LlmMessageToolCall, ...]:
    tool_calls_value = message.get("tool_calls")
    if tool_calls_value is None:
        return ()
    if not isinstance(tool_calls_value, list):
        raise RuntimeError("chutes message tool_calls must be an array")

    calls: list[LlmMessageToolCall] = []
    for index, call_payload in enumerate(tool_calls_value):
        call = _tool_call_from_payload(call_payload, index=index)
        if call is not None:
            calls.append(call)
    return tuple(calls)


def _message_reasoning(message: Mapping[str, object]) -> str | None:
    reasoning_value = message.get("reasoning")
    if reasoning_value is None:
        return None
    if isinstance(reasoning_value, str):
        return _normalize_reasoning_text(reasoning_value)
    if isinstance(reasoning_value, Mapping):
        return _reasoning_text_from_object(
            _require_object_mapping(reasoning_value, label="chutes message reasoning must be a JSON object")
        )
    raise RuntimeError("chutes message reasoning must be a string or JSON object")


def _reasoning_text_from_object(reasoning_value: Mapping[str, object]) -> str | None:
    try:
        reasoning_payload = _ChutesReasoningObject.model_validate(reasoning_value, strict=True)
    except ValidationError as exc:
        raise RuntimeError("chutes message reasoning object shape is invalid") from exc
    return reasoning_payload.reasoning_text


def _normalize_reasoning_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized_text = value.strip()
    return normalized_text or None


def _tool_call_from_payload(payload: object, index: int) -> LlmMessageToolCall | None:
    call = _mapping_with_string_keys(payload)
    if call is None:
        return None
    function = _mapping_with_string_keys(call.get("function"))
    if function is None:
        return None

    name_raw = function.get("name")
    if not isinstance(name_raw, str) or not name_raw:
        return None
    arguments_raw = function.get("arguments")
    if isinstance(arguments_raw, Mapping):
        arguments = json.dumps(arguments_raw)
    elif isinstance(arguments_raw, str):
        arguments = arguments_raw
    else:
        return None

    return LlmMessageToolCall(
        id=str(call.get("id", f"toolcall-{index}")),
        type=str(call.get("type", "function")),
        name=name_raw,
        arguments=arguments,
    )


def _extract_usage(usage_payload: object) -> LlmUsage | None:
    if usage_payload is None:
        return None
    usage_mapping = _require_object_mapping(usage_payload, label="chutes usage payload must be a JSON object")
    prompt_tokens = usage_mapping.get("prompt_tokens")
    completion_tokens = usage_mapping.get("completion_tokens")
    total_tokens = usage_mapping.get("total_tokens")
    return LlmUsage(
        prompt_tokens=_optional_int(prompt_tokens),
        completion_tokens=_optional_int(completion_tokens),
        total_tokens=_optional_int(total_tokens),
    )


def _summarize_response(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = response.text
    summary_payload = _parse_response_summary_payload(data)
    summary_value = summary_payload.detail if summary_payload.detail is not None else summary_payload.raw
    text = str(summary_value)
    return text if len(text) <= 500 else text[:500] + "…"


@dataclass(frozen=True, slots=True)
class _ResponseSummaryPayload:
    raw: object
    detail: object | None


def _parse_response_summary_payload(value: object) -> _ResponseSummaryPayload:
    data_mapping = _mapping_with_string_keys(value)
    if data_mapping is None:
        return _ResponseSummaryPayload(raw=value, detail=None)
    return _ResponseSummaryPayload(raw=value, detail=data_mapping.get("detail"))


def _mapping_with_string_keys(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _require_object_mapping(value: object, *, label: str) -> dict[str, Any]:
    result = _mapping_with_string_keys(value)
    if result is None:
        raise RuntimeError(label)
    return result


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return int(value)
    raise TypeError("usage token fields must be numeric")


def _iter_tool_calls(response: LlmResponse) -> tuple[LlmMessageToolCall, ...]:
    calls: list[LlmMessageToolCall] = []
    for choice in response.choices:
        calls.extend(choice.message.tool_calls or ())
    return tuple(calls)


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


__all__ = [
    "ChutesLlmProvider",
    "ChutesTextEmbeddingClient",
    "resolve_chutes_embedding_base_url",
]
