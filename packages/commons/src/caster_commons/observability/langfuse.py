"""Shared Langfuse helpers for LLM generation observability."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, is_dataclass
from types import TracebackType
from typing import Literal, Protocol, cast

from langfuse import Langfuse, propagate_attributes

from caster_commons.llm.schema import (
    AbstractLlmRequest,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessage,
    LlmResponse,
    LlmUsage,
)

_LOGGER = logging.getLogger("caster_commons.observability.langfuse")
_REQUIRED_ENV_VARS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
_SERVER_LABEL_ENV_VARS = ("OTEL_SERVICE_NAME", "K_SERVICE", "SERVICE_NAME")
_LOW_CARDINALITY_TAG_KEYS = ("server", "use_case")
_UNKNOWN_SERVER_LABEL = "unknown"
_LANGFUSE_CLIENT: Langfuse | None = None


class LangfuseGeneration(Protocol):
    def update(self, **kwargs: object) -> None: ...


class _LangfuseGenerationScope(AbstractContextManager[LangfuseGeneration | None]):
    def __init__(
        self,
        *,
        client: Langfuse | None,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> None:
        self._client = client
        self._provider_label = provider_label
        self._request = request
        self._observation_cm: AbstractContextManager[object] | None = None
        self._propagate_cm: AbstractContextManager[object] | None = None

    def __enter__(self) -> LangfuseGeneration | None:
        if self._client is None:
            return None

        metadata = build_generation_metadata(
            provider_label=self._provider_label,
            request=self._request,
        )
        try:
            tags = _derive_tags(metadata)
            if tags:
                self._propagate_cm = cast(
                    AbstractContextManager[object],
                    propagate_attributes(tags=tags),
                )
                self._propagate_cm.__enter__()
            self._observation_cm = cast(
                AbstractContextManager[object],
                self._client.start_as_current_observation(
                    name="llm.invoke",
                    as_type="generation",
                    model=self._request.model,
                    model_parameters=_model_parameters(self._request),
                ),
            )
            generation = cast(LangfuseGeneration, self._observation_cm.__enter__())
            update_generation_best_effort(
                generation,
                input_payload=build_generation_input_payload(self._request),
                metadata=metadata,
            )
            return generation
        except Exception:
            self._close_propagate_scope()
            _LOGGER.exception(
                "langfuse.generation.start_failed",
                extra={
                    "data": {
                        "provider": self._provider_label,
                        "model": self._request.model,
                    }
                },
            )
            self._observation_cm = None
            return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        result = False
        try:
            if self._observation_cm is None:
                return False
            result = bool(self._observation_cm.__exit__(exc_type, exc, exc_tb))
        except Exception:
            _LOGGER.exception(
                "langfuse.generation.finish_failed",
                extra={
                    "data": {
                        "provider": self._provider_label,
                        "model": self._request.model,
                    }
                },
            )
            result = False
        finally:
            self._close_propagate_scope(exc_type=exc_type, exc=exc, exc_tb=exc_tb)
        return result

    def _close_propagate_scope(
        self,
        *,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        if self._propagate_cm is None:
            return
        propagate_cm = self._propagate_cm
        self._propagate_cm = None
        try:
            propagate_cm.__exit__(exc_type, exc, exc_tb)
        except Exception:
            _LOGGER.exception(
                "langfuse.generation.propagate_cleanup_failed",
                extra={
                    "data": {
                        "provider": self._provider_label,
                        "model": self._request.model,
                    }
                },
            )


class _LangfuseTraceAttributesScope(AbstractContextManager[None]):
    def __init__(
        self,
        *,
        trace_name: str | None,
        session_id: str | None,
        metadata: Mapping[str, object] | None,
        tags: Sequence[str] | None,
    ) -> None:
        self._trace_name = trace_name
        self._session_id = session_id
        self._metadata = metadata
        self._tags = tags
        self._propagate_cm: AbstractContextManager[object] | None = None

    def __enter__(self) -> None:
        metadata_payload = _normalize_trace_metadata(self._metadata)
        tags_payload: list[str] | None = None
        if self._tags is not None:
            tags_payload = [str(tag) for tag in self._tags]

        if (
            self._trace_name is None
            and self._session_id is None
            and metadata_payload is None
            and tags_payload is None
        ):
            return None

        try:
            self._propagate_cm = cast(
                AbstractContextManager[object],
                propagate_attributes(
                    trace_name=self._trace_name,
                    session_id=self._session_id,
                    metadata=metadata_payload,
                    tags=tags_payload,
                ),
            )
            self._propagate_cm.__enter__()
        except Exception:
            self._propagate_cm = None
            _LOGGER.exception(
                "langfuse.trace.propagate_start_failed",
                extra={
                    "data": {
                        "trace_name": self._trace_name,
                        "session_id": self._session_id,
                    }
                },
            )
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self._propagate_cm is None:
            return False

        propagate_cm = self._propagate_cm
        self._propagate_cm = None
        try:
            return bool(propagate_cm.__exit__(exc_type, exc, exc_tb))
        except Exception:
            _LOGGER.exception(
                "langfuse.trace.propagate_cleanup_failed",
                extra={
                    "data": {
                        "trace_name": self._trace_name,
                        "session_id": self._session_id,
                    }
                },
            )
            return False


def get_client() -> Langfuse | None:
    """Return the singleton Langfuse client when fully configured."""

    global _LANGFUSE_CLIENT
    config = _read_config()
    if config is None:
        return None
    if _LANGFUSE_CLIENT is None:
        _LANGFUSE_CLIENT = Langfuse(
            base_url=config["LANGFUSE_HOST"],
            public_key=config["LANGFUSE_PUBLIC_KEY"],
            secret_key=config["LANGFUSE_SECRET_KEY"],
        )
    return _LANGFUSE_CLIENT


def start_llm_generation(
    *,
    provider_label: str,
    request: AbstractLlmRequest,
) -> AbstractContextManager[LangfuseGeneration | None]:
    """Start a Langfuse generation scope for an LLM call.

    - Returns a no-op scope when Langfuse is not configured.
    - Raises when Langfuse env vars are partially configured.
    """

    client = get_client()
    return _LangfuseGenerationScope(
        client=client,
        provider_label=provider_label,
        request=request,
    )


def propagate_trace_attributes_best_effort(
    *,
    trace_name: str | None = None,
    session_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    tags: Sequence[str] | None = None,
) -> AbstractContextManager[None]:
    """Best-effort trace attribute propagation.

    - Returns a no-op scope when Langfuse is not configured.
    - Raises when Langfuse env vars are partially configured.
    - Never raises when trace propagation enter/exit fails.
    """

    client = get_client()
    if client is None:
        return nullcontext()
    return _LangfuseTraceAttributesScope(
        trace_name=trace_name,
        session_id=session_id,
        metadata=metadata,
        tags=tags,
    )


def build_generation_input_payload(request: AbstractLlmRequest) -> dict[str, object]:
    return {
        "messages": [_request_message_payload(message) for message in request.messages],
        "request_config": {
            "provider": request.provider,
            "model": request.model,
            "grounded": request.grounded,
            "output_mode": request.output_mode,
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "timeout_seconds": request.timeout_seconds,
            "tool_choice": request.tool_choice,
            "reasoning_effort": request.reasoning_effort,
        },
        "tools": _sanitize_for_json(list(request.tools or ())),
        "include": [str(item) for item in request.include] if request.include is not None else None,
        "extra": _sanitize_for_json(dict(request.extra)) if request.extra is not None else None,
    }


def build_generation_output_payload(response: LlmResponse) -> dict[str, object]:
    payload: dict[str, object] = {
        "assistant": {
            "role": "assistant",
            "text": response.raw_text,
        },
        "finish_reason": response.finish_reason,
    }
    if response.postprocessed is not None:
        payload["postprocessed"] = _sanitize_for_json(response.postprocessed)
    if response.tool_calls:
        payload["tool_calls"] = [
            {
                "name": call.name,
                "arguments": call.arguments,
                "output": call.output,
            }
            for call in response.tool_calls
        ]
    return payload


def update_generation_best_effort(
    generation: LangfuseGeneration | None,
    *,
    input_payload: object | None = None,
    output: object | None = None,
    usage: LlmUsage | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Best-effort generation update that never raises."""

    if generation is None:
        return

    update_data: dict[str, object] = {}
    if input_payload is not None:
        update_data["input"] = _sanitize_for_json(input_payload)
    if output is not None:
        update_data["output"] = _sanitize_for_json(output)
    if usage is not None:
        update_data["usage_details"] = _usage_details(usage)
    if metadata is not None:
        update_data["metadata"] = _sanitize_for_json(dict(metadata))
    if not update_data:
        return

    try:
        generation.update(**update_data)
    except Exception:
        _LOGGER.exception(
            "langfuse.generation.update_failed",
            extra={"data": {"fields": sorted(update_data.keys())}},
        )


def record_child_observation_best_effort(
    *,
    name: str,
    as_type: Literal["tool", "retriever", "agent"],
    input_payload: object | None = None,
    output: object | None = None,
    usage: LlmUsage | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Best-effort child observation recording that never raises."""

    try:
        client = get_client()
    except Exception:
        _LOGGER.exception("langfuse.client.read_failed")
        return

    if client is None:
        return

    update_data: dict[str, object] = {}
    if input_payload is not None:
        update_data["input"] = _sanitize_for_json(input_payload)
    if output is not None:
        update_data["output"] = _sanitize_for_json(output)
    if usage is not None:
        update_data["usage_details"] = _usage_details(usage)
    if metadata is not None:
        update_data["metadata"] = _sanitize_for_json(dict(metadata))

    try:
        observation_cm = cast(
            AbstractContextManager[object],
            client.start_as_current_observation(name=name, as_type=as_type),
        )
        with observation_cm as observation:
            if update_data:
                cast(LangfuseGeneration, observation).update(**update_data)
    except Exception:
        _LOGGER.exception(
            "langfuse.observation.record_failed",
            extra={"data": {"name": name, "as_type": as_type}},
        )


def build_generation_metadata(
    *,
    provider_label: str,
    request: AbstractLlmRequest,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    internal_metadata = request.internal_metadata or {}
    merged: dict[str, object] = {str(key): value for key, value in internal_metadata.items()}
    merged["provider"] = provider_label
    merged["server"] = _resolve_server_label()
    if metadata is not None:
        merged.update({str(key): value for key, value in metadata.items()})
    return merged


def _read_config() -> dict[str, str] | None:
    values = {key: (os.getenv(key) or "").strip() for key in _REQUIRED_ENV_VARS}
    configured = [key for key, value in values.items() if value]
    if not configured:
        return None

    if len(configured) != len(_REQUIRED_ENV_VARS):
        missing = [key for key in _REQUIRED_ENV_VARS if not values[key]]
        raise RuntimeError(
            "Langfuse configuration is partial. Set LANGFUSE_HOST, "
            "LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY together. "
            f"Missing: {', '.join(missing)}."
        )

    return values


def _model_parameters(request: AbstractLlmRequest) -> dict[str, str | int | bool | list[str] | None]:
    params: dict[str, str | int | bool | list[str] | None] = {
        "temperature": None if request.temperature is None else str(request.temperature),
        "max_output_tokens": request.max_output_tokens,
        "tool_choice": request.tool_choice,
        "grounded": request.grounded,
        "output_mode": request.output_mode,
        "reasoning_effort": request.reasoning_effort,
        "timeout_seconds": None if request.timeout_seconds is None else str(request.timeout_seconds),
    }
    if request.include is not None:
        params["include"] = [str(item) for item in request.include]
    return {key: value for key, value in params.items() if value is not None}


def _request_message_payload(message: LlmMessage) -> dict[str, object]:
    return {
        "role": message.role,
        "content": [_request_content_part_payload(part) for part in message.content],
    }


def _request_content_part_payload(part: object) -> dict[str, object]:
    match part:
        case LlmInputTextPart(text=text):
            return {"type": "input_text", "text": text}
        case LlmInputImagePart(data=image_data):
            return {
                "type": "input_image",
                "url": image_data.url,
                "mime_type": image_data.mime_type,
            }
        case LlmInputToolResultPart(tool_call_id=tool_call_id, name=name, output_json=output_json):
            return {
                "type": "input_tool_result",
                "tool_call_id": tool_call_id,
                "name": name,
                "output_json": output_json,
            }
        case _:
            return {
                "type": "unknown",
                "value": _sanitize_for_json(part),
            }


def _resolve_server_label() -> str:
    for key in _SERVER_LABEL_ENV_VARS:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return _UNKNOWN_SERVER_LABEL


def _derive_tags(metadata: Mapping[str, object]) -> list[str]:
    tags: list[str] = []
    for key in _LOW_CARDINALITY_TAG_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        tags.append(f"{key}:{normalized}")
    return tags


def _normalize_trace_metadata(metadata: Mapping[str, object] | None) -> dict[str, str] | None:
    if metadata is None:
        return None
    normalized = {
        str(key): str(value)
        for key, value in metadata.items()
        if value is not None
    }
    return normalized or None


def _usage_details(usage: LlmUsage) -> dict[str, int]:
    details: dict[str, int] = {
        "input": int(usage.prompt_tokens or 0),
        "output": int(usage.completion_tokens or 0),
        "total": int(usage.total_tokens or 0),
        "input_cached": int(usage.prompt_cached_tokens or 0),
        "reasoning": int(usage.reasoning_tokens or 0),
        "web_search_calls": int(usage.web_search_calls or 0),
    }
    return details


def _sanitize_for_json(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize_for_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_for_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


__all__ = [
    "LangfuseGeneration",
    "build_generation_input_payload",
    "build_generation_metadata",
    "build_generation_output_payload",
    "get_client",
    "propagate_trace_attributes_best_effort",
    "record_child_observation_best_effort",
    "start_llm_generation",
    "update_generation_best_effort",
]
