"""Shared Langfuse helpers for LLM generation observability."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from types import TracebackType
from typing import Any, Protocol, cast

from langfuse import Langfuse

from caster_commons.llm.schema import AbstractLlmRequest, LlmUsage

_LOGGER = logging.getLogger("caster_commons.observability.langfuse")
_REQUIRED_ENV_VARS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
_LANGFUSE_CLIENT: Langfuse | None = None


class LangfuseGeneration(Protocol):
    def update(self, **kwargs: object) -> None: ...


class _LangfuseGenerationScope(AbstractContextManager[LangfuseGeneration | None]):
    def __init__(
        self,
        *,
        client: Langfuse | None,
        trace_id: str | None,
        provider_label: str,
        request: AbstractLlmRequest,
    ) -> None:
        self._client = client
        self._trace_id = trace_id
        self._provider_label = provider_label
        self._request = request
        self._observation_cm: AbstractContextManager[object] | None = None

    def __enter__(self) -> LangfuseGeneration | None:
        if self._client is None:
            return None

        try:
            self._observation_cm = cast(
                AbstractContextManager[object],
                self._client.start_as_current_observation(
                    name="llm.invoke",
                    as_type="generation",
                    model=self._request.model,
                    model_parameters=_model_parameters(self._request),
                    trace_context=cast(Any, {"trace_id": self._trace_id} if self._trace_id else None),
                ),
            )
            generation = cast(LangfuseGeneration, self._observation_cm.__enter__())
            update_generation_best_effort(
                generation,
                input_payload=_request_payload(self._request),
                metadata={"provider": self._provider_label},
            )
            return generation
        except Exception:
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
        if self._observation_cm is None:
            return False
        try:
            return bool(self._observation_cm.__exit__(exc_type, exc, exc_tb))
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
    trace_id: str | None,
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
        trace_id=trace_id,
        provider_label=provider_label,
        request=request,
    )


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


def _request_payload(request: AbstractLlmRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider": request.provider,
        "model": request.model,
        "messages": _sanitize_for_json(list(request.messages)),
        "tools": _sanitize_for_json(list(request.tools or ())),
        "tool_choice": request.tool_choice,
        "grounded": request.grounded,
        "output_mode": request.output_mode,
        "reasoning_effort": request.reasoning_effort,
        "max_output_tokens": request.max_output_tokens,
        "timeout_seconds": request.timeout_seconds,
        "temperature": request.temperature,
    }
    if request.extra is not None:
        payload["extra"] = _sanitize_for_json(dict(request.extra))
    if request.include is not None:
        payload["include"] = [str(item) for item in request.include]
    return payload


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
    "get_client",
    "start_llm_generation",
    "update_generation_best_effort",
]
