"""Provider interface for LLM clients.

Shared between platform and validator so that provider implementations live
in ``caster_commons``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Protocol, cast

from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.util.types import AttributeValue

from caster_commons.config.external_client import ExternalClientRetrySettings
from caster_commons.llm.provider_types import LlmProviderName
from caster_commons.llm.retry_utils import RetryPolicy, backoff_ms
from caster_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmCitation,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmTool,
    LlmToolCall,
    LlmUsage,
    PostprocessResult,
)
from caster_commons.observability.langfuse import (
    build_generation_metadata,
    build_generation_output_payload,
    record_child_observation_best_effort,
    start_llm_generation,
    update_generation_best_effort,
)

ALLOWED_LLM_PROVIDERS: tuple[LlmProviderName, ...] = (
    "chutes",
    "vertex",
    "vertex-maas",
)



def parse_provider_name(raw: str | None, *, component: str) -> LlmProviderName:
    """Parse and validate an LLM provider label."""
    if raw is None:
        raise ValueError(f"{component} llm provider must be specified")
    value = raw.strip()
    if not value or value not in ALLOWED_LLM_PROVIDERS:
        raise ValueError(f"{component} llm provider {value!r} is not allowed")
    return cast(LlmProviderName, value)


@dataclass
class RetryContext:
    """Encapsulates retry state across attempts."""

    policy: RetryPolicy
    reasons: list[str] = field(default_factory=list)
    total_usage: LlmUsage = field(default_factory=LlmUsage)
    total_latency_ms: float = 0.0
    last_response: LlmResponse | None = None

    def is_exhausted(self, attempt: int) -> bool:
        return (attempt + 1) >= self.policy.attempts


class LlmRetryExhaustedError(RuntimeError):
    """Retry flow failed after exhausting attempts."""

    def __init__(self, reason: str, *, response: LlmResponse | None = None) -> None:
        super().__init__(reason)
        self.response = response


class LlmProviderPort(Protocol):
    """Abstraction over provider/model specific LLM clients."""

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        """Execute the supplied request and return a normalized response."""
        ...

    async def aclose(self) -> None:
        """Release provider resources."""
        ...


class BaseLlmProvider(ABC, LlmProviderPort):
    """Base class that adds timing-aware logging around provider invocations."""

    _LOGGER_NAME = "caster_commons.llm.calls"

    def __init__(
        self,
        *,
        provider_label: str,
        logger: logging.Logger | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        self._provider_label = provider_label
        self._llm_logger = logger or logging.getLogger(self._LOGGER_NAME)
        self._retry_policy = ExternalClientRetrySettings().retry_policy
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None
        )

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        data: dict[str, object] = {
            "provider": self._provider_label,
            "model": request.model,
            "max_output_tokens": request.max_output_tokens,
            "reasoning_effort": request.reasoning_effort,
            "timeout_seconds": request.timeout_seconds,
        }
        data |= request.internal_metadata or {}
        data |= request.extra or {}

        span_attributes: dict[str, AttributeValue] = {
            "llm.provider": self._provider_label,
            "llm.model": request.model,
            "llm.grounded": bool(request.grounded),
        }
        if request.max_output_tokens is not None:
            span_attributes["llm.max_output_tokens"] = int(request.max_output_tokens)
        if request.reasoning_effort is not None:
            span_attributes["llm.reasoning_effort"] = str(request.reasoning_effort)

        tracer = trace.get_tracer("caster_commons.llm")
        with tracer.start_as_current_span(
            "llm.invoke",
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
        ) as span:
            with start_llm_generation(
                provider_label=self._provider_label,
                request=request,
            ) as generation:
                self._llm_logger.debug("llm.invoke.start", extra={"data": data})
                start = time.perf_counter()
                wait_ms = 0.0
                try:
                    if self._semaphore is None:
                        response = await self._invoke(request)
                    else:
                        wait_start = time.perf_counter()
                        async with self._semaphore:
                            wait_ms = (time.perf_counter() - wait_start) * 1000
                            self._llm_logger.debug(
                                "llm.invoke.semaphore.wait",
                                extra={"data": data | {"wait_ms": wait_ms}},
                            )
                            response = await self._invoke(request)
                except Exception as exc:
                    elapsed = round((time.perf_counter() - start) * 1000, 2)
                    error_metadata: dict[str, object] = {
                        "error": repr(exc),
                        "elapsed_ms": elapsed,
                        "wait_ms": round(wait_ms, 2),
                    }
                    raw_error_payload = _error_raw_payload_metadata(request=request, exc=exc)
                    if raw_error_payload is not None:
                        error_metadata["raw"] = raw_error_payload
                    update_generation_best_effort(
                        generation,
                        metadata=build_generation_metadata(
                            provider_label=self._provider_label,
                            request=request,
                            metadata=error_metadata,
                        ),
                    )
                    self._llm_logger.exception(
                        "llm.invoke.error",
                        extra={"data": data | {"elapsed_ms": elapsed}},
                    )
                    span.set_attributes(
                        {
                            "llm.elapsed_ms": elapsed,
                            "llm.wait_ms": round(wait_ms, 2),
                        }
                    )
                    raise

                elapsed = round((time.perf_counter() - start) * 1000, 2)
                usage = response.usage or LlmUsage()
                web_search_calls = int(usage.web_search_calls or 0)
                response_metadata = dict(response.metadata or {})
                reasoning_metadata = _build_reasoning_metadata(
                    provider_label=self._provider_label,
                    request=request,
                    response=response,
                    usage=usage,
                )
                grounding_metadata = _build_grounding_metadata(
                    response_metadata=response_metadata,
                    web_search_calls=web_search_calls,
                )
                generation_metadata: dict[str, object] = {
                    "elapsed_ms": elapsed,
                    "wait_ms": round(wait_ms, 2),
                    "finish_reason": response.finish_reason,
                    "response_metadata": response_metadata,
                    "raw": _build_raw_payload_metadata(
                        request=request,
                        response=response,
                        response_metadata=response_metadata,
                    ),
                }
                if reasoning_metadata is not None:
                    generation_metadata["reasoning"] = reasoning_metadata
                if grounding_metadata is not None:
                    generation_metadata["grounding"] = grounding_metadata

                span.set_attributes(
                    {
                        "llm.elapsed_ms": elapsed,
                        "llm.wait_ms": round(wait_ms, 2),
                        "llm.usage.total_tokens": int(usage.total_tokens or 0),
                        "llm.usage.prompt_tokens": int(usage.prompt_tokens or 0),
                        "llm.usage.completion_tokens": int(usage.completion_tokens or 0),
                        "llm.usage.reasoning_tokens": int(usage.reasoning_tokens or 0),
                        "llm.usage.web_search_calls": web_search_calls,
                    }
                )

                data |= {
                    "request": _request_snapshot(request),
                    "response": response.payload,
                    "response_metadata": response_metadata,
                    "elapsed_ms": elapsed,
                    "usage_prompt": usage.prompt_tokens or 0,
                    "usage_completion": usage.completion_tokens or 0,
                    "usage_total": usage.total_tokens or 0,
                    "reasoning_tokens": usage.reasoning_tokens or 0,
                    "finish_reason": response.finish_reason,
                    "web_search_calls": web_search_calls,
                    "wait_ms": round(wait_ms, 2),
                }
                update_generation_best_effort(
                    generation,
                    output=build_generation_output_payload(response),
                    usage=usage,
                    metadata=build_generation_metadata(
                        provider_label=self._provider_label,
                        request=request,
                        metadata=generation_metadata,
                    ),
                )
                if generation is not None:
                    _record_child_observations(
                        provider_label=self._provider_label,
                        model=request.model,
                        response=response,
                        response_metadata=response_metadata,
                    )
                return response

    async def aclose(self) -> None:
        """Providers may override to close network clients."""
        return None

    @abstractmethod
    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        """Provider-specific invocation; implemented by concrete providers."""

    async def _call_with_retry(
        self,
        request: AbstractLlmRequest,
        *,
        call_coro: Callable[[], Awaitable[LlmResponse]],
        verifier: Callable[[LlmResponse], tuple[bool, bool, str | None]],
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
        policy: RetryPolicy | None = None,
    ) -> LlmResponse:
        """Execute LLM call with retry. Main orchestrator."""
        policy = policy or self._retry_policy
        ctx = RetryContext(policy)

        for attempt in range(policy.attempts):
            # Phase 1: Attempt
            response = await self._try_call(attempt, ctx, request, call_coro, classify_exception)
            if response is None:
                continue

            # Phase 2: Verify
            if not await self._try_verify(attempt, ctx, request, response, verifier):
                continue

            # Phase 3: Postprocess
            processed, retry = await self._try_postprocess(attempt, ctx, request, response)
            if retry:
                continue

            # Success
            return self._build_response(request, response, ctx, processed)

        raise LlmRetryExhaustedError("LLM call exhausted all retry attempts", response=ctx.last_response)

    async def _try_call(
        self,
        attempt: int,
        ctx: RetryContext,
        request: AbstractLlmRequest,
        call_coro: Callable[[], Awaitable[LlmResponse]],
        classify_exception: Callable[[Exception], tuple[bool, str]] | None,
    ) -> LlmResponse | None:
        """Attempt the LLM call. Returns None if retry needed."""
        start = time.perf_counter()
        try:
            response = await call_coro()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            retryable, reason = classify_exception(exc) if classify_exception else (False, str(exc))
            await self._handle_failure(attempt, ctx, request, "exception", reason, retryable)
            return None

        latency_ms = (time.perf_counter() - start) * 1000
        ctx.total_latency_ms += latency_ms
        ctx.total_usage += response.usage or LlmUsage()
        ctx.last_response = response
        return response

    async def _try_verify(
        self,
        attempt: int,
        ctx: RetryContext,
        request: AbstractLlmRequest,
        response: LlmResponse,
        verifier: Callable[[LlmResponse], tuple[bool, bool, str | None]],
    ) -> bool:
        """Verify response. Returns False if retry needed."""
        ok, retryable, reason = verifier(response)
        if ok:
            return True
        await self._handle_failure(
            attempt,
            ctx,
            request,
            "verifier",
            reason or "verification_failed",
            retryable,
            response=response,
        )
        return False

    async def _try_postprocess(
        self,
        attempt: int,
        ctx: RetryContext,
        request: AbstractLlmRequest,
        response: LlmResponse,
    ) -> tuple[object | None, bool]:
        """Postprocess response. Returns (processed, should_retry)."""
        if request.postprocessor is None:
            return None, False

        result = request.postprocessor(response)
        if result.ok:
            return result.processed, False

        await self._handle_failure(
            attempt,
            ctx,
            request,
            "postprocess",
            result.reason or "postprocess_failed",
            result.retryable,
            response=response,
        )
        return None, True

    async def _handle_failure(
        self,
        attempt: int,
        ctx: RetryContext,
        request: AbstractLlmRequest,
        phase: str,
        reason: str,
        retryable: bool,
        response: LlmResponse | None = None,
    ) -> None:
        """Handle a phase failure - either raise or prepare for retry."""
        response_for_error = response if response is not None else ctx.last_response
        if ctx.is_exhausted(attempt) or not retryable:
            raise LlmRetryExhaustedError(reason, response=response_for_error)
        ctx.reasons.append(reason)
        self._log_retry(phase, request, attempt, reason, ctx.policy)
        await asyncio.sleep(backoff_ms(attempt, ctx.policy) / 1000)

    def _build_response(
        self,
        request: AbstractLlmRequest,
        response: LlmResponse,
        ctx: RetryContext,
        processed: object | None,
    ) -> LlmResponse:
        """Construct final response with accumulated metadata."""
        metadata = dict(response.metadata or {})
        metadata["attempts"] = len(ctx.reasons) + 1
        metadata["retry_reasons"] = tuple(ctx.reasons)

        self._log_retry_complete(request=request, response=response, ctx=ctx)

        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=ctx.total_usage,
            metadata=metadata,
            postprocessed=processed,
            finish_reason=response.finish_reason,
        )

    def _log_retry_complete(self, *, request: AbstractLlmRequest, response: LlmResponse, ctx: RetryContext) -> None:
        request_snapshot = _request_snapshot(request)
        response_payload = response.payload
        response_raw = (response.metadata or {}).get("raw_response")
        self._llm_logger.info(
            "llm.invoke.retry.complete",
            extra={
                "data": {
                    "provider": request.provider,
                    "model": request.model,
                    "attempts": len(ctx.reasons) + 1,
                    "latency_ms_total": round(ctx.total_latency_ms, 2),
                    "retry_reasons": tuple(ctx.reasons),
                    "usage": _usage_snapshot(ctx.total_usage),
                },
                "json_fields": {
                    "request": request_snapshot,
                    "response": response_payload,
                    "response_raw": response_raw,
                },
            },
        )

    def _log_retry(
        self,
        phase: str,
        request: AbstractLlmRequest,
        attempt: int,
        reason: str,
        policy: RetryPolicy,
    ) -> None:
        """Unified retry event logger."""
        self._llm_logger.warning(
            f"llm.retry.{phase}",
            extra={
                "data": {
                    "provider": self._provider_label,
                    "model": request.model,
                    "attempt": attempt + 1,
                    "reason": reason,
                    "backoff_ms": backoff_ms(attempt, policy),
                },
            },
        )


def _request_snapshot(request: AbstractLlmRequest) -> dict[str, object]:
    if is_dataclass(request):
        return cast(dict[str, object], asdict(request))
    raise TypeError("llm request snapshot requires dataclass request")


def _usage_snapshot(usage: LlmUsage) -> dict[str, object]:
    return {
        "prompt": usage.prompt_tokens or 0,
        "prompt_cached": usage.prompt_cached_tokens or 0,
        "completion": usage.completion_tokens or 0,
        "total": usage.total_tokens or 0,
        "reasoning": usage.reasoning_tokens or 0,
        "web_search_calls": usage.web_search_calls or 0,
    }


def _build_raw_payload_metadata(
    *,
    request: AbstractLlmRequest,
    response: LlmResponse,
    response_metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "request": _request_snapshot(request),
        "response_payload": response.payload,
        "response_metadata": dict(response_metadata),
        "provider_response": _provider_response_payload(
            response=response,
            response_metadata=response_metadata,
        ),
    }


def _error_raw_payload_metadata(
    *,
    request: AbstractLlmRequest,
    exc: Exception,
) -> dict[str, object] | None:
    if not isinstance(exc, LlmRetryExhaustedError):
        return None

    response = exc.response
    if response is None:
        return None
    response_metadata = dict(response.metadata or {})
    return _build_raw_payload_metadata(
        request=request,
        response=response,
        response_metadata=response_metadata,
    )


def _provider_response_payload(
    *,
    response: LlmResponse,
    response_metadata: Mapping[str, object],
) -> object:
    raw_response = response_metadata.get("raw_response")
    if isinstance(raw_response, Mapping):
        return dict(raw_response)
    if isinstance(raw_response, Sequence) and not isinstance(raw_response, (str, bytes, bytearray)):
        return list(raw_response)
    return response.payload


def _build_reasoning_metadata(
    *,
    provider_label: str,
    request: AbstractLlmRequest,
    response: LlmResponse,
    usage: LlmUsage,
) -> dict[str, object] | None:
    reasoning_payload = _first_reasoning_payload(response)
    thought_text_parts = _normalize_thought_text_parts(
        reasoning_payload.get("thought_text_parts") if reasoning_payload is not None else None
    )
    has_thought_signature = bool(reasoning_payload.get("has_thought_signature")) if reasoning_payload else False
    include_thoughts_requested = _is_vertex_include_thoughts_request(
        provider_label=provider_label,
        model=request.model,
        reasoning_effort=request.reasoning_effort,
    )
    reasoning_tokens = int(usage.reasoning_tokens or 0)

    if not (
        include_thoughts_requested
        or thought_text_parts
        or has_thought_signature
        or reasoning_tokens > 0
    ):
        return None

    return {
        "include_thoughts_requested": include_thoughts_requested,
        "reasoning_effort": request.reasoning_effort,
        "reasoning_text_available": bool(thought_text_parts),
        "thought_text_parts": thought_text_parts,
        "has_thought_signature": has_thought_signature,
        "reasoning_tokens": reasoning_tokens,
    }


def _is_vertex_include_thoughts_request(
    *,
    provider_label: str,
    model: str,
    reasoning_effort: str | None,
) -> bool:
    if reasoning_effort is None:
        return False
    if not provider_label.startswith("vertex"):
        return False

    # Must stay aligned with Vertex thinking_config support behavior.
    normalized_model = model.strip().lower()
    return "gemini" in normalized_model


def _first_reasoning_payload(response: LlmResponse) -> Mapping[str, object] | None:
    for choice in response.choices:
        reasoning = choice.message.reasoning
        if isinstance(reasoning, Mapping):
            return cast(Mapping[str, object], reasoning)
    return None


def _normalize_thought_text_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    normalized: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            continue
        thought_text = entry.strip()
        if thought_text:
            normalized.append(thought_text)
    return tuple(normalized)


def _build_grounding_metadata(
    *,
    response_metadata: Mapping[str, object],
    web_search_calls: int,
) -> dict[str, object] | None:
    queries = _extract_web_search_queries(response_metadata)
    if not queries and web_search_calls == 0:
        return None

    payload: dict[str, object] = {"web_search_calls": web_search_calls}
    if queries:
        payload["web_search_queries"] = queries
    return payload


def _extract_web_search_queries(response_metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw_queries = response_metadata.get("web_search_queries")
    if isinstance(raw_queries, str):
        normalized = raw_queries.strip()
        return (normalized,) if normalized else ()
    if not isinstance(raw_queries, Sequence):
        return ()

    queries: list[str] = []
    for entry in raw_queries:
        if not isinstance(entry, str):
            continue
        normalized = entry.strip()
        if normalized:
            queries.append(normalized)
    return tuple(queries)


def _retriever_observation_name(provider_label: str) -> str:
    if provider_label == "vertex":
        return "vertex.grounding.search"
    return f"{provider_label}.search.query"


def _record_child_observations(
    *,
    provider_label: str,
    model: str,
    response: LlmResponse,
    response_metadata: Mapping[str, object],
) -> None:
    queries = _extract_web_search_queries(response_metadata)
    for index, query in enumerate(queries):
        record_child_observation_best_effort(
            as_type="retriever",
            name=_retriever_observation_name(provider_label),
            input_payload={"query": query, "index": index},
            output={"issued": True},
            metadata={"provider": provider_label, "model": model},
        )

    for tool_call in response.tool_calls:
        record_child_observation_best_effort(
            as_type="tool",
            name=tool_call.name or "tool",
            input_payload={"arguments": tool_call.arguments},
            output={"result": tool_call.output},
            metadata={"provider": provider_label},
        )

__all__ = [
    "ALLOWED_LLM_PROVIDERS",
    "LlmProviderName",
    "LlmProviderPort",
    "BaseLlmProvider",
    "LlmMessage",
    "LlmRequest",
    "LlmResponse",
    "LlmCitation",
    "LlmChoice",
    "LlmChoiceMessage",
    "LlmMessageContentPart",
    "LlmMessageToolCall",
    "LlmTool",
    "LlmToolCall",
    "LlmUsage",
    "PostprocessResult",
    "parse_provider_name",
]
