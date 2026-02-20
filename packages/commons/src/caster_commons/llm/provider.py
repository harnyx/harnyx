"""Provider interface for LLM clients.

Shared between platform and validator so that provider implementations live
in ``caster_commons``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
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
from caster_commons.observability.langfuse import start_llm_generation, update_generation_best_effort

ALLOWED_LLM_PROVIDERS: tuple[LlmProviderName, ...] = (
    "openai",
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

    def is_exhausted(self, attempt: int) -> bool:
        return (attempt + 1) >= self.policy.attempts


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
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, "032x") if span_context.trace_id else None
            with start_llm_generation(
                trace_id=trace_id,
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
                    update_generation_best_effort(
                        generation,
                        metadata={
                            "error": repr(exc),
                            "elapsed_ms": elapsed,
                            "wait_ms": round(wait_ms, 2),
                        },
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
                response_metadata = response.metadata or {}

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
                    output={
                        "raw_text": response.raw_text,
                        "payload": response.payload,
                    },
                    usage=usage,
                    metadata={
                        "provider": self._provider_label,
                        "elapsed_ms": elapsed,
                        "wait_ms": round(wait_ms, 2),
                        "finish_reason": response.finish_reason,
                        "response_metadata": response_metadata,
                    },
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

        raise RuntimeError("LLM call exhausted all retry attempts")

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
        await self._handle_failure(attempt, ctx, request, "verifier", reason or "verification_failed", retryable)
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
            attempt, ctx, request, "postprocess", result.reason or "postprocess_failed", result.retryable
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
    ) -> None:
        """Handle a phase failure - either raise or prepare for retry."""
        if ctx.is_exhausted(attempt) or not retryable:
            raise RuntimeError(reason)
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
        return asdict(request)


def _usage_snapshot(usage: LlmUsage) -> dict[str, object]:
    return {
        "prompt": usage.prompt_tokens or 0,
        "prompt_cached": usage.prompt_cached_tokens or 0,
        "completion": usage.completion_tokens or 0,
        "total": usage.total_tokens or 0,
        "reasoning": usage.reasoning_tokens or 0,
        "web_search_calls": usage.web_search_calls or 0,
    }

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
