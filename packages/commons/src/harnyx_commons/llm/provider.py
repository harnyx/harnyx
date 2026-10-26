"""Provider interface for LLM clients.

Shared between platform and validator so that provider implementations live
in ``harnyx_commons``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Literal, Protocol, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from harnyx_commons.config.external_client import ExternalClientRetrySettings
from harnyx_commons.llm.judge_observability import observe_judge_llm_attempt
from harnyx_commons.llm.provider_types import (
    ALLOWED_LLM_PROVIDERS,
    LlmProviderName,
    LlmRouteTarget,
    parse_builtin_provider_name,
)
from harnyx_commons.llm.retry_utils import RetryPolicy, backoff_ms
from harnyx_commons.llm.schema import (
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
    PostprocessRecovery,
    PostprocessResult,
    supports_tool_result_messages,
)
from harnyx_commons.llm.timeout import LlmAttemptTimeoutError, enforce_attempt_deadlines, resolve_timeout

_ModelT = TypeVar("_ModelT", bound=BaseModel)

# Bump when the feedback message or its request assembly changes.
POSTPROCESS_FEEDBACK_POLICY_VERSION = "retry_with_feedback_original_messages_v1"


def parse_provider_name(raw: str | None, *, component: str) -> LlmProviderName:
    """Parse and validate an LLM provider label."""
    return parse_builtin_provider_name(raw, component=component)


@dataclass
class _AttemptResponseObservation:
    response: LlmResponse | None = None


_attempt_response: ContextVar[_AttemptResponseObservation | None] = ContextVar("llm_attempt_response", default=None)


def record_observed_llm_response(response: LlmResponse) -> None:
    """Replace the current attempt's cumulative billing snapshot, without counting it twice."""
    if (observation := _attempt_response.get()) is not None:
        observation.response = response


class LlmInvocationCancelledError(asyncio.CancelledError):
    """Cancellation carrying all responses observed before execution stopped."""

    def __init__(self, response: LlmResponse) -> None:
        super().__init__("LLM invocation cancelled after a response was observed")
        self.response = response


@dataclass
class RetryContext:
    """Encapsulates retry state across attempts."""

    policy: RetryPolicy
    reasons: list[str] = field(default_factory=list)
    recovery_events: list[dict[str, object]] = field(default_factory=list)
    total_usage: LlmUsage = field(default_factory=LlmUsage)
    total_latency_ms: float = 0.0
    last_response: LlmResponse | None = None
    billable_response_count: int = 0
    _actual_cost_usd_values: list[float] = field(default_factory=list)
    _actual_cost_complete: bool = True

    def is_exhausted(self, attempt: int) -> bool:
        return (attempt + 1) >= self.policy.attempts

    @property
    def actual_cost_usd_total(self) -> float | None:
        if (
            self.billable_response_count == 0
            or not self._actual_cost_complete
            or len(self._actual_cost_usd_values) != self.billable_response_count
        ):
            return None
        return round(sum(self._actual_cost_usd_values), 12)

    def record_billable_response(self, response: LlmResponse) -> None:
        self.billable_response_count += 1
        cost = _actual_cost_usd_for_retry_total(response)
        if cost is None:
            self._actual_cost_complete = False
            return
        self._actual_cost_usd_values.append(cost)


def _actual_cost_usd_for_retry_total(response: LlmResponse) -> float | None:
    metadata = response.metadata or {}
    if "actual_cost_usd" not in metadata:
        return None
    value = metadata["actual_cost_usd"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("actual_cost_usd must be numeric when supplied")
    cost = float(value)
    if not math.isfinite(cost):
        raise ValueError("actual_cost_usd must be finite when supplied")
    if cost < 0.0:
        raise ValueError("actual_cost_usd must be non-negative when supplied")
    return cost


class LlmProviderError(RuntimeError):
    """Provider-owned failure before or during an LLM operation."""

    def __init__(
        self,
        reason: str,
        *,
        response: LlmResponse | None = None,
        effective_provider: LlmRouteTarget | None = None,
        effective_model: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.response = response
        self.effective_provider = effective_provider
        self.effective_model = effective_model


class LlmProviderConfigurationError(LlmProviderError):
    """LLM provider is missing required configuration."""


class LlmRetryExhaustedError(RuntimeError):
    """Retry flow failed after exhausting attempts."""

    def __init__(
        self,
        reason: str,
        *,
        response: LlmResponse | None = None,
        attempts: int | None = None,
        retry_reasons: tuple[str, ...] = (),
        latency_ms_total: float | None = None,
        effective_provider: LlmRouteTarget | None = None,
        effective_model: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.response = response
        self.attempts = attempts
        self.retry_reasons = retry_reasons
        self.latency_ms_total = latency_ms_total
        self.effective_provider = effective_provider
        self.effective_model = effective_model


@dataclass(frozen=True)
class RetryFailureDetail:
    reason: str
    exception_type: str | None = None
    exception_message: str | None = None
    exception_repr: str | None = None
    cause_chain: tuple[str, ...] = ()


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

    _LOGGER_NAME = "harnyx_commons.llm.calls"

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
        return await self._invoke_instrumented(request, acquire_semaphore=True)

    async def _invoke_instrumented(
        self,
        request: AbstractLlmRequest,
        *,
        acquire_semaphore: bool,
    ) -> LlmResponse:
        data: dict[str, object] = {
            "provider": self._provider_label,
            "model": request.model,
            "max_output_tokens": request.max_output_tokens,
            "reasoning_effort": request.reasoning_effort,
            "timeout": request.timeout,
        }
        if request.use_case is not None:
            data["use_case"] = request.use_case
        detailed_metadata = (
            {**(request.internal_metadata or {}), **(request.extra or {})}
            if request.include_payloads_in_observability
            else {}
        )

        span_attributes: dict[str, AttributeValue] = {
            "llm.provider": self._provider_label,
            "llm.model": request.model,
            "llm.grounded": bool(request.grounded),
        }
        if request.max_output_tokens is not None:
            span_attributes["llm.max_output_tokens"] = int(request.max_output_tokens)
        if request.reasoning_effort is not None:
            span_attributes["llm.reasoning_effort"] = str(request.reasoning_effort)

        tracer = trace.get_tracer("harnyx_commons.llm")
        with tracer.start_as_current_span(
            "llm.invoke",
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            record_exception=request.include_payloads_in_observability,
            set_status_on_exception=request.include_payloads_in_observability,
        ) as span:
            self._llm_logger.debug("llm.invoke.start", extra={"data": data | detailed_metadata})
            start = time.perf_counter()
            wait_ms = 0.0
            try:
                if self._semaphore is None or not acquire_semaphore:
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
                    "elapsed_ms": elapsed,
                    "wait_ms": round(wait_ms, 2),
                }
                if request.include_payloads_in_observability:
                    error_metadata["error"] = repr(exc)
                    raw_error_payload = _error_raw_payload_metadata(request=request, exc=exc)
                    if raw_error_payload is not None:
                        error_metadata["raw"] = raw_error_payload
                self._llm_logger.error(
                    "llm.invoke.error",
                    extra={"data": data | {"elapsed_ms": elapsed, "exception_type": type(exc).__name__}},
                )
                if request.include_payloads_in_observability:
                    self._llm_logger.debug(
                        "llm.invoke.error.details",
                        exc_info=True,
                        extra={"data": data | detailed_metadata | error_metadata},
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
            span.set_attributes(
                {
                    "llm.elapsed_ms": elapsed,
                    "llm.wait_ms": round(wait_ms, 2),
                    "llm.usage.total_tokens": int(usage.total_tokens or 0),
                    "llm.usage.prompt_tokens": int(usage.prompt_tokens or 0),
                    "llm.usage.completion_tokens": int(usage.completion_tokens or 0),
                    "llm.usage.web_search_calls": web_search_calls,
                }
            )
            if usage.reasoning_tokens is not None:
                span.set_attribute("llm.usage.reasoning_tokens", int(usage.reasoning_tokens))

            return response

    async def aclose(self) -> None:
        """Providers may override to close network clients."""
        return None

    @abstractmethod
    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        """Provider-specific invocation; implemented by concrete providers."""

    async def _annotate_response_cost(
        self,
        request: AbstractLlmRequest,
        response: LlmResponse,
    ) -> LlmResponse:
        """Attach provider-owned settled-cost metadata when available."""
        return response

    async def _call_with_retry(
        self,
        request: AbstractLlmRequest,
        *,
        call_coro: Callable[[AbstractLlmRequest], Awaitable[LlmResponse]],
        verifier: Callable[[LlmResponse], tuple[bool, bool, str | None]],
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
        policy: RetryPolicy | None = None,
    ) -> LlmResponse:
        """Execute LLM call with retry. Main orchestrator."""
        policy = policy or self._retry_policy
        ctx = RetryContext(policy)
        current_request = request
        try:
            for attempt in range(policy.attempts):
                # Phase 1: Attempt
                response = await self._try_call(attempt, ctx, current_request, call_coro, classify_exception)
                if response is None:
                    continue

                # Phase 2: Verify
                if not await self._try_verify(attempt, ctx, current_request, response, verifier):
                    continue

                # Phase 3: Postprocess
                processed, retry, next_request = await self._try_postprocess(attempt, ctx, current_request, response)
                if retry:
                    current_request = next_request or current_request
                    continue

                # Success
                return self._build_response(request, response, ctx, processed)

            raise LlmRetryExhaustedError(
                "LLM call exhausted all retry attempts",
                response=self._build_response_for_error(ctx.last_response, ctx),
                attempts=len(ctx.reasons) + 1,
                retry_reasons=tuple(ctx.reasons),
                latency_ms_total=round(ctx.total_latency_ms, 2),
            )
        except asyncio.CancelledError as exc:
            response = self._build_response_for_error(ctx.last_response, ctx)
            if response is not None:
                raise LlmInvocationCancelledError(response) from exc
            raise

    async def _try_call(
        self,
        attempt: int,
        ctx: RetryContext,
        request: AbstractLlmRequest,
        call_coro: Callable[[AbstractLlmRequest], Awaitable[LlmResponse]],
        classify_exception: Callable[[Exception], tuple[bool, str]] | None,
    ) -> LlmResponse | None:
        """Attempt the LLM call. Returns None if retry needed."""
        start = time.perf_counter()
        with observe_judge_llm_attempt(
            provider=self._provider_label,
            request=request,
            attempt_number=attempt + 1,
            attempt_limit=ctx.policy.attempts,
        ) as observation:
            evidence = _AttemptResponseObservation()
            evidence_token = _attempt_response.set(evidence)
            try:
                async with enforce_attempt_deadlines(resolve_timeout(request.timeout)):
                    response = await call_coro(request)
            except asyncio.CancelledError:
                if evidence.response is not None:
                    await self._record_attempt_response(request, evidence.response, ctx)
                if observation is not None:
                    observation.finish_cancelled()
                raise
            except Exception as exc:
                if evidence.response is not None:
                    await self._record_attempt_response(request, evidence.response, ctx)
                latency_ms = (time.perf_counter() - start) * 1000
                ctx.total_latency_ms += latency_ms
                retryable, reason = self._classify_retry_exception(exc, classify_exception)
                if observation is not None:
                    observation.finish_failure(exc, retryable=retryable)
                await self._handle_failure(
                    attempt,
                    ctx,
                    request,
                    "exception",
                    RetryFailureDetail(
                        reason=reason,
                        exception_type=type(exc).__name__,
                        exception_message=str(exc),
                        exception_repr=repr(exc),
                        cause_chain=_exception_cause_chain(exc),
                    ),
                    retryable,
                    source_exception=exc,
                )
                return None
            finally:
                _attempt_response.reset(evidence_token)
            if observation is not None:
                observation.finish_response(response)

        response = await self._record_attempt_response(request, response, ctx)
        latency_ms = (time.perf_counter() - start) * 1000
        ctx.total_latency_ms += latency_ms
        return response

    async def _record_attempt_response(
        self, request: AbstractLlmRequest, response: LlmResponse, ctx: RetryContext,
    ) -> LlmResponse:
        response = await self._annotate_response_cost(request, response)
        ctx.total_usage += response.usage or LlmUsage()
        ctx.record_billable_response(response)
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
            RetryFailureDetail(reason=reason or "verification_failed"),
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
    ) -> tuple[object | None, bool, AbstractLlmRequest | None]:
        """Postprocess response. Returns (processed, should_retry, next_request)."""
        if request.postprocessor is None:
            return None, False, None

        result = request.postprocessor(response)
        if result.ok:
            return result.processed, False, None

        next_request: AbstractLlmRequest | None = None
        if (
            request.allow_postprocess_recovery
            and result.recovery is not None
            and result.retryable
            and not ctx.is_exhausted(attempt)
        ):
            feedback_retry = self._build_postprocess_feedback_retry(
                request=request,
                response=response,
                recovery=result.recovery,
            )
            if feedback_retry is not None:
                next_request, recovery_event = feedback_retry
                ctx.recovery_events.append(recovery_event)

        await self._handle_failure(
            attempt,
            ctx,
            request,
            "postprocess",
            RetryFailureDetail(reason=result.reason or "postprocess_failed"),
            result.retryable,
            response=response,
        )
        return None, True, next_request

    def _build_postprocess_feedback_retry(
        self,
        *,
        request: AbstractLlmRequest,
        response: LlmResponse,
        recovery: PostprocessRecovery,
    ) -> tuple[AbstractLlmRequest, dict[str, object]] | None:
        if recovery.kind != "retry_with_feedback":
            return None

        failed_message = _failed_response_message(response)
        if failed_message is None:
            return None

        feedback_role = _feedback_role(request=request, response=response)
        retry_request = self._build_postprocess_feedback_request(
            request=request,
            failed_message=failed_message,
            feedback_role=feedback_role,
            failure_reason=recovery.failure_reason,
        )
        recovery_event = {
            "kind": recovery.kind,
            "response_id": response.id,
            "feedback_role": feedback_role,
        }
        self._llm_logger.info(
            "llm.recovery.postprocess.retry_scheduled",
            extra={
                "data": {
                    "provider": self._provider_label,
                    "model": request.model,
                    **recovery_event,
                }
            },
        )
        return retry_request, recovery_event

    def _build_postprocess_feedback_request(
        self,
        *,
        request: AbstractLlmRequest,
        failed_message: LlmMessage,
        feedback_role: Literal["user", "tool"],
        failure_reason: str,
    ) -> AbstractLlmRequest:
        updated_internal_metadata = dict(request.internal_metadata or {})
        updated_internal_metadata["postprocess_feedback_retry"] = True
        updated_internal_metadata["postprocess_feedback_depth"] = (
            int(updated_internal_metadata.get("postprocess_feedback_depth", 0)) + 1
        )
        base_messages = _base_messages_for_feedback_retry(request)
        feedback_message = LlmMessage(
            role=feedback_role,
            content=(
                LlmMessageContentPart.input_text(
                    "Your previous response failed the original output contract.\n\n"
                    f"Validation/parsing error:\n{failure_reason}\n\n"
                    "Correct your previous response so it follows the original instructions, "
                    "output contract, and formatting constraints. "
                    "Do not add extra wrapper text or commentary unless the original instructions required it."
                ),
            ),
        )
        return replace(
            request,
            messages=(*base_messages, failed_message, feedback_message),
            internal_metadata=updated_internal_metadata,
        )

    async def _handle_failure(
        self,
        attempt: int,
        ctx: RetryContext,
        request: AbstractLlmRequest,
        phase: str,
        failure: RetryFailureDetail,
        retryable: bool,
        response: LlmResponse | None = None,
        source_exception: Exception | None = None,
    ) -> None:
        """Handle a phase failure - either raise or prepare for retry."""
        response_for_error = response if response is not None else ctx.last_response
        if retryable and ctx.is_exhausted(attempt):
            raise LlmRetryExhaustedError(
                failure.reason,
                response=self._build_response_for_error(response_for_error, ctx),
                attempts=len(ctx.reasons) + 1,
                retry_reasons=tuple(ctx.reasons),
                latency_ms_total=round(ctx.total_latency_ms, 2),
            ) from source_exception
        if not retryable:
            if source_exception is not None:
                accumulated_response = self._build_response_for_error(response_for_error, ctx)
                if isinstance(source_exception, LlmProviderError):
                    if source_exception.response is None:
                        source_exception.response = accumulated_response
                    raise source_exception
                raise LlmProviderError(failure.reason, response=accumulated_response) from source_exception
            raise LlmProviderError(failure.reason, response=self._build_response_for_error(response_for_error, ctx))
        ctx.reasons.append(failure.reason)
        self._log_retry(phase, request, attempt, failure, ctx.policy)
        await asyncio.sleep(self._retry_delay_seconds(attempt, ctx.policy, source_exception))

    def _classify_retry_exception(
        self,
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None,
    ) -> tuple[bool, str]:
        if isinstance(exc, LlmAttemptTimeoutError):
            return True, f"{exc.phase}_timeout"
        return classify_exception(exc) if classify_exception else (False, str(exc))

    def _retry_delay_seconds(self, attempt: int, policy: RetryPolicy, exc: Exception | None) -> float:
        return backoff_ms(attempt, policy) / 1000

    def _build_response(
        self,
        request: AbstractLlmRequest,
        response: LlmResponse,
        ctx: RetryContext,
        processed: object | None,
    ) -> LlmResponse:
        """Construct final response with accumulated metadata."""
        self._log_retry_complete(request=request, response=response, ctx=ctx)
        return self._build_accumulated_response(response, ctx, processed)

    def _build_response_for_error(self, response: LlmResponse | None, ctx: RetryContext) -> LlmResponse | None:
        if response is None:
            return None
        return self._build_accumulated_response(response, ctx, response.postprocessed)

    def _build_accumulated_response(
        self,
        response: LlmResponse,
        ctx: RetryContext,
        processed: object | None,
    ) -> LlmResponse:
        metadata = dict(response.metadata or {})
        metadata["attempts"] = len(ctx.reasons) + 1
        metadata["billable_response_count"] = ctx.billable_response_count
        actual_cost_total = ctx.actual_cost_usd_total
        if actual_cost_total is not None:
            metadata["actual_cost_usd_total"] = actual_cost_total
        else:
            metadata.pop("actual_cost_usd_total", None)
        metadata["retry_reasons"] = tuple(ctx.reasons)
        metadata["latency_ms_total"] = round(ctx.total_latency_ms, 2)
        if ctx.recovery_events:
            metadata["postprocess_recoveries"] = tuple(ctx.recovery_events)

        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=ctx.total_usage,
            metadata=metadata,
            postprocessed=processed,
            finish_reason=response.finish_reason,
        )

    def _log_retry_complete(self, *, request: AbstractLlmRequest, response: LlmResponse, ctx: RetryContext) -> None:
        json_fields = (
            {
                "request": _request_snapshot(request),
                "response": response.payload,
                "response_raw": (response.metadata or {}).get("raw_response"),
            }
            if request.include_payloads_in_observability
            else None
        )
        metadata = {
            "provider": request.provider,
            "model": request.model,
            "attempts": len(ctx.reasons) + 1,
            "latency_ms_total": round(ctx.total_latency_ms, 2),
            "retry_count": len(ctx.reasons),
            "postprocess_recovery_count": len(ctx.recovery_events),
            "usage": _usage_snapshot(ctx.total_usage),
            "actual_cost_usd_total": ctx.actual_cost_usd_total,
            "billable_response_count": ctx.billable_response_count,
        }
        self._llm_logger.info(
            "llm.invoke.retry.complete",
            extra={"data": metadata},
        )
        if json_fields is not None:
            self._llm_logger.debug(
                "llm.invoke.retry.complete.details",
                extra={
                    "data": metadata
                    | {
                        "retry_reasons": tuple(ctx.reasons),
                        "postprocess_recoveries": tuple(ctx.recovery_events),
                    },
                    "json_fields": json_fields,
                },
            )

    def _log_retry(
        self,
        phase: str,
        request: AbstractLlmRequest,
        attempt: int,
        failure: RetryFailureDetail,
        policy: RetryPolicy,
    ) -> None:
        """Unified retry event logger."""
        data: dict[str, object] = {
            "provider": self._provider_label,
            "model": request.model,
            "attempt": attempt + 1,
            "reason": "provider_retry",
            "backoff_ms": backoff_ms(attempt, policy),
        }
        if request.use_case is not None:
            data["use_case"] = request.use_case
        if failure.exception_type is not None:
            data["exception_type"] = failure.exception_type
        message = f"llm.retry.{phase}"
        self._llm_logger.warning(
            message,
            extra={"data": data},
        )
        if request.include_payloads_in_observability:
            self._llm_logger.debug(
                f"{message}.details",
                extra={
                    "data": data
                    | {
                        "reason": failure.reason,
                        "exception_message": failure.exception_message,
                        "exception_repr": failure.exception_repr,
                        "cause_chain": failure.cause_chain,
                    }
                },
            )


def _request_snapshot(request: AbstractLlmRequest) -> dict[str, object]:
    if is_dataclass(request):
        snapshot = cast(dict[str, object], asdict(request))
        redacted = _redact_tool_auth_secrets(snapshot)
        if not isinstance(redacted, dict):
            raise TypeError("llm request snapshot redaction must return dict")
        return cast(dict[str, object], redacted)
    raise TypeError("llm request snapshot requires dataclass request")


def _usage_snapshot(usage: LlmUsage) -> dict[str, object]:
    snapshot = {
        "prompt": usage.prompt_tokens or 0,
        "prompt_cached": usage.prompt_cached_tokens or 0,
        "completion": usage.completion_tokens or 0,
        "total": usage.total_tokens or 0,
        "web_search_calls": usage.web_search_calls or 0,
    }
    if usage.reasoning_tokens is not None:
        snapshot["reasoning"] = usage.reasoning_tokens
    return snapshot


def _exception_cause_chain(exc: BaseException) -> tuple[str, ...]:
    chain: list[str] = []
    current = exc.__cause__ or exc.__context__
    seen: set[int] = set()
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        chain.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return tuple(chain)


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
    if not isinstance(exc, (LlmProviderError, LlmRetryExhaustedError)):
        return None

    response = exc.response
    if response is None:
        return None
    response_metadata = _instrumentation_response_metadata(response)
    return _build_raw_payload_metadata(
        request=request,
        response=response,
        response_metadata=response_metadata,
    )


def _instrumentation_response_metadata(response: LlmResponse) -> dict[str, object]:
    metadata = dict(response.metadata or {})
    metadata.pop("ttft_ms", None)
    return metadata


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


def _failed_response_message(response: LlmResponse) -> LlmMessage | None:
    if not response.choices:
        return None
    message = response.choices[0].message
    text_parts = tuple(
        LlmMessageContentPart.input_text(part.text)
        for part in message.content
        if isinstance(part.text, str) and part.text.strip()
    )
    if not text_parts:
        return None

    response_role = message.role
    role: Literal["assistant", "tool"]
    if response_role == "tool":
        role = "tool"
    else:
        role = "assistant"
    return LlmMessage(role=role, content=text_parts)


def _feedback_role(*, request: AbstractLlmRequest, response: LlmResponse) -> Literal["user", "tool"]:
    if not response.choices:
        return "user"
    response_role = response.choices[0].message.role
    if response_role != "tool":
        return "user"
    if supports_tool_result_messages(provider=request.provider, model=request.model):
        return "tool"
    return "user"


def _base_messages_for_feedback_retry(request: AbstractLlmRequest) -> tuple[LlmMessage, ...]:
    metadata = request.internal_metadata or {}
    if not metadata.get("postprocess_feedback_retry"):
        return tuple(request.messages)
    if len(request.messages) < 2:
        return tuple(request.messages)
    return tuple(request.messages[:-2])


_TOOL_SECRET_KEYS = frozenset(
    {
        "api_key_string",
        "apiKeyString",
    }
)


def _redact_tool_auth_secrets(value: object) -> object:
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in _TOOL_SECRET_KEYS and isinstance(child, str):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_tool_auth_secrets(child)
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_tool_auth_secrets(item) for item in value]
    return value


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
