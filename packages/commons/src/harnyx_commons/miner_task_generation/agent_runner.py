"""Generation provider calls, accounting and exposed streaming diagnostics."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from time import monotonic, time
from typing import Any, TypeVar

import httpx
import httpx2
from agents.exceptions import ModelBehaviorError
from google.auth import exceptions as google_auth_errors
from google.genai import errors as google_errors
from openai import APIConnectionError, APIStatusError

from harnyx_commons.domain.miner_task import Query, ReferenceAnswer, Response
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import (
    known_zero_actual_cost_tool_usage,
    merge_complete_actual_cost_usage,
    tool_usage_from_llm_usage,
)
from harnyx_commons.llm.cost_settlement import settled_generation_llm_cost
from harnyx_commons.llm.provider import LlmProviderConfigurationError, LlmProviderError, LlmProviderPort
from harnyx_commons.llm.retry_utils import RetryPolicy, backoff_ms
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmUsage
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment, build_fast_judge_messages

from .contracts import (
    AgentResult,
    BatchTerminalGenerationError,
    CandidateSession,
    CandidateStageError,
    GenerationStageSummary,
    QuestionDraft,
    TerminalReference,
)

logger = logging.getLogger("harnyx_commons.llm.calls")
AUTHOR_MODEL = "gpt-6.1-sol"
JUDGE_MODEL = "gemini-3.1-pro-preview"
RETRY_POLICY = RetryPolicy(attempts=2, initial_ms=500, max_ms=30_000, jitter=0.2)
T = TypeVar("T")
_REQUEST_DATE: ContextVar[date | None] = ContextVar("generation_request_date", default=None)
_ATTEMPT_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar("generation_attempt_events", default=None)


class OpenAIStreamError(RuntimeError):
    """A failed Responses stream carries a provider error code, not an HTTP status."""

    def __init__(self, code: str | None) -> None:
        super().__init__("OpenAI response was incomplete or failed")
        self.code = code


def temporal_instruction(session: CandidateSession) -> str:
    return (
        f"Public run context: current date {session.effective_date}; timezone UTC; explicit historical "
        "replay/as-of context: none. Question target dates, publication dates and event dates are distinct; "
        "do not replace them with the run date."
    )


def exposed(value: Any) -> Any:
    """Encrypted reasoning is state, not readable evidence; credentials are never submitted here."""
    if isinstance(value, dict):
        return {key: exposed(item) for key, item in value.items() if key != "encrypted_content"}
    if isinstance(value, list):
        return [exposed(item) for item in value]
    return value


@dataclass
class CallCapture:
    session: CandidateSession
    role: str
    model: str
    provider: str
    responses: list[dict[str, Any]] = field(default_factory=list)
    workers: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    tool_usage: ToolUsageSummary = field(default_factory=known_zero_actual_cost_tool_usage)
    attempted: int = 0
    usage_records: int = 0
    started: float = field(default_factory=monotonic)
    available_cost_usd: float = 0

    def identity(self) -> dict[str, object]:
        return {
            "task_id": self.session.task_id,
            "cycle": self.session.cycle,
            "role": self.role,
            "author_attempt": self.session.attempt,
            "provider": self.provider,
            "model": self.model,
        }

    def event(self, event: dict[str, Any]) -> int:
        event = exposed(event)
        envelope = {"provider": self.provider, "response": event}
        self.responses.append(envelope)
        attempt_events = _ATTEMPT_EVENTS.get()
        if attempt_events is not None:
            attempt_events.append(envelope)
        logger.debug("task_generation.provider_event", extra={"data": self.identity() | {"event": event}})
        return len(self.responses) - 1

    @contextmanager
    def attempt(self) -> Iterator[None]:
        events: list[dict[str, Any]] = []
        token = _ATTEMPT_EVENTS.set(events)
        status = "completed"
        try:
            yield
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            raise
        finally:
            for envelope in events:
                envelope["attempt_status"] = status
            _ATTEMPT_EVENTS.reset(token)
            logger.debug(
                "task_generation.provider_attempt",
                extra={"data": self.identity() | {"attempt_status": status, "event_count": len(events)}},
            )

    def account(self, usage: LlmUsage | None, *, model: str | None = None, provider: str | None = None) -> None:
        if usage is None:
            return
        self.usage_records += 1
        model, provider = model or self.model, provider or self.provider
        pricing_date = _REQUEST_DATE.get() or datetime.now(UTC).date()
        summary = tool_usage_from_llm_usage(usage, model=model, provider=provider, pricing_date=pricing_date)
        search_cost = (
            summary.search_tool.reference_cost
            if provider in {"vertex", "openai"} and usage.web_search_calls is not None
            else None
            if provider in {"vertex", "openai", "codex-lb"} or usage.web_search_calls
            else 0.0
        )
        summary = replace(summary, search_tool=replace(summary.search_tool, actual_cost=search_cost))
        self.available_cost_usd += search_cost or 0
        settled = settled_generation_llm_cost(usage=usage, model=model, provider=provider, pricing_date=pricing_date)
        if settled is not None:
            self.available_cost_usd += settled.cost_usd
            total_cost = settled.cost_usd + (search_cost or 0)
            summary = replace(
                summary,
                llm=replace(
                    summary.llm,
                    actual_cost=settled.cost_usd,
                    providers={
                        provider: {model: replace(summary.llm.providers[provider][model], actual_cost=settled.cost_usd)}
                    },
                ),
                search_tool=replace(summary.search_tool, actual_cost=search_cost),
                actual_total_cost_usd=total_cost,
                actual_cost_by_provider={provider: total_cost},
            )
            if provider in {"vertex", "openai"} and usage.web_search_calls is None:
                # Missing receipts cannot establish free native search.
                summary = replace(summary, actual_total_cost_usd=None, actual_cost_by_provider={})
        self.tool_usage = merge_complete_actual_cost_usage(self.tool_usage, summary)

    def finish(self) -> None:
        missing = max(0, self.attempted - self.usage_records)
        if missing:
            self.tool_usage = replace(self.tool_usage, actual_total_cost_usd=None, actual_cost_by_provider={})
        self.session.tool_usage = merge_complete_actual_cost_usage(self.session.tool_usage, self.tool_usage)
        self.session.attempted_calls += self.attempted
        self.session.missing_usage_calls += max(0, self.attempted - self.usage_records)
        summary = GenerationStageSummary(
            stage=self.role,
            outcome="failed" if sys.exc_info()[0] is not None else "completed",
            elapsed_ms=(monotonic() - self.started) * 1000,
            provider=self.provider,
            model=self.model,
            tool_usage=self.tool_usage,
            attempted_calls=self.attempted,
            missing_usage_calls=missing,
            available_cost_usd=self.available_cost_usd,
        )
        self.session.stage_summaries.append(summary)
        logger.info(
            "task_generation.stage.summary",
            extra={
                "data": self.identity()
                | {
                    "outcome": summary.outcome,
                    "elapsed_ms": summary.elapsed_ms,
                    "attempted_calls": self.attempted,
                    "missing_usage_calls": missing,
                    "actual_total_cost_usd": self.tool_usage.actual_total_cost_usd,
                }
            },
        )


ACTIVE_CAPTURE: ContextVar[CallCapture | None] = ContextVar("generation_call_capture", default=None)


def provider_status(exc: Exception) -> int | None:
    if isinstance(exc, LlmProviderError) and isinstance(exc.__cause__, Exception):
        return provider_status(exc.__cause__)
    status = getattr(exc, "status_code", getattr(exc, "code", None))
    return status if isinstance(status, int) else None


def transient_error(exc: Exception) -> bool:
    if isinstance(exc, ModelBehaviorError):
        return True
    if isinstance(exc, OpenAIStreamError):
        return exc.code in {
            "server_error",
            "rate_limit_exceeded",
            "server_is_overloaded",
            "upstream_stream_truncated",
        }
    if isinstance(exc, LlmProviderError) and isinstance(exc.__cause__, Exception):
        return transient_error(exc.__cause__)
    if isinstance(exc, CandidateStageError):
        return exc.failure_class == "transient_provider"
    if isinstance(exc, google_auth_errors.TransportError):
        return True
    if isinstance(exc, google_auth_errors.GoogleAuthError):
        return exc.retryable
    if isinstance(exc, (APIConnectionError, httpx.TransportError, httpx2.TransportError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    if isinstance(exc, google_errors.APIError):
        return exc.code in {408, 409, 429} or exc.code >= 500
    return False


def shared_provider_failure(exc: Exception) -> str | None:
    """Classify faults that invalidate every candidate, including local credentials."""
    if isinstance(exc, LlmProviderConfigurationError):
        return "provider_configuration"
    if isinstance(exc, LlmProviderError) and isinstance(exc.__cause__, Exception):
        return shared_provider_failure(exc.__cause__)
    if isinstance(
        exc,
        (
            google_auth_errors.DefaultCredentialsError,
            google_auth_errors.RefreshError,
            google_auth_errors.UserAccessTokenError,
        ),
    ) and not transient_error(exc):
        return "provider_auth"
    status = provider_status(exc)
    if status in {401, 403}:
        return "provider_auth"
    if status == 404:
        return "provider_configuration"
    return None


def retry_after(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("retry-after")
    if value is None:
        return getattr(exc, "retry_after_seconds", None) or 0
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time())
        except (ValueError, TypeError, OverflowError):
            return 0


async def retry_request(call: Callable[[], Awaitable[T]], capture: CallCapture, deadline: float, *, solver: bool) -> T:
    async with asyncio.timeout(max(0, deadline - monotonic())):
        attempt = 0
        while True:
            if monotonic() >= deadline:
                raise TimeoutError("Generation deadline reached before provider request")
            capture.attempted += 1
            pricing_token = _REQUEST_DATE.set(datetime.now(UTC).date())
            logger.info(
                "task_generation.request.started", extra={"data": capture.identity() | {"attempt": attempt + 1}}
            )
            try:
                with capture.attempt():
                    return await call()
            except Exception as exc:
                transient = transient_error(exc)
                failure = {
                    "attempt": attempt + 1,
                    "exception_type": type(exc).__name__,
                    "failure_class": "transient_provider" if transient else "provider_or_contract",
                    "provider_status": provider_status(exc),
                }
                if not transient or (not solver and attempt >= 1):
                    logger.error("task_generation.request.failed", extra={"data": capture.identity() | failure})
                    logger.debug(
                        "task_generation.request.failure.details", exc_info=True, extra={"data": capture.identity()}
                    )
                    status = provider_status(exc)
                    failure_class = shared_provider_failure(exc)
                    if failure_class is not None:
                        raise BatchTerminalGenerationError(
                            failure_class,
                            f"Shared {capture.provider} failure during {capture.role}: status {status}",
                            stage="question_generation",
                            tool_usage=capture.tool_usage,
                        ) from exc
                    raise
                wait = max(backoff_ms(min(attempt, 10), RETRY_POLICY) / 1000, retry_after(exc))
                logger.warning(
                    "task_generation.request.retry",
                    extra={"data": capture.identity() | failure | {"backoff_seconds": wait}},
                )
                logger.debug("task_generation.request.retry.details", exc_info=True, extra={"data": capture.identity()})
                await asyncio.sleep(wait)
                attempt += 1
            finally:
                _REQUEST_DATE.reset(pricing_token)


class GenerationAgentRunner:
    """Identical Luna challenges and the existing Gemini Pro assessments."""

    def __init__(
        self, *, judge: LlmProviderPort, solve: Callable[[str, str, CandidateSession, float], Awaitable[AgentResult]]
    ) -> None:
        self._judge, self._solve = judge, solve

    async def aclose(self) -> None:
        await self._judge.aclose()

    async def invoke(self, role: str, message: str, session: CandidateSession, deadline: float) -> AgentResult:
        if role not in {"solver", "grounded_solver"}:
            raise ValueError("Only difficulty solver roles are available in this runner")
        return await self._solve(role, message, session, deadline)

    async def assess(
        self, draft: QuestionDraft, result: AgentResult, session: CandidateSession, deadline: float
    ) -> FastJudgeAssessment:
        return await self._assess_response(draft, Response(text=result.answer), session, deadline)

    async def assess_reference(
        self, draft: QuestionDraft, reference: TerminalReference, session: CandidateSession, deadline: float
    ) -> FastJudgeAssessment:
        answer = reference.reference_answer
        if answer is None:
            raise ValueError("Reference consistency requires a rendered answer")
        return await self._assess_response(
            draft,
            Response(text=answer.text, note=answer.note),
            session,
            deadline,
            role="reference_consistency",
        )

    async def _assess_response(
        self,
        draft: QuestionDraft,
        response: Response,
        session: CandidateSession,
        deadline: float,
        *,
        role: str = "assessment",
    ) -> FastJudgeAssessment:
        messages = build_fast_judge_messages(
            query=Query(text=draft.question),
            reference_answer=ReferenceAnswer(
                text=draft.answer,
                note="Essential components: " + json.dumps(draft.answer_components, ensure_ascii=False),
            ),
            miner_response=response,
        )
        request = LlmRequest(
            provider="vertex",
            model=JUDGE_MODEL,
            temperature=None,
            max_output_tokens=None,
            reasoning_effort="high",
            output_mode="structured",
            output_schema=FastJudgeAssessment,
            use_case=f"miner_task_generation.{role}",
            retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0),
            messages=(
                LlmMessage(
                    role="system",
                    content=(
                        LlmMessageContentPart.input_text(
                            messages.system_prompt
                            + "\nFor this generation assessment, judge essential factual content independently "
                            "of serialization. Ignore JSON/schema formatting errors; do not turn ancillary excess "
                            "into an expected-component miss."
                        ),
                    ),
                ),
                LlmMessage(role="user", content=(LlmMessageContentPart.input_text(messages.user_prompt),)),
            ),
        )
        capture = CallCapture(session, role, JUDGE_MODEL, "vertex")

        async def call() -> FastJudgeAssessment:
            try:
                response = await self._judge.invoke(request)
            except LlmProviderError as exc:
                if exc.response is not None:
                    capture.account(exc.response.usage)
                raise
            capture.account(response.usage)
            if response.raw_text is None:
                raise ValueError("Judge returned no text")
            return FastJudgeAssessment.model_validate_json(response.raw_text)

        try:
            return await retry_request(call, capture, deadline, solver=False)
        finally:
            capture.finish()
