"""Provider calls, in-memory author history and exposed streaming diagnostics."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from time import monotonic, time
from typing import Any, TypeVar, cast

import httpx
import httpx2
from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner, WebSearchTool
from agents.exceptions import ModelBehaviorError
from agents.items import TResponseInputItem
from agents.result import RunResultStreaming
from agents.stream_events import RawResponsesStreamEvent, StreamEvent
from google.auth import exceptions as google_auth_errors
from google.genai import errors as google_errors
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError
from openai.types.responses import ResponseErrorEvent, ResponseFailedEvent
from openai.types.shared import Reasoning
from pydantic import BaseModel

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
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment, build_fast_judge_messages, calculate_fast_f1
from harnyx_commons.observability.langfuse import start_llm_generation, update_generation_best_effort
from harnyx_commons.task_ownership import wait_for_owned_task

from .contracts import (
    AgentResult,
    BatchTerminalGenerationError,
    CandidateSession,
    CandidateStageError,
    GenerationStageSummary,
    QuestionDraft,
    ReviewDecision,
    Seed,
    Seeds,
    StructuredQuestionDraft,
    TerminalReference,
)
from .prompts import role_prompt
from .source_fetch import PublicSourceFetcher

logger = logging.getLogger("harnyx_commons.llm.calls")
AUTHOR_MODEL = "gpt-6-luna"
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
            if provider in {"vertex", "openai"} or usage.web_search_calls
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
        return exc.code in {"server_error", "rate_limit_exceeded"}
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
                            stage="reference" if capture.role == "reference" else "question_generation",
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


async def record_openai_request(request: httpx2.Request) -> None:
    capture = ACTIVE_CAPTURE.get()
    if capture is not None and request.url.path.endswith("/responses"):
        payload = exposed(json.loads(request.content))
        capture.requests.append(payload)
        logger.debug("task_generation.provider_request", extra={"data": capture.identity() | {"request": payload}})


@asynccontextmanager
async def openai_stream(result: RunResultStreaming) -> AsyncIterator[AsyncIterator[StreamEvent]]:
    """Settle SDK work before a failed attempt retries or releases its slot."""
    # The SDK annotates this as AsyncIterator, but implements an async generator with aclose.
    events = cast(AsyncGenerator[StreamEvent, None], result.stream_events())
    try:
        yield events
    finally:
        failed = sys.exc_info()[0] is not None
        if failed:
            result.cancel()

        async def close() -> None:
            try:
                await events.aclose()
            finally:
                if result.run_loop_task is not None:
                    await wait_for_owned_task(result.run_loop_task)
                    if not result.run_loop_task.cancelled():
                        result.run_loop_task.exception()

        closing = asyncio.create_task(close())
        cancellation = await wait_for_owned_task(closing)
        try:
            closing.result()
        except Exception:
            if not failed:
                raise
            logger.debug("task_generation.provider_cleanup.failure.details", exc_info=True)
        if cancellation is not None:
            raise cancellation


class GenerationAgentRunner:
    def __init__(
        self, *, project_id: str | None, judge: LlmProviderPort, openai_client: AsyncOpenAI | None = None
    ) -> None:
        self._slots = asyncio.Semaphore(2)
        self._source_fetcher = PublicSourceFetcher()
        self._judge = judge
        self._project_id = project_id
        self._client = openai_client
        self._http: httpx2.AsyncClient | None = None

    async def _openai(self) -> AsyncOpenAI:
        if self._client is None:
            client = httpx2.AsyncClient(event_hooks={"request": [record_openai_request]}, timeout=600)
            try:
                self._client = AsyncOpenAI(http_client=client, max_retries=0)
            except BaseException as exc:
                await client.aclose()
                if isinstance(exc, OpenAIError):
                    raise BatchTerminalGenerationError(
                        "provider_configuration",
                        "OpenAI initialization failed; check provider credentials",
                        stage="question_generation",
                    ) from exc
                raise
            self._http = client
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._http is not None:
            await self._http.aclose()
        await self._judge.aclose()

    async def invoke(self, role: str, message: str, session: CandidateSession, deadline: float) -> AgentResult:
        if role in {"solver", "grounded_solver"}:
            from .solver_runner import solve

            return await solve(role, message, session, deadline, project_id=self._project_id)
        async with asyncio.timeout(max(0, deadline - monotonic())):
            async with self._slots:
                return await self._invoke_openai(role, message, session, deadline)

    async def _invoke_openai(self, role: str, message: str, session: CandidateSession, deadline: float) -> AgentResult:
        capture = CallCapture(session, role, AUTHOR_MODEL, "openai")
        instruction = ("" if role == "seed" else temporal_instruction(session) + "\n") + role_prompt(role, session.mode)
        output_type = (
            Seeds
            if role == "seed"
            else ReviewDecision
            if role in {"reviewer", "verifier"}
            else StructuredQuestionDraft
            if role == "author" and session.mode == "structured"
            else QuestionDraft
            if role == "author"
            else None
        )
        agent = Agent(
            name=role,
            instructions=instruction,
            model=OpenAIResponsesModel(model=AUTHOR_MODEL, openai_client=await self._openai()),
            tools=[WebSearchTool()],
            output_type=output_type,
            model_settings=ModelSettings(
                reasoning=Reasoning(
                    effort="max" if role == "verifier" else "high", summary="auto" if role == "seed" else "detailed"
                ),
                truncation="disabled",
                response_include=["web_search_call.action.sources"],
            ),
        )
        request = LlmRequest(
            provider="openai",
            model=AUTHOR_MODEL,
            temperature=None,
            max_output_tokens=None,
            messages=(
                LlmMessage(role="system", content=(LlmMessageContentPart.input_text(instruction),)),
                LlmMessage(role="user", content=(LlmMessageContentPart.input_text(message),)),
            ),
            use_case=f"miner_task_generation.{role}",
            reasoning_effort="max" if role == "verifier" else "high",
        )
        author_input: str | list[TResponseInputItem] = (
            cast(list[TResponseInputItem], [*session.author_history, {"role": "user", "content": message}])
            if role == "author"
            else message
        )

        async def call() -> str:
            result = Runner.run_streamed(agent, author_input, max_turns=1, run_config=RunConfig(tracing_disabled=True))
            latest_usage: LlmUsage | None = None
            try:
                async with openai_stream(result) as events:
                    async for event in events:
                        if isinstance(event, RawResponsesStreamEvent):
                            data = event.data.model_dump(mode="json", exclude_none=True)
                            capture.event(data)
                            if data["type"] in {"response.completed", "response.incomplete", "response.failed"}:
                                raw = data["response"].get("usage")
                                if raw is not None:
                                    latest_usage = LlmUsage(
                                        prompt_tokens=raw["input_tokens"],
                                        completion_tokens=raw["output_tokens"],
                                        total_tokens=raw["total_tokens"],
                                        prompt_cached_tokens=raw.get("input_tokens_details", {}).get("cached_tokens"),
                                        prompt_cache_write_tokens=raw.get("input_tokens_details", {}).get(
                                            "cache_write_tokens"
                                        ),
                                        reasoning_tokens=raw.get("output_tokens_details", {}).get("reasoning_tokens"),
                                        web_search_calls=data["response"]
                                        .get("tool_usage", {})
                                        .get("web_search", {})
                                        .get("num_requests"),
                                    )
                            if data["type"] in {"response.failed", "response.incomplete", "error"}:
                                code = None
                                if (
                                    isinstance(event.data, ResponseFailedEvent)
                                    and event.data.response.error is not None
                                ):
                                    code = event.data.response.error.code
                                elif isinstance(event.data, ResponseErrorEvent):
                                    code = event.data.code
                                raise OpenAIStreamError(code)
                final = result.final_output
                answer = final.model_dump_json(by_alias=True) if isinstance(final, BaseModel) else final
                if not isinstance(answer, str) or not answer.strip():
                    raise ModelBehaviorError("Agent did not return a complete answer")
                if role == "author":
                    session.author_history = result.to_input_list()
                return answer
            finally:
                capture.account(latest_usage)

        token = ACTIVE_CAPTURE.set(capture)
        try:
            with start_llm_generation(provider_label="openai", request=request) as generation:
                text = await retry_request(call, capture, deadline, solver=False)
                answer = AgentResult(
                    answer=text,
                    request={"message": message, "instruction": instruction, "model": AUTHOR_MODEL},
                    provider_responses=capture.responses,
                )
                update_generation_best_effort(
                    generation,
                    output=answer.model_dump(mode="json"),
                    usage_details={
                        "input": capture.tool_usage.llm.prompt_tokens,
                        "output": capture.tool_usage.llm.completion_tokens,
                    },
                    metadata=capture.identity(),
                )
                logger.info(
                    "task_generation.stage.completed",
                    extra={
                        "data": capture.identity()
                        | {"attempts": capture.attempted, "usage_missing": capture.attempted - capture.usage_records}
                    },
                )
                return answer
        finally:
            ACTIVE_CAPTURE.reset(token)
            capture.finish()

    async def generate_seeds(
        self,
        count: int,
        session: CandidateSession,
        deadline: float,
        *,
        start: date,
        end: date,
        previous_seeds: list[Seed],
    ) -> list[Seed]:
        packet = {
            "requested_count": count,
            "previous_seeds": [seed.model_dump() for seed in previous_seeds],
            "as_of_date": str(session.effective_date),
            "event_date_window": {"start": str(start), "end": str(end)},
        }
        result = await self.invoke("seed", json.dumps(packet, ensure_ascii=False), session, deadline)
        seeds = Seeds.model_validate_json(result.answer).seeds
        if len(seeds) != count or len({seed.seed_id for seed in seeds}) != count:
            raise ValueError("Seed count or identifiers differ from requested contract")
        if len({seed.seed_question for seed in seeds}) != count:
            raise ValueError("Duplicate seed questions")
        for seed in seeds:
            if not start <= date.fromisoformat(seed.event_date) <= end:
                raise ValueError("Seed event outside requested window")
        return seeds

    async def assess(
        self, draft: QuestionDraft, result: AgentResult, session: CandidateSession, deadline: float
    ) -> FastJudgeAssessment:
        return await self._assess_response(draft, Response(text=result.answer), session, deadline)

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

    async def derive_reference(
        self, draft: QuestionDraft, verification: ReviewDecision | None, session: CandidateSession, deadline: float
    ) -> TerminalReference:
        from .reference_generation import derive_reference

        reference = await derive_reference(
            draft,
            verification,
            session,
            deadline,
            project_id=self._project_id,
            fetcher=self._source_fetcher,
        )
        if (
            verification is not None
            and verification.passed
            and not isinstance(draft, StructuredQuestionDraft)
            and reference.reference_answer is not None
            and not reference.factual_correction
        ):
            answer = reference.reference_answer
            assessment = await self._assess_response(
                draft,
                Response(text=answer.text, note=answer.note),
                session,
                deadline,
                role="reference_assessment",
            )
            score = calculate_fast_f1(assessment)
            logger.info(
                "task_generation.reference.assessed",
                extra={"data": {"task_id": session.task_id, "f1": score}},
            )
            if score < 1:
                reference = reference.model_copy(
                    update={
                        "unsupported_reason": "Terminal reference disagrees with the verified proposal",
                    }
                )
        return reference
