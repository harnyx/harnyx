"""Google ADK execution boundary for source-aware domain-tweak stages."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.genai.errors import ClientError
from pydantic import BaseModel

from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries
from harnyx_commons.domain_tweak_generation.adk_events import (
    final_text_from_event,
    summarize_adk_event,
    tool_usage_from_adk_events,
)
from harnyx_commons.domain_tweak_generation.prompts import feedback_prompt
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkAttempt,
    DomainTweakAdkEventSummary,
    DomainTweakAdkPhase,
    DomainTweakAdkPhaseResult,
    DomainTweakAdkPromptKind,
    DomainTweakAdkRunConfig,
    DomainTweakAdkTerminalStatus,
    DomainTweakValidationOutcome,
)
from harnyx_commons.errors import ToolProviderError, is_tool_provider_credential_failure

ValidationFunction = Callable[[str], DomainTweakValidationOutcome]
FunctionToolCallable = Callable[..., object]
_MAX_RETAINED_EVENT_SUMMARIES = 200
_SOURCE_TOOL_NAMES = frozenset(("acquire_sources", "read_cached_source"))
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DomainTweakAdkTurn:
    final_text: str
    events: tuple[DomainTweakAdkEventSummary, ...]


class DomainTweakAdkTurnExecutor(Protocol):
    async def __call__(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        attempt_index: int,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
        search_enabled: bool,
        function_tool_names: tuple[str, ...],
        output_schema: type[BaseModel],
    ) -> DomainTweakAdkTurn:
        """Run one ADK turn. Tests use this to avoid live Google APIs."""
        del function_tool_names
        raise NotImplementedError


class DomainTweakAdkRunner:
    """Runs one native-schema ADK stage inside one bounded session."""

    def __init__(self, *, turn_executor: DomainTweakAdkTurnExecutor | None = None) -> None:
        self._turn_executor = turn_executor

    async def run_phase(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
        search_enabled: bool,
        function_tools: Sequence[FunctionToolCallable],
        output_schema: type[BaseModel],
        validate: ValidationFunction,
    ) -> DomainTweakAdkPhaseResult:
        started = time.perf_counter()
        deadline = started + config.phase_timeout_seconds
        attempts: list[DomainTweakAdkAttempt] = []
        total_usage = ToolUsageSummary.zero()
        live_context: _LiveAdkContext | None = None
        if self._turn_executor is None:
            try:
                live_context = await asyncio.wait_for(
                    _LiveAdkContext.create(
                        phase=phase,
                        config=config,
                        agent_instruction=agent_instruction,
                        search_enabled=search_enabled,
                        function_tools=function_tools,
                        output_schema=output_schema,
                    ),
                    timeout=_remaining_timeout(deadline),
                )
            except TimeoutError as exc:
                return DomainTweakAdkPhaseResult(
                    phase=phase,
                    terminal_status="timeout",
                    attempts=(),
                    tool_usage=total_usage,
                    elapsed_ms=_elapsed_ms(started),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        try:
            attempt_index = 0
            while True:
                prompt_kind: DomainTweakAdkPromptKind = "initial" if attempt_index == 0 else "feedback"
                turn_prompt = (
                    prompt
                    if prompt_kind == "initial"
                    else feedback_prompt(attempts[-1].validation_feedback)
                )
                remaining_timeout = deadline - time.perf_counter()
                if remaining_timeout <= 0:
                    raise TimeoutError("ADK stage timeout exceeded before validation retry")
                turn_events: list[DomainTweakAdkEventSummary] = []
                try:
                    turn = await asyncio.wait_for(
                        self._run_turn(
                            phase=phase,
                            prompt=turn_prompt,
                            attempt_index=attempt_index,
                            config=config,
                            agent_instruction=agent_instruction,
                            search_enabled=search_enabled,
                            function_tools=function_tools,
                            output_schema=output_schema,
                            live_context=live_context,
                            event_summaries=turn_events,
                            deadline=deadline,
                        ),
                        timeout=remaining_timeout,
                    )
                except TimeoutError as exc:
                    return self._failed_phase_result(
                        phase=phase,
                        attempts=attempts,
                        total_usage=total_usage,
                        attempt_index=attempt_index,
                        prompt_kind=prompt_kind,
                        event_summaries=turn_events,
                        config=config,
                        started=started,
                        exc=exc,
                        terminal_status="timeout",
                    )
                except Exception as exc:
                    if _is_batch_terminal_adk_error(exc):
                        raise
                    return self._failed_phase_result(
                        phase=phase,
                        attempts=attempts,
                        total_usage=total_usage,
                        attempt_index=attempt_index,
                        prompt_kind=prompt_kind,
                        event_summaries=turn_events,
                        config=config,
                        started=started,
                        exc=exc,
                        terminal_status="invocation_error",
                    )
                validation = validate(turn.final_text)
                attempt_usage = tool_usage_from_adk_events(
                    turn.events,
                    provider=config.provider,
                    model=config.model,
                )
                attempts.append(
                    _attempt(
                        attempt_index=attempt_index,
                        prompt_kind=prompt_kind,
                        final_text=turn.final_text,
                        validation=validation,
                        event_summaries=turn.events,
                        tool_usage=attempt_usage,
                    )
                )
                total_usage = merge_tool_usage_summaries(total_usage, attempt_usage)
                if validation.ok:
                    return DomainTweakAdkPhaseResult(
                        phase=phase,
                        terminal_status=validation.terminal_status,
                        parsed_output=validation.parsed_output,
                        attempts=tuple(attempts),
                        tool_usage=total_usage,
                        elapsed_ms=_elapsed_ms(started),
                    )
                if attempt_index >= config.max_retries:
                    return DomainTweakAdkPhaseResult(
                        phase=phase,
                        terminal_status="validation_failed",
                        attempts=tuple(attempts),
                        tool_usage=total_usage,
                        elapsed_ms=_elapsed_ms(started),
                        error_type=validation.error_type,
                        error=validation.error,
                    )
                attempt_index += 1
        except TimeoutError as exc:
            return DomainTweakAdkPhaseResult(
                phase=phase,
                terminal_status="timeout",
                attempts=tuple(attempts),
                tool_usage=total_usage,
                elapsed_ms=_elapsed_ms(started),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except Exception as exc:
            if _is_batch_terminal_adk_error(exc):
                raise
            return DomainTweakAdkPhaseResult(
                phase=phase,
                terminal_status="invocation_error",
                attempts=tuple(attempts),
                tool_usage=total_usage,
                elapsed_ms=_elapsed_ms(started),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        finally:
            if live_context is not None:
                await _close_live_context(live_context, deadline=deadline)

    async def _run_turn(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        attempt_index: int,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
        search_enabled: bool,
        function_tools: Sequence[FunctionToolCallable],
        output_schema: type[BaseModel],
        live_context: _LiveAdkContext | None,
        event_summaries: list[DomainTweakAdkEventSummary],
        deadline: float,
    ) -> DomainTweakAdkTurn:
        if self._turn_executor is not None:
            return await self._turn_executor(
                phase=phase,
                prompt=prompt,
                attempt_index=attempt_index,
                config=config,
                agent_instruction=agent_instruction,
                search_enabled=search_enabled,
                function_tool_names=tuple(_function_tool_name(tool) for tool in function_tools),
                output_schema=output_schema,
            )
        if live_context is None:
            raise RuntimeError("live ADK context was not initialized")
        return await live_context.run_turn(
            prompt,
            event_summaries=event_summaries,
            deadline=deadline,
        )

    def _failed_phase_result(
        self,
        *,
        phase: DomainTweakAdkPhase,
        attempts: list[DomainTweakAdkAttempt],
        total_usage: ToolUsageSummary,
        attempt_index: int,
        prompt_kind: DomainTweakAdkPromptKind,
        event_summaries: list[DomainTweakAdkEventSummary],
        config: DomainTweakAdkRunConfig,
        started: float,
        exc: BaseException,
        terminal_status: DomainTweakAdkTerminalStatus,
    ) -> DomainTweakAdkPhaseResult:
        attempt_usage = (
            tool_usage_from_adk_events(event_summaries, provider=config.provider, model=config.model)
            if event_summaries
            else ToolUsageSummary.zero()
        )
        attempts.append(
            _attempt(
                attempt_index=attempt_index,
                prompt_kind=prompt_kind,
                final_text="",
                validation=DomainTweakValidationOutcome(
                    ok=False,
                    terminal_status=terminal_status,
                    feedback=(str(exc),) if str(exc) else (),
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
                event_summaries=tuple(event_summaries),
                tool_usage=attempt_usage,
            )
        )
        total_usage = merge_tool_usage_summaries(total_usage, attempt_usage)
        return DomainTweakAdkPhaseResult(
            phase=phase,
            terminal_status=terminal_status,
            attempts=tuple(attempts),
            tool_usage=total_usage,
            elapsed_ms=_elapsed_ms(started),
            error_type=type(exc).__name__,
            error=str(exc),
        )


class _LiveAdkContext:
    def __init__(self, *, runner: Any, user_id: str, session_id: str) -> None:
        self._runner = runner
        self._user_id = user_id
        self._session_id = session_id

    @classmethod
    async def create(
        cls,
        *,
        phase: DomainTweakAdkPhase,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
        search_enabled: bool,
        function_tools: Sequence[FunctionToolCallable],
        output_schema: type[BaseModel],
    ) -> _LiveAdkContext:
        from google.adk.agents import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        agent = Agent(
            name=_agent_name(phase),
            model=config.model,
            instruction=agent_instruction,
            tools=_adk_tools(search_enabled=search_enabled, function_tools=function_tools),
            output_schema=output_schema,
        )
        session_service = InMemorySessionService()
        session_id = f"{phase}-{time.time_ns()}"
        await session_service.create_session(
            app_name=config.app_name,
            user_id=config.user_id,
            session_id=session_id,
        )
        runner = Runner(app_name=config.app_name, agent=agent, session_service=session_service)
        return cls(runner=runner, user_id=config.user_id, session_id=session_id)

    async def run_turn(
        self,
        prompt: str,
        *,
        event_summaries: list[DomainTweakAdkEventSummary],
        deadline: float | None = None,
    ) -> DomainTweakAdkTurn:
        from google.genai import types

        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        final_text = ""
        events: list[DomainTweakAdkEventSummary] = []
        event_iterator = self._runner.run_async(
            user_id=self._user_id,
            session_id=self._session_id,
            new_message=content,
        ).__aiter__()
        try:
            async for event in event_iterator:
                summary = _redact_source_tool_content(summarize_adk_event(event))
                events.append(summary)
                event_summaries.append(summary)
                if summary.is_final_response:
                    final_text = final_text_from_event(event)
        finally:
            close_iterator = getattr(event_iterator, "aclose", None)
            if callable(close_iterator):
                try:
                    if deadline is None:
                        await close_iterator()
                    else:
                        await _run_cleanup_before_deadline(close_iterator, deadline=deadline)
                except Exception:
                    _LOGGER.warning("domain_tweak.adk_event_iterator_close_failed", exc_info=True)
        return DomainTweakAdkTurn(final_text=final_text, events=tuple(events))

    async def close(self) -> None:
        close = getattr(self._runner, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _adk_tools(
    *,
    search_enabled: bool,
    function_tools: Sequence[FunctionToolCallable],
) -> list[Any]:
    from google.adk.tools import FunctionTool
    from google.adk.tools.google_search_tool import GoogleSearchTool

    tools: list[Any] = []
    if search_enabled:
        tools.append(GoogleSearchTool(bypass_multi_tools_limit=True))
    tools.extend(FunctionTool(tool) for tool in function_tools)
    return tools


def _attempt(
    *,
    attempt_index: int,
    prompt_kind: DomainTweakAdkPromptKind,
    final_text: str,
    validation: DomainTweakValidationOutcome,
    event_summaries: Sequence[DomainTweakAdkEventSummary],
    tool_usage: ToolUsageSummary,
) -> DomainTweakAdkAttempt:
    retained = _bounded_event_summaries(event_summaries)
    return DomainTweakAdkAttempt(
        attempt_index=attempt_index,
        prompt_kind=prompt_kind,
        final_text_preview=final_text[:500],
        final_text_length=len(final_text),
        validation_ok=validation.ok,
        validation_feedback=validation.feedback,
        event_summaries=retained,
        event_summary_count=len(event_summaries),
        event_summaries_truncated=len(retained) < len(event_summaries),
        tool_usage=tool_usage,
    )


def _bounded_event_summaries(
    events: Sequence[DomainTweakAdkEventSummary],
) -> tuple[DomainTweakAdkEventSummary, ...]:
    if len(events) <= _MAX_RETAINED_EVENT_SUMMARIES:
        return tuple(events)
    half = _MAX_RETAINED_EVENT_SUMMARIES // 2
    return (*events[:half], *events[-half:])


def _redact_source_tool_content(summary: DomainTweakAdkEventSummary) -> DomainTweakAdkEventSummary:
    tool_names = {*summary.function_call_names, *summary.function_response_names}
    if not tool_names.intersection(_SOURCE_TOOL_NAMES):
        return summary
    return summary.model_copy(update={"content_text_preview": None})


def _agent_name(phase: DomainTweakAdkPhase) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", f"domain_tweak_{phase}")[:64]


def _function_tool_name(tool: FunctionToolCallable) -> str:
    return str(getattr(tool, "__name__", type(tool).__name__))


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("ADK stage timeout exceeded before live context setup")
    return remaining


def _is_batch_terminal_adk_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (DefaultCredentialsError, RefreshError)):
            return True
        if isinstance(current, ClientError) and current.code in {401, 403}:
            return True
        if isinstance(current, ToolProviderError) and is_tool_provider_credential_failure(current):
            return True
        current = current.__cause__ or current.__context__
    return False


async def _close_live_context(context: _LiveAdkContext, *, deadline: float) -> None:
    try:
        await _run_cleanup_before_deadline(context.close, deadline=deadline)
    except Exception:
        _LOGGER.warning("domain_tweak.adk_context_close_failed", exc_info=True)


async def _run_cleanup_before_deadline(cleanup: Callable[[], Any], *, deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    task = asyncio.ensure_future(cleanup())
    try:
        if remaining <= 0:
            await asyncio.sleep(0)
            done = {task} if task.done() else set()
        else:
            done, _pending = await asyncio.wait((task,), timeout=remaining)
    except BaseException:
        task.cancel()
        task.add_done_callback(_consume_cleanup_result)
        raise
    if task in done:
        await task
        return
    task.cancel()
    task.add_done_callback(_consume_cleanup_result)
    await asyncio.sleep(0)


def _consume_cleanup_result(task: asyncio.Future[Any]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        return


__all__ = [
    "DomainTweakAdkRunner",
    "DomainTweakAdkTurn",
    "DomainTweakAdkTurnExecutor",
]
