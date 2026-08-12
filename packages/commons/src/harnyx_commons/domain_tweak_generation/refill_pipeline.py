"""Exact-shortfall refill orchestration with bounded retained history."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import (
    known_zero_actual_cost_tool_usage,
    merge_complete_actual_cost_usage,
)
from harnyx_commons.domain_tweak_generation.agent_runner import DomainTweakAgentRunner
from harnyx_commons.domain_tweak_generation.candidate_pipeline import CandidatePipeline
from harnyx_commons.domain_tweak_generation.contracts import (
    CAPABILITY_PREFERENCES,
    AcceptedRouteContext,
    BatchTerminalGenerationError,
    CandidateFailure,
    CandidateStageError,
    CapabilityPreference,
    DomainTweakBatchGenerationResult,
    DomainTweakFinalizedTask,
    DomainTweakFinalizedTaskCallback,
    PortfolioAllocation,
    PortfolioCallCallback,
    PortfolioCallEvent,
    PortfolioPacket,
    SlotAttemptCallback,
    SlotAttemptEvent,
)
from harnyx_commons.domain_tweak_generation.prompts import PORTFOLIO_SYSTEM, portfolio_prompt
from harnyx_commons.observability.langfuse import (
    LangfuseGeneration,
    start_metadata_only_observation,
    update_observation_best_effort,
)

PORTFOLIO_TIMEOUT_SECONDS = 180.0
_MAX_PORTFOLIO_GROUP_SIZE = 10
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SlotInput:
    output_slot: int
    capability_preference: CapabilityPreference


def _capability_preferences(target_count: int) -> tuple[CapabilityPreference, ...]:
    if target_count <= 0:
        return ()
    capacity = math.ceil(target_count / len(CAPABILITY_PREFERENCES))
    return tuple(
        preference
        for preference in CAPABILITY_PREFERENCES
        for _ in range(capacity)
    )[:target_count]


@dataclass(frozen=True, slots=True)
class _PortfolioSuccess:
    call_event: PortfolioCallEvent
    allocations: dict[int, PortfolioAllocation]


@dataclass(frozen=True, slots=True)
class _PortfolioFailure:
    call_event: PortfolioCallEvent
    error: CandidateStageError


@dataclass(frozen=True, slots=True)
class _PortfolioTerminal:
    call_event: PortfolioCallEvent
    error: BatchTerminalGenerationError


@dataclass(frozen=True, slots=True)
class _CandidateRun:
    item: _SlotInput
    portfolio_call_id: str
    attempt_id: str
    elapsed_ms: float
    outcome: DomainTweakFinalizedTask | CandidateFailure | BatchTerminalGenerationError
    verdict: asyncio.Future[SlotAttemptEvent]
    observation_done: asyncio.Future[None]


class ShortfallRefillPipeline:
    def __init__(
        self,
        *,
        runner: DomainTweakAgentRunner,
        candidate_pipeline: CandidatePipeline,
        random_uniform: Callable[[float, float], float] = random.uniform,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        attempt_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._runner = runner
        self._candidate_pipeline = candidate_pipeline
        self._random_uniform = random_uniform
        self._sleep = sleep
        self._attempt_id_factory = attempt_id_factory

    async def generate_batch(
        self,
        *,
        target_count: int,
        on_finalized_task: DomainTweakFinalizedTaskCallback | None = None,
        on_portfolio_completed: PortfolioCallCallback | None = None,
        on_attempt_completed: SlotAttemptCallback | None = None,
    ) -> DomainTweakBatchGenerationResult:
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        started = time.perf_counter()
        open_slots = list(range(target_count))
        preference_by_slot = _capability_preferences(target_count)
        finalized_by_slot: dict[int, DomainTweakFinalizedTask] = {}
        canonical_questions: set[str] = set()
        accepted_route_contexts: list[AcceptedRouteContext] = []
        failures: Counter[str] = Counter()
        total_usage = known_zero_actual_cost_tool_usage()
        portfolio_call_count = 0
        slot_attempt_count = 0
        round_index = 0
        consecutive_transient_rounds = 0

        while open_slots:
            round_index += 1
            slot_inputs = tuple(_SlotInput(slot, preference_by_slot[slot]) for slot in open_slots)
            groups = tuple(
                slot_inputs[start : start + _MAX_PORTFOLIO_GROUP_SIZE]
                for start in range(0, len(slot_inputs), _MAX_PORTFOLIO_GROUP_SIZE)
            )
            round_transient = False
            retry_after_values: list[float] = []
            candidate_inputs: list[tuple[_SlotInput, PortfolioAllocation, str]] = []

            async def run_portfolio(
                group_index: int,
                group: Sequence[_SlotInput],
                current_round: int = round_index,
            ) -> tuple[Sequence[_SlotInput], _PortfolioSuccess | _PortfolioFailure | _PortfolioTerminal]:
                return group, await self._run_portfolio_group(
                    current_round,
                    group_index,
                    group,
                    accepted_route_contexts=tuple(accepted_route_contexts),
                )

            portfolio_tasks = [
                asyncio.create_task(run_portfolio(group_index, group)) for group_index, group in enumerate(groups)
            ]

            async def settle_portfolio(
                group: Sequence[_SlotInput],
                outcome: _PortfolioSuccess | _PortfolioFailure | _PortfolioTerminal,
                current_round: int = round_index,
                current_retry_after_values: list[float] = retry_after_values,
                current_candidate_inputs: list[tuple[_SlotInput, PortfolioAllocation, str]] = candidate_inputs,
            ) -> BatchTerminalGenerationError | None:
                nonlocal portfolio_call_count, round_transient, slot_attempt_count, total_usage
                portfolio_call_count += 1
                total_usage = merge_complete_actual_cost_usage(total_usage, outcome.call_event.tool_usage)
                if on_portfolio_completed is not None:
                    await on_portfolio_completed(outcome.call_event)
                if isinstance(outcome, _PortfolioTerminal):
                    return outcome.error
                if isinstance(outcome, _PortfolioFailure):
                    failure_name = outcome.error.failure_class
                    failures[failure_name] += len(group)
                    if failure_name == "transient_provider":
                        round_transient = True
                        if outcome.error.retry_after_seconds is not None:
                            current_retry_after_values.append(outcome.error.retry_after_seconds)
                    slot_attempt_count += len(group)
                    await _emit_portfolio_failure_attempts(
                        group,
                        outcome,
                        round_index=current_round,
                        attempt_id_factory=self._attempt_id_factory,
                        on_attempt_completed=on_attempt_completed,
                    )
                    return None
                current_candidate_inputs.extend(
                    (
                        item,
                        outcome.allocations[item.output_slot],
                        outcome.call_event.portfolio_call_id,
                    )
                    for item in group
                )
                return None

            try:
                pending_portfolios = set(portfolio_tasks)
                while pending_portfolios:
                    done, _ = await asyncio.wait(
                        pending_portfolios,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    ready = [task for task in portfolio_tasks if task in done]
                    pending_portfolios.difference_update(done)
                    terminal_error: BatchTerminalGenerationError | None = None
                    for portfolio_task in ready:
                        completed_group, completed_outcome = portfolio_task.result()
                        detected = await settle_portfolio(completed_group, completed_outcome)
                        terminal_error = terminal_error or detected
                    if terminal_error is None:
                        continue
                    detected = await _drain_ready_portfolio_results(
                        portfolio_tasks,
                        pending_portfolios,
                        settle_portfolio,
                    )
                    terminal_error = terminal_error or detected
                    assert terminal_error is not None
                    total_usage = _usage_after_terminal_cancellation(total_usage, bool(pending_portfolios))
                    raise _with_batch_usage(terminal_error, total_usage)
            except BaseException:
                await _cancel_and_drain(portfolio_tasks)
                raise

            async def run_candidate(
                item: _SlotInput,
                allocation: PortfolioAllocation,
                portfolio_call_id: str,
                result_future: asyncio.Future[_CandidateRun],
                verdict_future: asyncio.Future[SlotAttemptEvent],
                observation_done: asyncio.Future[None],
                current_round: int = round_index,
            ) -> None:
                attempt_id = str(self._attempt_id_factory())
                candidate_started = time.perf_counter()
                try:
                    with start_metadata_only_observation(
                        "miner_task_generation.candidate_attempt",
                        "agent",
                        metadata={
                            "attempt_id": attempt_id,
                            "round_index": current_round,
                            "output_slot": item.output_slot,
                            "portfolio_call_id": portfolio_call_id,
                        },
                    ) as parent:
                        try:
                            result = await self._candidate_pipeline.run(
                                allocation,
                                capability_preference=item.capability_preference,
                            )
                        except BatchTerminalGenerationError as exc:
                            result = exc
                        except BaseException as exc:
                            update_observation_best_effort(
                                parent,
                                metadata={"failure_type": type(exc).__name__},
                                level="ERROR",
                                status_message=type(exc).__name__,
                            )
                            raise
                        result_future.set_result(
                            _CandidateRun(
                                item=item,
                                portfolio_call_id=portfolio_call_id,
                                attempt_id=attempt_id,
                                elapsed_ms=(time.perf_counter() - candidate_started) * 1000,
                                outcome=result,
                                verdict=verdict_future,
                                observation_done=observation_done,
                            )
                        )
                        event = await verdict_future
                        _update_candidate_parent(parent, event)
                except BaseException as exc:
                    if not result_future.done():
                        if isinstance(exc, asyncio.CancelledError):
                            result_future.cancel()
                        else:
                            result_future.set_exception(exc)
                    raise
                finally:
                    if not observation_done.done():
                        observation_done.set_result(None)

            candidate_tasks: list[asyncio.Task[None]] = []
            candidate_results: list[asyncio.Future[_CandidateRun]] = []
            loop = asyncio.get_running_loop()
            for item, allocation, portfolio_call_id in candidate_inputs:
                result_future: asyncio.Future[_CandidateRun] = loop.create_future()
                verdict_future: asyncio.Future[SlotAttemptEvent] = loop.create_future()
                observation_done: asyncio.Future[None] = loop.create_future()
                candidate_results.append(result_future)
                candidate_tasks.append(
                    asyncio.create_task(
                        run_candidate(
                            item,
                            allocation,
                            portfolio_call_id,
                            result_future,
                            verdict_future,
                            observation_done,
                        )
                    )
                )

            async def settle_candidate(
                candidate_run: _CandidateRun,
                current_round: int = round_index,
                current_retry_after_values: list[float] = retry_after_values,
            ) -> BatchTerminalGenerationError | None:
                nonlocal round_transient, slot_attempt_count, total_usage
                item = candidate_run.item
                outcome = candidate_run.outcome
                slot_attempt_count += 1
                if isinstance(outcome, BatchTerminalGenerationError):
                    total_usage = merge_complete_actual_cost_usage(total_usage, outcome.tool_usage)
                    event = SlotAttemptEvent(
                        attempt_id=candidate_run.attempt_id,
                        round_index=current_round,
                        output_slot=item.output_slot,
                        outcome="batch_terminal",
                        terminal_stage=outcome.stage,
                        elapsed_ms=candidate_run.elapsed_ms,
                        stage_summaries=outcome.stage_summaries,
                        tool_usage=outcome.tool_usage,
                        portfolio_call_id=candidate_run.portfolio_call_id,
                        failure_class=outcome.failure_class,
                    )
                    candidate_run.verdict.set_result(event)
                    await candidate_run.observation_done
                    if on_attempt_completed is not None:
                        await on_attempt_completed(event)
                    return outcome
                if isinstance(outcome, DomainTweakFinalizedTask):
                    canonical = _canonical_question(outcome.task.query.text)
                    if canonical in canonical_questions:
                        outcome = CandidateFailure(
                            "contract_invalid",
                            "question_generation",
                            outcome.stage_summaries,
                            outcome.tool_usage,
                            failure_reason="generated question duplicates an already accepted question",
                        )
                    else:
                        canonical_questions.add(canonical)
                total_usage = merge_complete_actual_cost_usage(total_usage, outcome.tool_usage)
                if isinstance(outcome, DomainTweakFinalizedTask):
                    if outcome.route_context is None:
                        raise BatchTerminalGenerationError(
                            "unexpected_pipeline_failure",
                            "finalized candidate omitted request-private accepted route context",
                            stage="audit",
                            tool_usage=outcome.tool_usage,
                            stage_summaries=outcome.stage_summaries,
                            elapsed_ms=candidate_run.elapsed_ms,
                        )
                    event = SlotAttemptEvent(
                        attempt_id=candidate_run.attempt_id,
                        round_index=current_round,
                        output_slot=item.output_slot,
                        outcome="finalized",
                        terminal_stage="audit",
                        elapsed_ms=candidate_run.elapsed_ms,
                        stage_summaries=outcome.stage_summaries,
                        tool_usage=outcome.tool_usage,
                        portfolio_call_id=candidate_run.portfolio_call_id,
                        repaired=outcome.repaired,
                    )
                    candidate_run.verdict.set_result(event)
                    await candidate_run.observation_done
                    if on_attempt_completed is not None:
                        await on_attempt_completed(event)
                    accepted_route_contexts.append(outcome.route_context)
                    public_outcome = outcome.model_copy(update={"route_context": None})
                    if on_finalized_task is not None:
                        await on_finalized_task(item.output_slot, public_outcome)
                    finalized_by_slot[item.output_slot] = public_outcome
                    return None
                failures[outcome.failure_class] += 1
                if outcome.failure_class == "transient_provider":
                    round_transient = True
                    if outcome.retry_after_seconds is not None:
                        current_retry_after_values.append(outcome.retry_after_seconds)
                event = SlotAttemptEvent(
                    attempt_id=candidate_run.attempt_id,
                    round_index=current_round,
                    output_slot=item.output_slot,
                    outcome=outcome.failure_class,
                    terminal_stage=outcome.terminal_stage,
                    elapsed_ms=candidate_run.elapsed_ms,
                    stage_summaries=outcome.stage_summaries,
                    tool_usage=outcome.tool_usage,
                    retry_after_seconds=outcome.retry_after_seconds,
                    portfolio_call_id=candidate_run.portfolio_call_id,
                    failure_reason=outcome.failure_reason,
                    source_failure_id=outcome.source_failure_id,
                )
                candidate_run.verdict.set_result(event)
                await candidate_run.observation_done
                if on_attempt_completed is not None:
                    await on_attempt_completed(event)
                return None

            try:
                pending_results = set(candidate_results)
                while pending_results:
                    done, _ = await asyncio.wait(pending_results, return_when=asyncio.FIRST_COMPLETED)
                    ready = [result for result in candidate_results if result in done]
                    pending_results.difference_update(done)
                    terminal_error: BatchTerminalGenerationError | None = None
                    for result_future in ready:
                        detected = await settle_candidate(result_future.result())
                        terminal_error = terminal_error or detected
                    if terminal_error is None:
                        continue
                    detected = await _drain_ready_candidate_results(
                        candidate_results,
                        pending_results,
                        settle_candidate,
                    )
                    terminal_error = terminal_error or detected
                    assert terminal_error is not None
                    total_usage = _usage_after_terminal_cancellation(total_usage, bool(pending_results))
                    raise _with_batch_usage(terminal_error, total_usage)
                await asyncio.gather(*candidate_tasks)
            except BaseException:
                await _cancel_and_drain(candidate_tasks)
                raise

            open_slots = [slot for slot in open_slots if slot not in finalized_by_slot]
            if open_slots:
                if round_transient:
                    consecutive_transient_rounds += 1
                    delay, reason = self._retry_delay(consecutive_transient_rounds, retry_after_values)
                    _LOGGER.warning(
                        "domain_tweak_generation.refill_pacing",
                        extra={
                            "data": {
                                "round_index": round_index,
                                "shortfall": len(open_slots),
                                "delay_seconds": delay,
                                "reason": reason,
                            }
                        },
                    )
                    await self._sleep(delay)
                else:
                    consecutive_transient_rounds = 0

        return DomainTweakBatchGenerationResult(
            target_count=target_count,
            finalized_tasks=tuple(finalized_by_slot[index] for index in range(target_count)),
            portfolio_call_count=portfolio_call_count,
            slot_attempt_count=slot_attempt_count,
            round_count=round_index,
            failure_counts=dict(failures),
            tool_usage=total_usage,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    async def _run_portfolio_group(
        self,
        round_index: int,
        group_index: int,
        group: Sequence[_SlotInput],
        *,
        accepted_route_contexts: tuple[AcceptedRouteContext, ...],
    ) -> _PortfolioSuccess | _PortfolioFailure | _PortfolioTerminal:
        call_id = f"portfolio:{round_index}:{group_index}:{uuid4()}"
        started = time.perf_counter()
        with start_metadata_only_observation(
            "miner_task_generation.portfolio_call",
            "agent",
            metadata={
                "portfolio_call_id": call_id,
                "round_index": round_index,
                "group_index": group_index,
                "slot_count": len(group),
            },
        ) as parent:
            try:
                outcome = await self._execute_portfolio_group(
                    round_index=round_index,
                    group_index=group_index,
                    group=group,
                    accepted_route_contexts=accepted_route_contexts,
                    call_id=call_id,
                    started=started,
                )
            except BaseException as exc:
                update_observation_best_effort(
                    parent,
                    metadata={"failure_type": type(exc).__name__},
                    level="ERROR",
                    status_message=type(exc).__name__,
                )
                raise
            _update_portfolio_parent(parent, outcome.call_event)
            return outcome

    async def _execute_portfolio_group(
        self,
        *,
        round_index: int,
        group_index: int,
        group: Sequence[_SlotInput],
        accepted_route_contexts: tuple[AcceptedRouteContext, ...],
        call_id: str,
        started: float,
    ) -> _PortfolioSuccess | _PortfolioFailure | _PortfolioTerminal:
        portfolio_usage = known_zero_actual_cost_tool_usage()
        try:
            result = await self._runner.run_stage(
                stage="portfolio",
                system_prompt=PORTFOLIO_SYSTEM,
                prompt=portfolio_prompt(
                    tuple(item.output_slot for item in group),
                    accepted_route_contexts=accepted_route_contexts,
                ),
                output_model=PortfolioPacket,
                timeout_seconds=PORTFOLIO_TIMEOUT_SECONDS,
            )
            portfolio_usage = result.tool_usage
            if not isinstance(result.output, PortfolioPacket):
                raise BatchTerminalGenerationError(
                    "unexpected_pipeline_failure",
                    "portfolio output type differs",
                    stage="portfolio",
                    tool_usage=result.tool_usage,
                    elapsed_ms=result.elapsed_ms,
                    actual_llm_cost_usd=result.actual_llm_cost_usd,
                )
            try:
                allocations = _validate_portfolio(group, result.output)
            except CandidateStageError as exc:
                raise CandidateStageError(
                    exc.failure_class,
                    exc.stage,
                    str(exc),
                    tool_usage=result.tool_usage,
                    elapsed_ms=result.elapsed_ms,
                ) from exc
            event = PortfolioCallEvent(
                portfolio_call_id=call_id,
                round_index=round_index,
                group_index=group_index,
                slot_count=len(group),
                outcome="succeeded",
                elapsed_ms=result.elapsed_ms,
                tool_usage=result.tool_usage,
                retry_after_seconds=result.retry_after_seconds,
            )
            return _PortfolioSuccess(event, allocations)
        except BatchTerminalGenerationError as exc:
            event = PortfolioCallEvent(
                portfolio_call_id=call_id,
                round_index=round_index,
                group_index=group_index,
                slot_count=len(group),
                outcome="batch_terminal",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                tool_usage=exc.tool_usage,
                failure_class=exc.failure_class,
            )
            return _PortfolioTerminal(event, exc)
        except CandidateStageError as exc:
            event = PortfolioCallEvent(
                portfolio_call_id=call_id,
                round_index=round_index,
                group_index=group_index,
                slot_count=len(group),
                outcome=("transient_provider" if exc.failure_class == "transient_provider" else "contract_invalid"),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                tool_usage=exc.tool_usage,
                retry_after_seconds=exc.retry_after_seconds,
            )
            return _PortfolioFailure(event, exc)
        except Exception as exc:
            terminal = BatchTerminalGenerationError(
                "unexpected_pipeline_failure",
                f"{type(exc).__name__}: {exc}",
                stage="portfolio",
                tool_usage=portfolio_usage,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                actual_llm_cost_usd=portfolio_usage.llm.actual_cost,
            )
            event = PortfolioCallEvent(
                portfolio_call_id=call_id,
                round_index=round_index,
                group_index=group_index,
                slot_count=len(group),
                outcome="batch_terminal",
                elapsed_ms=terminal.elapsed_ms,
                tool_usage=terminal.tool_usage,
                failure_class=terminal.failure_class,
            )
            return _PortfolioTerminal(event, terminal)

    def _retry_delay(self, consecutive_rounds: int, retry_after_values: Sequence[float]) -> tuple[float, str]:
        if retry_after_values:
            return max(retry_after_values), "provider_retry_after"
        ceiling = min(60.0, 2.0 ** (consecutive_rounds - 1))
        return self._random_uniform(0.0, ceiling), "full_jitter_exponential"


async def _drain_ready_candidate_results(
    ordered_results: Sequence[asyncio.Future[_CandidateRun]],
    pending_results: set[asyncio.Future[_CandidateRun]],
    settle_candidate: Callable[[_CandidateRun], Awaitable[BatchTerminalGenerationError | None]],
) -> BatchTerminalGenerationError | None:
    terminal_error: BatchTerminalGenerationError | None = None
    while True:
        ready = [result for result in ordered_results if result in pending_results and result.done()]
        if not ready:
            return terminal_error
        pending_results.difference_update(ready)
        for result_future in ready:
            detected = await settle_candidate(result_future.result())
            terminal_error = terminal_error or detected


async def _emit_portfolio_failure_attempts(
    group: Sequence[_SlotInput],
    outcome: _PortfolioFailure,
    *,
    round_index: int,
    attempt_id_factory: Callable[[], UUID],
    on_attempt_completed: SlotAttemptCallback | None,
) -> None:
    for item in group:
        event = SlotAttemptEvent(
            attempt_id=str(attempt_id_factory()),
            round_index=round_index,
            output_slot=item.output_slot,
            outcome=outcome.error.failure_class,
            terminal_stage="portfolio",
            elapsed_ms=outcome.call_event.elapsed_ms,
            tool_usage=known_zero_actual_cost_tool_usage(),
            retry_after_seconds=outcome.error.retry_after_seconds,
            portfolio_call_id=outcome.call_event.portfolio_call_id,
        )
        if on_attempt_completed is not None:
            await on_attempt_completed(event)


async def _drain_ready_portfolio_results(
    ordered_results: Sequence[
        asyncio.Task[tuple[Sequence[_SlotInput], _PortfolioSuccess | _PortfolioFailure | _PortfolioTerminal]]
    ],
    pending_results: set[
        asyncio.Task[tuple[Sequence[_SlotInput], _PortfolioSuccess | _PortfolioFailure | _PortfolioTerminal]]
    ],
    settle_portfolio: Callable[
        [Sequence[_SlotInput], _PortfolioSuccess | _PortfolioFailure | _PortfolioTerminal],
        Awaitable[BatchTerminalGenerationError | None],
    ],
) -> BatchTerminalGenerationError | None:
    terminal_error: BatchTerminalGenerationError | None = None
    while True:
        ready = [result for result in ordered_results if result in pending_results and result.done()]
        if not ready:
            return terminal_error
        pending_results.difference_update(ready)
        for result_future in ready:
            completed_group, completed_outcome = result_future.result()
            detected = await settle_portfolio(completed_group, completed_outcome)
            terminal_error = terminal_error or detected


def _update_portfolio_parent(
    parent: LangfuseGeneration | None,
    event: PortfolioCallEvent,
) -> None:
    update_observation_best_effort(
        parent,
        metadata={
            "outcome": event.outcome,
            "failure_class": event.failure_class,
            "elapsed_ms": event.elapsed_ms,
        },
        level="DEFAULT" if event.outcome == "succeeded" else "ERROR",
        status_message=event.outcome,
    )


def _update_candidate_parent(
    parent: LangfuseGeneration | None,
    event: SlotAttemptEvent,
) -> None:
    failure_class = event.failure_class
    if event.outcome not in {"batch_terminal", "finalized"}:
        failure_class = event.outcome
    update_observation_best_effort(
        parent,
        metadata={
            "outcome": event.outcome,
            "terminal_stage": event.terminal_stage,
            "failure_class": failure_class,
        },
        level="DEFAULT" if event.outcome == "finalized" else "ERROR",
        status_message=event.outcome,
    )


def _validate_portfolio(
    group: Sequence[_SlotInput],
    packet: PortfolioPacket,
) -> dict[int, PortfolioAllocation]:
    expected_slots = {item.output_slot for item in group}
    allocations = {item.slot: item for item in packet.allocations}
    if set(allocations) != expected_slots:
        raise CandidateStageError("contract_invalid", "portfolio", "portfolio slot set differs from request")
    return allocations


async def _cancel_and_drain(tasks: Sequence[asyncio.Task[Any]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _usage_after_terminal_cancellation(usage: ToolUsageSummary, has_unresolved_siblings: bool) -> ToolUsageSummary:
    if not has_unresolved_siblings:
        return usage
    return merge_complete_actual_cost_usage(usage, ToolUsageSummary.zero())


def _canonical_question(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _with_batch_usage(
    error: BatchTerminalGenerationError,
    usage: ToolUsageSummary,
) -> BatchTerminalGenerationError:
    return BatchTerminalGenerationError(
        error.failure_class,
        str(error),
        stage=error.stage,
        tool_usage=usage,
        stage_summaries=error.stage_summaries,
        elapsed_ms=error.elapsed_ms,
        actual_llm_cost_usd=usage.llm.actual_cost,
    )


__all__ = ["PORTFOLIO_TIMEOUT_SECONDS", "ShortfallRefillPipeline"]
