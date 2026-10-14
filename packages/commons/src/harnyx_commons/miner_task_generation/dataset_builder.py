"""Concurrent candidate slots, immediate acceptance and complete batch outcomes."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from time import monotonic

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.domain.tool_usage_accounting import (
    merge_complete_actual_cost_usage,
)
from harnyx_commons.task_ownership import wait_for_owned_task

from .candidate_pipeline import CandidatePipeline, GenerationRunnerPort
from .contracts import (
    AuthorReference,
    BatchGenerationResult,
    BatchTerminalGenerationError,
    CandidateResult,
    CandidateSession,
    ConstructionAuthorPort,
    FinalizedTaskCallback,
    FinalizedTaskRejectedError,
    GenerationCheckpoint,
    MinerTaskDatasetRequest,
    QuestionDraft,
    ResponseMode,
    TerminalReference,
)

logger = logging.getLogger("harnyx_commons.miner_task_generation")


class UnderfilledDatasetError(RuntimeError):
    def __init__(self, result: BatchGenerationResult) -> None:
        super().__init__(f"Generated {len(result.finalized_tasks)} of {result.target_count} tasks")
        self.result = result


class MinerTaskDatasetBuilder:
    def __init__(
        self,
        *,
        runner: GenerationRunnerPort,
        author: ConstructionAuthorPort,
        render_reference: Callable[[QuestionDraft, AuthorReference], Awaitable[TerminalReference]],
        rng: random.Random | None = None,
        timeout_seconds: float = 7200,
    ) -> None:
        if not 0 < timeout_seconds <= 7200:
            raise ValueError("Generation timeout must be positive and at most two hours")
        self._runner = runner
        self._author = author
        self._pipeline = CandidatePipeline(runner=runner, author=author, render_reference=render_reference)
        self._rng = rng or random.Random()  # noqa: S311 - response-mode sampling, not cryptography
        self._timeout = timeout_seconds

    async def aclose(self) -> None:
        try:
            await self._author.aclose()
        finally:
            await self._runner.aclose()

    async def build(self, request: MinerTaskDatasetRequest) -> tuple[MinerTask, ...]:
        result = await self.build_with_result(request)
        if len(result.finalized_tasks) != request.minimum_task_total:
            raise UnderfilledDatasetError(result)
        return tuple(item.task for item in result.finalized_tasks)

    async def build_with_result(
        self,
        request: MinerTaskDatasetRequest,
        *,
        on_finalized_task: FinalizedTaskCallback | None = None,
        outcome: BatchGenerationResult | None = None,
        on_checkpoint: Callable[[int, GenerationCheckpoint], Awaitable[None]] | None = None,
    ) -> BatchGenerationResult:
        if request.created_at is None:
            raise ValueError("Generation requires a creation timestamp")
        started = monotonic()
        result = outcome if outcome is not None else BatchGenerationResult(target_count=request.minimum_task_total)
        if result.target_count != request.minimum_task_total or result.candidates:
            raise ValueError("Invocation outcome must be empty and match the requested count")
        remaining = (request.created_at + timedelta(seconds=self._timeout) - datetime.now(UTC)).total_seconds()
        deadline = started + max(0, min(remaining, self._timeout))
        count = request.minimum_task_total
        plain_probability = 0.7 if request.plain_text_probability is None else request.plain_text_probability
        fast_probability = 0.0 if request.fast_probability is None else request.fast_probability
        modes: list[ResponseMode] = [
            "plain_text" if self._rng.random() < plain_probability else "structured" for _ in range(count)
        ]
        fast_modes = [self._rng.random() < fast_probability for _ in range(count)]
        sessions = [
            CandidateSession(task_id=f"{request.batch_id}:{slot}", effective_date=request.created_at.date())
            for slot in range(count)
        ]
        result.candidates = [
            CandidateResult(output_slot=slot, mode=modes[slot], status="operational_failure") for slot in range(count)
        ]
        failure: BaseException | None = None
        try:
            await self._run_candidates(result, sessions, fast_modes, deadline, on_finalized_task, on_checkpoint)
            return result
        except BaseException as exc:
            failure = exc
            result.error_type = type(exc).__name__
            raise
        finally:
            for candidate, session in zip(result.candidates, sessions, strict=True):
                candidate.tool_usage = session.tool_usage
                candidate.attempted_calls = session.attempted_calls
                candidate.missing_usage_calls = session.missing_usage_calls
                candidate.stage_summaries = session.stage_summaries
                if failure is not None and candidate.status == "operational_failure" and candidate.error_type is None:
                    candidate.error_type = type(failure).__name__
                result.tool_usage = merge_complete_actual_cost_usage(result.tool_usage, candidate.tool_usage)
            result.elapsed_ms = (monotonic() - started) * 1000
            if isinstance(failure, BatchTerminalGenerationError):
                failure.tool_usage = result.tool_usage
                failure.stage_summaries = tuple(
                    [
                        *result.stage_summaries,
                        *(stage for candidate in result.candidates for stage in candidate.stage_summaries),
                    ]
                )
                failure.elapsed_ms = result.elapsed_ms

    async def _run_candidates(
        self,
        result: BatchGenerationResult,
        sessions: list[CandidateSession],
        fast_modes: list[bool],
        deadline: float,
        on_finalized_task: FinalizedTaskCallback | None,
        on_checkpoint: Callable[[int, GenerationCheckpoint], Awaitable[None]] | None,
    ) -> None:
        acceptance_errors: list[Exception] = []

        async def generate(slot: int) -> None:
            session = sessions[slot]
            candidate = result.candidates[slot]

            async def checkpoint(state: GenerationCheckpoint) -> None:
                if on_checkpoint is not None:
                    await on_checkpoint(slot, state)

            result.candidates[slot] = candidate = await self._pipeline.run(
                mode=candidate.mode,
                fast=fast_modes[slot],
                session=session,
                deadline=deadline,
                result=candidate,
                checkpoint=checkpoint,
            )
            candidate.output_slot = slot
            if candidate.finalized is not None:
                cancellation = None
                try:
                    if on_finalized_task is not None:
                        finalized = candidate.finalized

                        async def publish() -> None:
                            await on_finalized_task(slot, finalized)

                        publication = asyncio.create_task(publish())
                        cancellation = await wait_for_owned_task(publication)
                        publication.result()
                except asyncio.CancelledError:
                    candidate.status = "operational_failure"
                    candidate.error_type = "CancelledError"
                    candidate.finalized = None
                    raise
                except Exception as exc:
                    acceptance_errors.append(exc)
                    candidate.status = "operational_failure"
                    candidate.error_type = type(exc).__name__
                    candidate.finalized = None
                    logger.error(
                        "task_generation.acceptance.failed",
                        extra={"data": {"task_id": session.task_id, "exception_type": type(exc).__name__}},
                    )
                    logger.debug("task_generation.acceptance.failure.details", exc_info=True)
                    if not isinstance(exc, FinalizedTaskRejectedError):
                        raise
                if cancellation is not None:
                    raise cancellation
            logger.info(
                "task_generation.candidate.completed",
                extra={"data": {"output_slot": slot, "outcome": candidate.status, "elapsed_ms": candidate.elapsed_ms}},
            )
            logger.debug("task_generation.candidate.details", extra={"data": candidate.model_dump(mode="json")})

        tasks = [asyncio.create_task(generate(slot)) for slot in range(result.target_count)]
        try:
            for completed in asyncio.as_completed(tasks):
                await completed
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                await wait_for_owned_task(task)
                if task.done() and not task.cancelled():
                    task.exception()  # Retrieve drained failures; the primary error still propagates.
        if acceptance_errors:
            raise acceptance_errors[0]
