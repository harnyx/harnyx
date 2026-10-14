"""One author context, bounded Gemini challenges and exact tested-draft admission."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Protocol, TypeVar
from uuid import uuid4

from harnyx_commons.domain.miner_task import MinerTask, Query
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment, calculate_fast_f1
from harnyx_commons.task_ownership import wait_for_owned_task

from .contracts import (
    AgentResult,
    AuthorDeclaration,
    AuthorReference,
    BatchTerminalGenerationError,
    CandidateResult,
    CandidateSession,
    ChallengeFeedback,
    CheckpointCallback,
    ConstructionAuthorPort,
    CycleRecord,
    FinalizedTask,
    FinalizedTaskRejectedError,
    FinishDecision,
    GenerationCheckpoint,
    QuestionDraft,
    ReferenceValidationError,
    ResponseMode,
    StructuredQuestionDraft,
    TerminalReference,
)
from .output_schema import format_assessment, strict_json

logger = logging.getLogger("harnyx_commons.miner_task_generation")
MAX_CYCLES = 5
SOLVER_MISS_F1 = 0.5
SOLVER_ROLES = ("solver", "grounded_solver")
TResult = TypeVar("TResult")


async def settle_pair(tasks: dict[str, asyncio.Task[TResult]], completed: dict[str, TResult]) -> None:
    """Stop on any failed peer, retaining successful results and draining owned work."""
    try:
        pending = set(tasks.values())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    raise asyncio.CancelledError
                error = task.exception()
                if error is not None:
                    raise error
        for task in tasks.values():
            task.result()
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        for task in tasks.values():
            await wait_for_owned_task(task)
            if task.done() and not task.cancelled():
                task.exception()
        for role, task in tasks.items():
            if task.done() and not task.cancelled() and task.exception() is None:
                completed[role] = task.result()


class GenerationRunnerPort(Protocol):
    async def invoke(self, role: str, message: str, session: CandidateSession, deadline: float) -> AgentResult: ...
    async def assess(
        self, draft: QuestionDraft, result: AgentResult, session: CandidateSession, deadline: float
    ) -> FastJudgeAssessment: ...
    async def assess_reference(
        self, draft: QuestionDraft, reference: TerminalReference, session: CandidateSession, deadline: float
    ) -> FastJudgeAssessment: ...
    async def aclose(self) -> None: ...


def public_contract(draft: QuestionDraft) -> dict[str, object]:
    packet: dict[str, object] = {"question": draft.question}
    if isinstance(draft, StructuredQuestionDraft):
        packet["output_schema"] = strict_json(draft.output_schema_json)
    return packet


def draft_sha256(draft: QuestionDraft) -> str:
    payload = json.dumps(draft.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class CandidatePipeline:
    def __init__(
        self,
        *,
        runner: GenerationRunnerPort,
        author: ConstructionAuthorPort,
        render_reference: Callable[[QuestionDraft, AuthorReference], Awaitable[TerminalReference]],
    ) -> None:
        self._runner, self._author, self._render_reference = runner, author, render_reference

    async def run(
        self,
        *,
        mode: ResponseMode,
        fast: bool,
        session: CandidateSession,
        deadline: float,
        result: CandidateResult,
        checkpoint: CheckpointCallback | None = None,
    ) -> CandidateResult:
        started = monotonic()
        session.mode = mode
        rounds: dict[int, tuple[str, asyncio.Task[ChallengeFeedback]]] = {}
        decision: FinishDecision | None = None
        declaration: AuthorDeclaration | None = None
        last_tested_draft: QuestionDraft | None = None
        finish_lock = asyncio.Lock()

        async def record_checkpoint(state: GenerationCheckpoint) -> None:
            result.generation_metadata = state
            session.generation_checkpoint = state
            if checkpoint is not None:
                await checkpoint(state)

        def validate_mode(draft: QuestionDraft) -> None:
            if isinstance(draft, StructuredQuestionDraft) != (mode == "structured"):
                raise ValueError("Author draft does not match its assigned response mode")

        async def solve_round(draft: QuestionDraft, cycle: int, fingerprint: str) -> ChallengeFeedback:
            session.cycle = cycle
            record = CycleRecord(cycle=cycle, draft=draft.model_copy(deep=True))
            result.cycles.append(record)
            state = result.generation_metadata
            if state is not None:
                await record_checkpoint(
                    state.model_copy(update={"status": "testing", "round": cycle, "draft_sha256": fingerprint})
                )
            message = json.dumps(public_contract(draft), ensure_ascii=False) if mode == "structured" else draft.question
            await settle_pair(
                {
                    role: asyncio.create_task(self._runner.invoke(role, message, session, deadline))
                    for role in SOLVER_ROLES
                },
                record.solver_results,
            )
            schema = strict_json(draft.output_schema_json) if isinstance(draft, StructuredQuestionDraft) else None
            record.format_results = {
                role: format_assessment(answer.answer, schema) for role, answer in record.solver_results.items()
            }
            try:
                await settle_pair(
                    {
                        role: asyncio.create_task(self._runner.assess(draft, answer, session, deadline))
                        for role, answer in record.solver_results.items()
                    },
                    record.assessments,
                )
            finally:
                record.f1_scores = {role: calculate_fast_f1(grade) for role, grade in record.assessments.items()}
            paired_miss = all(record.f1_scores[role] <= SOLVER_MISS_F1 for role in SOLVER_ROLES)
            state = result.generation_metadata
            if state is not None:
                await record_checkpoint(
                    state.model_copy(
                        update={
                            "status": "tested",
                            "round": cycle,
                            "draft_sha256": fingerprint,
                            "f1_scores": record.f1_scores,
                        }
                    )
                )
            logger.debug("task_generation.cycle.completed", extra={"data": record.model_dump(mode="json")})
            return ChallengeFeedback(cycle, fingerprint, record, paired_miss, not paired_miss, MAX_CYCLES - cycle)

        async def challenge(draft: QuestionDraft, cycle: int) -> ChallengeFeedback:
            nonlocal last_tested_draft
            if declaration is not None:
                raise ValueError("Author requested a challenge after finishing")
            validate_mode(draft)
            fingerprint = draft_sha256(draft)
            if cycle in rounds:
                prior_hash, owned = rounds[cycle]
                if fingerprint != prior_hash:
                    raise ValueError("Consumed round cannot be retested with a different draft")
                return await await_round(owned)
            if rounds and rounds[len(rounds)][1].done() and rounds[len(rounds)][1].result().paired_miss:
                raise ValueError("Development stops after the first valid paired miss")
            if cycle != len(rounds) + 1 or cycle > MAX_CYCLES or any(not task.done() for _, task in rounds.values()):
                raise ValueError("Challenge rounds must be sequential and at most five")
            # Claim synchronously before the first await, including the paid work's owned task.
            last_tested_draft = draft.model_copy(deep=True)
            task = asyncio.create_task(solve_round(last_tested_draft.model_copy(deep=True), cycle, fingerprint))
            rounds[cycle] = fingerprint, task
            return await await_round(task)

        async def await_round(task: asyncio.Task[ChallengeFeedback]) -> ChallengeFeedback:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                task.cancel()
                await wait_for_owned_task(task)
                if not task.cancelled():
                    task.exception()
                raise

        async def finish(final: AuthorDeclaration) -> FinishDecision:
            nonlocal declaration, decision
            async with finish_lock:
                if declaration is not None:
                    raise ValueError("Author already supplied its terminal declaration")
                if any(not task.done() for _, task in rounds.values()):
                    raise ValueError("Author cannot finish while a challenge is running")
                if final.status != "completed":
                    status = "exhausted" if final.status == "exhausted" else "operational_failure"
                    decision = FinishDecision(False, status, final.status)
                elif last_tested_draft is None or not rounds:
                    decision = FinishDecision(False, "unresolved_reference", "untested_final_draft")
                else:
                    _, last_task = rounds[len(rounds)]
                    feedback = last_task.result()  # Operational failure never becomes an assessment.
                    if not feedback.paired_miss:
                        decision = FinishDecision(False, "exhausted", "not_paired_miss")
                    elif final.reference is None:
                        raise ReferenceValidationError("Reference completion omitted the author-built reference")
                    else:
                        rendered = await self._render_reference(last_tested_draft, final.reference)
                        if rendered.reference_answer is None:
                            decision = FinishDecision(False, "unresolved_reference", rendered.unsupported_reason)
                        else:
                            assessment = await self._runner.assess_reference(
                                last_tested_draft.model_copy(deep=True), rendered, session, deadline
                            )
                            result.reference_consistency_assessments.append(assessment)
                            logger.debug(
                                "task_generation.reference_consistency",
                                extra={"data": {"task_id": session.task_id, **assessment.model_dump(mode="json")}},
                            )
                            incorrect = [
                                item.component_id for item in assessment.expected_components if not item.is_correct
                            ]
                            excessive = [item.component_id for item in assessment.excessive_components]
                            if incorrect or excessive:
                                raise ReferenceValidationError(
                                    "The report must preserve the tested answer facts. "
                                    + json.dumps(
                                        {"missing_or_incorrect": incorrect, "contradictory_or_excessive": excessive}
                                    )
                                )
                            result.reference = rendered
                            decision = FinishDecision(True, "finalized")
                declaration = final
                result.author_declaration = final
                return decision

        try:
            if monotonic() >= deadline:
                raise TimeoutError("Reservation expired before author startup")
            async with asyncio.timeout(max(0, deadline - monotonic())):
                outcome = await self._author.run(
                    session=session,
                    mode=mode,
                    deadline=deadline,
                    challenge=challenge,
                    finish=finish,
                    checkpoint=record_checkpoint,
                )
                if not outcome.turn_completed or declaration is None or decision is None:
                    raise ValueError("Author did not finish on a successful matching turn")
                result.status, result.error_type = decision.outcome, decision.reason
                if decision.eligible:
                    assert last_tested_draft is not None
                    assert result.reference is not None
                    if result.reference.reference_answer is None:
                        result.status = "unresolved_reference"
                        result.error_type = result.reference.unsupported_reason
                    else:
                        schema = strict_json(last_tested_draft.output_schema_json) if mode == "structured" else None
                        result.finalized = FinalizedTask(
                            task=MinerTask(
                                task_id=uuid4(),
                                query=Query(text=last_tested_draft.question, output_schema=schema, fast=fast),
                                reference_answer=result.reference.reference_answer,
                            ),
                            tool_usage=session.tool_usage,
                            stage_summaries=tuple(session.stage_summaries),
                            generation_metadata=result.generation_metadata,
                        )
        except (asyncio.CancelledError, BatchTerminalGenerationError, FinalizedTaskRejectedError) as exc:
            result.status, result.error_type = "operational_failure", type(exc).__name__
            raise
        except Exception as exc:
            result.status, result.error_type = "operational_failure", type(exc).__name__
            logger.error(
                "task_generation.failed",
                extra={"data": {"task_id": session.task_id, "exception_type": type(exc).__name__}},
            )
            logger.debug("task_generation.failure.details", exc_info=True)
        finally:
            for _, task in rounds.values():
                if not task.done():
                    task.cancel()
            for _, task in rounds.values():
                await wait_for_owned_task(task)
                if not task.cancelled():
                    task.exception()
            result.tool_usage, result.stage_summaries = session.tool_usage, session.stage_summaries
            result.attempted_calls, result.missing_usage_calls = session.attempted_calls, session.missing_usage_calls
            result.elapsed_ms = (monotonic() - started) * 1000
        return result
