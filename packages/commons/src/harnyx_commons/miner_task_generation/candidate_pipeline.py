"""One seed's bounded author/review/solver improvement lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
from time import monotonic
from typing import Protocol, TypeVar
from uuid import uuid4

from harnyx_commons.domain.miner_task import MinerTask, Query
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment, calculate_fast_f1
from harnyx_commons.task_ownership import wait_for_owned_task

from .contracts import (
    AgentResult,
    BatchTerminalGenerationError,
    CandidateResult,
    CandidateSession,
    CycleRecord,
    FinalizedTask,
    QuestionDraft,
    ResponseMode,
    ReviewDecision,
    Seed,
    StructuredQuestionDraft,
    TerminalReference,
)
from .output_schema import format_assessment, strict_json

logger = logging.getLogger("harnyx_commons.miner_task_generation")
MAX_CYCLES = 5
AUTHOR_ATTEMPTS = 3
SOLVER_ROLES = ("solver", "grounded_solver")
TResult = TypeVar("TResult")


async def settle_pair(tasks: dict[str, asyncio.Task[TResult]], completed: dict[str, TResult]) -> None:
    """Retain ordinary peer work; stop the pair immediately on a shared fault."""
    try:
        pending = set(tasks.values())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    raise asyncio.CancelledError
                error = task.exception()
                if isinstance(error, BatchTerminalGenerationError):
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
    async def derive_reference(
        self, draft: QuestionDraft, verification: ReviewDecision | None, session: CandidateSession, deadline: float
    ) -> TerminalReference: ...


def public_contract(draft: QuestionDraft) -> dict[str, object]:
    packet: dict[str, object] = {"question": draft.question}
    if isinstance(draft, StructuredQuestionDraft):
        packet["output_schema"] = strict_json(draft.output_schema_json)
    return packet


def seed_origin(seed: Seed) -> dict[str, object]:
    return {
        "question": seed.seed_question,
        "answer": seed.reference_answer,
        "scope": seed.scope,
        "sources": [source.model_dump(mode="json") for source in seed.supporting_sources],
    }


def encode(packet: object) -> str:
    return json.dumps(packet, ensure_ascii=False)


def solver_trace_locations(record: CycleRecord, session: CandidateSession) -> list[dict[str, str]]:
    """Stable in-memory evidence pointers replace the experiment's archive paths."""
    locations = []
    for role, answer in record.solver_results.items():
        for index, chunk in enumerate(answer.evidence_responses()):
            for candidate_index, candidate in enumerate(chunk["response"].get("candidates", [])):
                pointer = (
                    f"task:{session.task_id}:cycle:{record.cycle}#/solver_results/{role}/provider_responses/"
                    f"{index}/response/candidates/{candidate_index}"
                )
                for part_index, part in enumerate((candidate.get("content") or {}).get("parts", [])):
                    if part.get("text"):
                        locations.append(
                            {
                                "reference": f"{pointer}/content/parts/{part_index}/text",
                                "kind": "summary" if part.get("thought") else "answer_fragment",
                            }
                        )
                if candidate.get("grounding_metadata"):
                    locations.append({"reference": f"{pointer}/grounding_metadata", "kind": "grounding_metadata"})
    return locations


class CandidatePipeline:
    def __init__(self, *, runner: GenerationRunnerPort) -> None:
        self._runner = runner

    async def run(
        self,
        seed: Seed,
        *,
        mode: ResponseMode,
        fast: bool,
        session: CandidateSession,
        deadline: float,
        result: CandidateResult | None = None,
    ) -> CandidateResult:
        started = monotonic()
        result = result if result is not None else CandidateResult(seed=seed, mode=mode, status="operational_failure")
        session.mode = mode
        try:
            async with asyncio.timeout(max(0, deadline - monotonic())):
                await self._improve(result, session, fast, deadline)
        except asyncio.CancelledError:
            result.status = "operational_failure"
            result.error_type = "CancelledError"
            logger.warning("task_generation.cancelled", extra={"data": {"task_id": session.task_id}})
            raise
        except BatchTerminalGenerationError as exc:
            result.status = "operational_failure"
            result.error_type = type(exc).__name__
            raise
        except Exception as exc:
            result.status = "operational_failure"
            result.error_type = type(exc).__name__
            logger.error(
                "task_generation.failed",
                extra={
                    "data": {"task_id": session.task_id, "cycle": session.cycle, "exception_type": type(exc).__name__}
                },
            )
            logger.debug(
                "task_generation.failure.details",
                exc_info=True,
                extra={"data": {"task_id": session.task_id, "retained_candidate": result.model_dump(mode="json")}},
            )
        finally:
            result.tool_usage = session.tool_usage
            result.attempted_calls = session.attempted_calls
            result.missing_usage_calls = session.missing_usage_calls
            result.elapsed_ms = (monotonic() - started) * 1000
            result.stage_summaries = session.stage_summaries
            session.author_history.clear()
        return result

    async def _improve(self, result: CandidateResult, session: CandidateSession, fast: bool, deadline: float) -> None:
        assert result.seed is not None
        session.source_support.extend(result.seed.supporting_sources)
        context: dict[str, object] = {
            "seed_origin": seed_origin(result.seed),
            "previous_question": None,
            "previous_draft": None,
            "solver_analysis": None,
        }
        if result.mode == "structured":
            context["required_response_mode"] = "structured"
        for cycle in range(1, MAX_CYCLES + 1):
            session.cycle = cycle
            logger.info(
                "task_generation.cycle.started",
                extra={"data": {"task_id": session.task_id, "cycle": cycle, "mode": result.mode}},
            )
            draft, review, packet = await self._select(context, session, deadline)
            record = CycleRecord(cycle=cycle, draft=draft, review=review)
            result.cycles.append(record)
            contract = public_contract(draft)
            message = encode(contract) if result.mode == "structured" else draft.question
            await self._solve(record, message, session, deadline)
            schema = strict_json(draft.output_schema_json) if isinstance(draft, StructuredQuestionDraft) else None
            record.format_results = {
                role: format_assessment(answer.answer, schema) for role, answer in record.solver_results.items()
            }
            assessment_tasks = {
                role: asyncio.create_task(self._runner.assess(draft, record.solver_results[role], session, deadline))
                for role in SOLVER_ROLES
            }
            try:
                await settle_pair(assessment_tasks, record.assessments)
            finally:
                record.f1_scores = {
                    role: calculate_fast_f1(assessment) for role, assessment in record.assessments.items()
                }
            paired_miss = all(record.f1_scores[role] < 1 for role in SOLVER_ROLES)
            if paired_miss:
                verification = await self._runner.invoke("verifier", encode(packet), session, deadline)
                record.verification = ReviewDecision.model_validate_json(verification.answer)
            approved = record.verification is not None and record.verification.passed
            if not approved and cycle < MAX_CYCLES:
                analysis_packet = {
                    "seed_origin": seed_origin(result.seed),
                    "current_draft": draft.model_dump(),
                    **contract,
                    "solver_results": {
                        role: {
                            **answer.model_dump(mode="json", exclude={"search_worker_responses"}),
                            "provider_responses": answer.evidence_responses(),
                            "search_worker_responses": [
                                {
                                    **{
                                        key: value
                                        for key, value in worker.items()
                                        if key not in {"response", "grounding"}
                                    },
                                    **(
                                        {
                                            "response_reference": (
                                                f"task:{session.task_id}:cycle:{cycle}#/solver_results/{role}/"
                                                f"provider_responses/{worker['provider_response_index']}/response"
                                            ),
                                        }
                                        if "provider_response_index" in worker
                                        else {}
                                    ),
                                }
                                for worker in answer.search_worker_responses
                            ],
                        }
                        for role, answer in record.solver_results.items()
                    },
                }
                analysis_packet["evidence_file"] = f"task:{session.task_id}:cycle:{cycle}"
                analysis_packet["trace_locations"] = solver_trace_locations(record, session)
                record.analysis = (
                    await self._runner.invoke("analyst", encode(analysis_packet), session, deadline)
                ).answer
            logger.debug(
                "task_generation.cycle.completed",
                extra={"data": {"task_id": session.task_id, "record": record.model_dump(mode="json")}},
            )
            if approved:
                result.status = "finalized"
                break
            if cycle == MAX_CYCLES:
                result.status = "exhausted"
                break
            context = {
                "seed_origin": seed_origin(result.seed),
                "previous_question": draft.question,
                "previous_draft": draft.model_dump(),
                "solver_analysis": record.analysis,
            }
            if result.mode == "structured":
                context["required_response_mode"] = "structured"
            if record.verification is not None:
                context["answer_verification"] = record.verification.model_dump(by_alias=True)
        # Preserve the stopped work before a final reference can fail or exceed the deadline.
        logger.info(
            "task_generation.improvements.stopped",
            extra={"data": {"task_id": session.task_id, "cycle": session.cycle, "outcome": result.status}},
        )
        logger.debug(
            "task_generation.stopped_draft",
            extra={
                "data": {
                    "task_id": session.task_id,
                    "draft": draft.model_dump(),
                    "verification": record.verification.model_dump(by_alias=True) if record.verification else None,
                }
            },
        )
        result.reference = await self._runner.derive_reference(draft, record.verification, session, deadline)
        if (
            result.reference.reference_answer is None
            or result.reference.unsupported_reason is not None
            or (result.status == "finalized" and result.reference.factual_correction)
        ):
            result.status = "unresolved_reference"
        if result.status == "finalized":
            assert result.reference.reference_answer is not None
            result.finalized = FinalizedTask(
                task=MinerTask(
                    task_id=uuid4(),
                    query=Query(text=draft.question, output_schema=schema, fast=fast),
                    reference_answer=result.reference.reference_answer,
                ),
                tool_usage=session.tool_usage,
                stage_summaries=tuple(session.stage_summaries),
            )

    async def _select(
        self, context: dict[str, object], session: CandidateSession, deadline: float
    ) -> tuple[QuestionDraft, ReviewDecision, dict[str, object]]:
        message = encode(
            {
                "shared_context": context,
                "task": "Construct the initial question."
                if context["previous_question"] is None
                else "Use the analyst research to improve the actual discovery problem. Preserve useful research, "
                "repair demonstrated easy routes, and evolve the target or requirements when supported. "
                "Test the final question "
                "and return the complete QuestionDraft with its current answer key and revision note.",
            }
        )
        for attempt in range(1, AUTHOR_ATTEMPTS + 1):
            session.attempt = attempt
            author = await self._runner.invoke("author", message, session, deadline)
            model = StructuredQuestionDraft if session.mode == "structured" else QuestionDraft
            draft = model.model_validate_json(author.answer)
            session.source_support.extend(draft.source_support)
            author_input = dict(author.request)
            if author_input.get("message"):
                original = json.loads(author_input["message"])
                if "shared_context" in original:
                    original["shared_context"] = {"reference": "shared_context"}
                    author_input["message"] = encode(original)
            summaries = [
                chunk["response"]["text"]
                for chunk in author.evidence_responses()
                if chunk["response"].get("type") == "response.reasoning_summary_text.done"
            ]
            completed = [
                {"provider": "openai", "response": chunk["response"]["response"]}
                for chunk in author.evidence_responses()
                if chunk["response"].get("type") == "response.completed"
            ]
            packet: dict[str, object] = {
                "shared_context": context,
                "question": draft.question,
                "candidate_draft": draft.model_dump(),
                "author_trace": [
                    {
                        "attempt": str(attempt),
                        "input": author_input,
                        "provider_responses": completed,
                        "thought_summaries": summaries,
                        "summary_status": "available" if summaries else "missing",
                        "result": {"status": "completed", "answer": author.answer},
                    }
                ],
                "author_input_archive": f"task:{session.task_id}:cycle:{session.cycle}:author:{attempt}#/request",
            }
            packet["public_contract"] = public_contract(draft)
            review = ReviewDecision.model_validate_json(
                (await self._runner.invoke("reviewer", encode(packet), session, deadline)).answer
            )
            if review.passed or attempt == AUTHOR_ATTEMPTS:
                return draft, review, packet
            message = encode(
                {
                    "draft": draft.model_dump(),
                    "review": review.model_dump(by_alias=True),
                    "task": "Repair this candidate using the critique and shared context. Recheck public selection and "
                    "answer-key support after edits. Return the complete QuestionDraft, retaining useful discoveries "
                    "rather than restarting from an unrelated task.",
                }
            )
        raise AssertionError("Author selection must return a draft")

    async def _solve(self, record: CycleRecord, message: str, session: CandidateSession, deadline: float) -> None:
        tasks = {
            role: asyncio.create_task(self._runner.invoke(role, message, session, deadline)) for role in SOLVER_ROLES
        }
        await settle_pair(tasks, record.solver_results)
