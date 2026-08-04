"""Batch orchestration for source-aware domain-tweak generation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Literal, TypeVar
from uuid import UUID, uuid4

from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries
from harnyx_commons.domain_tweak_generation.adk_runner import DomainTweakAdkRunner
from harnyx_commons.domain_tweak_generation.pipeline import (
    DomainTweakGenerationPipeline,
    DomainTweakPairRunResult,
)
from harnyx_commons.domain_tweak_generation.source_evidence import BatchSourceEvidence
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkPhaseResult,
    DomainTweakAdkRunConfig,
    DomainTweakBatchGenerationConfig,
    DomainTweakBatchGenerationResult,
    DomainTweakDiscardedCandidate,
    DomainTweakDiscardReason,
    DomainTweakFinalizedTask,
    DomainTweakQuestionPhasePolicy,
    DomainTweakReferenceAnswerPhasePolicy,
    DomainTweakRejectedQuestionAttempt,
    DomainTweakResponseMode,
    DomainTweakReviewedQuestion,
)
from harnyx_commons.miner_task_generation import (
    DomainTweakFormBlueprint,
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionPacket,
)
from harnyx_commons.tools.ports import PageExtractionProviderPort

_MAX_DOMAIN_TWEAK_GENERATION_CONCURRENCY = 8
_TResult = TypeVar("_TResult")
DomainTweakFinalizedTaskCallback = Callable[[int, DomainTweakFinalizedTask], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _FinalizationGroupResult:
    finalized_tasks: tuple[DomainTweakFinalizedTask, ...]
    finalized_output_slots: tuple[int, ...]
    discarded_candidates: tuple[DomainTweakDiscardedCandidate, ...]
    tool_usage: ToolUsageSummary


@dataclass(frozen=True, slots=True)
class _FinalizationCandidate:
    reviewed_question: DomainTweakReviewedQuestion
    response_mode: DomainTweakResponseMode
    output_slot: int


@dataclass(frozen=True, slots=True)
class _FinalizationOutcome:
    output_slot: int
    result: DomainTweakFinalizedTask | DomainTweakDiscardedCandidate


class DomainTweakBatchGenerationPipeline:
    """Fills to N finalized tasks within one global fresh-pair budget."""

    def __init__(
        self,
        *,
        base_config: DomainTweakAdkRunConfig,
        source_provider: PageExtractionProviderPort,
        runner: DomainTweakAdkRunner | None = None,
        task_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._base_config = base_config
        self._source_provider = source_provider
        self._runner = runner
        self._task_id_factory = task_id_factory

    async def generate_batch(
        self,
        pair_inputs: Sequence[DomainTweakPairInput],
        config: DomainTweakBatchGenerationConfig,
        *,
        on_finalized_task: DomainTweakFinalizedTaskCallback | None = None,
    ) -> DomainTweakBatchGenerationResult:
        selected: list[DomainTweakReviewedQuestion] = []
        selected_question_texts: set[str] = set()
        rejected: list[DomainTweakRejectedQuestionAttempt] = []
        discarded: list[DomainTweakDiscardedCandidate] = []
        total_usage = ToolUsageSummary.zero()
        finalized_by_question: dict[str, DomainTweakFinalizedTask] = {}
        finalized_output_slots: set[int] = set()
        candidate_finalization_attempt_count = 0
        candidate_budget = _InvocationBudget(config.candidate_attempt_cap)
        source_evidence = BatchSourceEvidence(
            provider=self._source_provider,
            policy=config.source_evidence_policy,
            client_model=self._base_config.model,
        )

        for window in _question_attempt_windows(pair_inputs, config.question_policy, config.target_count):
            if len(finalized_by_question) >= config.target_count:
                break
            window_index = 0
            while window_index < len(window):
                if len(finalized_by_question) >= config.target_count:
                    break
                remaining_candidates = await candidate_budget.remaining_count()
                if remaining_candidates <= 0:
                    break
                missing_response_modes = _missing_response_modes(
                    config,
                    finalized_by_question.values(),
                )
                candidate_target = min(
                    len(missing_response_modes),
                    remaining_candidates,
                )
                candidate_group: list[DomainTweakReviewedQuestion] = []

                while window_index < len(window) and len(candidate_group) < candidate_target:
                    chunk_size = min(
                        _MAX_DOMAIN_TWEAK_GENERATION_CONCURRENCY,
                        candidate_target - len(candidate_group),
                        len(window) - window_index,
                    )
                    chunk = window[window_index : window_index + chunk_size]
                    window_index += chunk_size
                    chunk_results = await self._generate_question_chunk(chunk, config.question_policy)
                    for pair_input, pair_result in zip(chunk, chunk_results, strict=True):
                        total_usage = merge_tool_usage_summaries(total_usage, pair_result.tool_usage)
                        reviewed_question = _reviewed_question_from_result(pair_input, pair_result)
                        if reviewed_question is None:
                            rejected.append(_rejected_attempt_from_result(pair_input, pair_result))
                            continue
                        question = reviewed_question.question_packet.question
                        if question is None:
                            raise ValueError("reviewed question packet must contain a question")
                        question_text = _canonical_question_text(question)
                        if question_text in selected_question_texts:
                            rejected.append(
                                _rejected_attempt_from_result(
                                    pair_input,
                                    pair_result,
                                    reason="duplicate_question",
                                )
                            )
                            continue
                        selected_question_texts.add(question_text)
                        selected.append(reviewed_question)
                        candidate_group.append(reviewed_question)

                if not candidate_group:
                    continue

                finalization_candidates = tuple(
                    _FinalizationCandidate(
                        reviewed_question=reviewed_question,
                        response_mode=response_mode,
                        output_slot=output_slot,
                    )
                    for reviewed_question, response_mode, output_slot in zip(
                        candidate_group,
                        missing_response_modes,
                        (
                            slot
                            for slot in range(config.target_count)
                            if slot not in finalized_output_slots
                        ),
                        strict=False,
                    )
                )
                group_result = await self._finalize_candidate_group_once(
                    finalization_candidates,
                    policy=config.reference_answer_policy,
                    source_evidence=source_evidence,
                    invocation_budget=candidate_budget,
                    on_finalized_task=on_finalized_task,
                )
                candidate_finalization_attempt_count += len(candidate_group)
                total_usage = merge_tool_usage_summaries(total_usage, group_result.tool_usage)
                for finalized_task in group_result.finalized_tasks:
                    question = finalized_task.reviewed_question.question_packet.question
                    if question is None:
                        raise ValueError("finalized question packet must contain a question")
                    finalized_by_question[_canonical_question_text(question)] = finalized_task
                finalized_output_slots.update(group_result.finalized_output_slots)
                discarded.extend(group_result.discarded_candidates)

        finalized = tuple(
            finalized_by_question[question_key]
            for item in selected
            if item.question_packet.question is not None
            and (question_key := _canonical_question_text(item.question_packet.question)) in finalized_by_question
        )
        return DomainTweakBatchGenerationResult(
            target_count=config.target_count,
            selected_questions=tuple(selected),
            finalized_tasks=finalized,
            rejected_attempts=tuple(rejected),
            discarded_candidates=tuple(discarded),
            candidate_finalization_attempt_count=candidate_finalization_attempt_count,
            underfilled=len(finalized) < config.target_count,
            tool_usage=total_usage,
        )

    async def _finalize_candidate_group_once(
        self,
        candidates: tuple[_FinalizationCandidate, ...],
        *,
        policy: DomainTweakReferenceAnswerPhasePolicy,
        source_evidence: BatchSourceEvidence,
        invocation_budget: _InvocationBudget,
        on_finalized_task: DomainTweakFinalizedTaskCallback | None,
    ) -> _FinalizationGroupResult:
        if not await invocation_budget.reserve(len(candidates)):
            raise RuntimeError("candidate budget cannot cover the budget-aware finalization group")
        outcomes: list[_FinalizationOutcome] = []
        for start in range(0, len(candidates), _MAX_DOMAIN_TWEAK_GENERATION_CONCURRENCY):
            chunk = candidates[start : start + _MAX_DOMAIN_TWEAK_GENERATION_CONCURRENCY]
            outcomes.extend(
                await _gather_or_cancel_siblings(
                    *(
                        self._finalize_candidate(
                            candidate,
                            policy=policy,
                            source_evidence=source_evidence,
                            on_finalized_task=on_finalized_task,
                        )
                        for candidate in chunk
                    )
                )
            )
        usage = ToolUsageSummary.zero()
        finalized: list[DomainTweakFinalizedTask] = []
        finalized_output_slots: list[int] = []
        discarded: list[DomainTweakDiscardedCandidate] = []
        for outcome in outcomes:
            usage = merge_tool_usage_summaries(usage, outcome.result.tool_usage)
            if isinstance(outcome.result, DomainTweakFinalizedTask):
                finalized.append(outcome.result)
                finalized_output_slots.append(outcome.output_slot)
            else:
                discarded.append(outcome.result)
        return _FinalizationGroupResult(
            finalized_tasks=tuple(finalized),
            finalized_output_slots=tuple(finalized_output_slots),
            discarded_candidates=tuple(discarded),
            tool_usage=usage,
        )

    async def _finalize_candidate(
        self,
        candidate: _FinalizationCandidate,
        *,
        policy: DomainTweakReferenceAnswerPhasePolicy,
        source_evidence: BatchSourceEvidence,
        on_finalized_task: DomainTweakFinalizedTaskCallback | None,
    ) -> _FinalizationOutcome:
        result = await self._reference_pipeline(policy, source_evidence).finalize_task(
            candidate.reviewed_question,
            requested_response_mode=candidate.response_mode,
            task_id_factory=self._task_id_factory,
        )
        if isinstance(result, DomainTweakFinalizedTask) and on_finalized_task is not None:
            await on_finalized_task(candidate.output_slot, result)
        return _FinalizationOutcome(output_slot=candidate.output_slot, result=result)

    async def _generate_question_chunk(
        self,
        pair_inputs: Sequence[DomainTweakPairInput],
        policy: DomainTweakQuestionPhasePolicy,
    ) -> tuple[DomainTweakPairRunResult, ...]:
        pipeline = self._question_pipeline(policy)
        return await _gather_or_cancel_siblings(
            *(pipeline.generate_reviewed_question(pair_input) for pair_input in pair_inputs)
        )

    def _question_pipeline(self, policy: DomainTweakQuestionPhasePolicy) -> DomainTweakGenerationPipeline:
        return DomainTweakGenerationPipeline(
            config=self._base_config.model_copy(
                update={
                    "max_retries": policy.validation_retries_per_stage,
                    "phase_timeout_seconds": policy.timeout_seconds,
                }
            ),
            runner=self._runner,
        )

    def _reference_pipeline(
        self,
        policy: DomainTweakReferenceAnswerPhasePolicy,
        source_evidence: BatchSourceEvidence,
    ) -> DomainTweakGenerationPipeline:
        return DomainTweakGenerationPipeline(
            config=self._base_config.model_copy(
                update={
                    "max_retries": 0,
                    "phase_timeout_seconds": policy.timeout_seconds,
                }
            ),
            runner=self._runner,
            source_evidence=source_evidence,
            structured_semantic_validation_retries=policy.structured_semantic_validation_retries,
        )


def _missing_response_modes(
    config: DomainTweakBatchGenerationConfig,
    finalized_tasks: Iterable[DomainTweakFinalizedTask],
) -> tuple[DomainTweakResponseMode, ...]:
    structured_count = 0
    plain_count = 0
    for finalized in finalized_tasks:
        if finalized.task.query.output_schema is None:
            plain_count += 1
        else:
            structured_count += 1
    plain_target_count = config.target_count - config.structured_target_count
    return ("plain",) * max(plain_target_count - plain_count, 0) + ("structured",) * max(
        config.structured_target_count - structured_count, 0
    )


def _question_attempt_windows(
    pair_inputs: Sequence[DomainTweakPairInput],
    policy: DomainTweakQuestionPhasePolicy,
    target_count: int,
) -> tuple[tuple[DomainTweakPairInput, ...], ...]:
    attempt_cap = min(len(pair_inputs), policy.hard_attempt_cap(target_count))
    if attempt_cap == 0:
        return ()
    initial_size = min(ceil(target_count * policy.target_attempt_multiplier), attempt_cap)
    windows = [tuple(pair_inputs[:initial_size])]
    consumed = initial_size
    for pass_index in range(policy.underfill_extra_passes):
        if consumed >= attempt_cap:
            break
        remaining_budget = attempt_cap - consumed
        remaining_passes = policy.underfill_extra_passes - pass_index
        window_size = ceil(remaining_budget / remaining_passes)
        windows.append(tuple(pair_inputs[consumed : consumed + window_size]))
        consumed += window_size
    return tuple(window for window in windows if window)


def _reviewed_question_from_result(
    pair_input: DomainTweakPairInput,
    result: DomainTweakPairRunResult,
) -> DomainTweakReviewedQuestion | None:
    blueprint = result.form_blueprint.parsed_output
    if not isinstance(blueprint, DomainTweakFormBlueprint) or blueprint.status != "proceed":
        return None
    if not isinstance(result.question_generation, DomainTweakAdkPhaseResult):
        return None
    packet = result.question_generation.parsed_output
    if not isinstance(packet, DomainTweakQuestionPacket) or packet.status != "ready":
        return None
    if not isinstance(result.form_review, DomainTweakAdkPhaseResult):
        return None
    if result.form_review.terminal_status != "validated":
        return None
    review = result.form_review.parsed_output
    if not isinstance(review, DomainTweakFormReview):
        return None
    return DomainTweakReviewedQuestion(
        pair_input=pair_input,
        form_blueprint=blueprint,
        question_packet=packet,
        form_review=review,
        form_blueprint_result=result.form_blueprint,
        question_generation_result=result.question_generation,
        form_review_result=result.form_review,
        stage_summaries=result.stage_summaries,
        tool_usage=result.tool_usage,
    )


def _rejected_attempt_from_result(
    pair_input: DomainTweakPairInput,
    result: DomainTweakPairRunResult,
    *,
    reason: DomainTweakDiscardReason | None = None,
) -> DomainTweakRejectedQuestionAttempt:
    terminal_stage: Literal["form_blueprint", "question_generation", "form_review"]
    terminal_result: DomainTweakAdkPhaseResult
    if result.question_generation is None:
        terminal_stage = "form_blueprint"
        terminal_result = result.form_blueprint
    elif result.form_review is None:
        terminal_stage = "question_generation"
        terminal_result = result.question_generation
    else:
        terminal_stage = "form_review"
        terminal_result = result.form_review
    return DomainTweakRejectedQuestionAttempt(
        pair_input=pair_input,
        terminal_stage=terminal_stage,
        reason=reason or _question_rejection_reason(terminal_result),
        form_blueprint_result=result.form_blueprint,
        question_generation_result=result.question_generation,
        form_review_result=result.form_review,
        stage_summaries=result.stage_summaries,
        tool_usage=result.tool_usage,
    )


def _question_rejection_reason(result: DomainTweakAdkPhaseResult) -> DomainTweakDiscardReason:
    if result.terminal_status == "no_generate":
        return "no_generate"
    if result.terminal_status == "form_rejected":
        return "form_rejected"
    if result.terminal_status == "timeout":
        return "timeout"
    if result.terminal_status == "invocation_error":
        return "invocation_error"
    return "validation_failed"


def _canonical_question_text(question: str) -> str:
    return " ".join(question.split()).casefold()


async def _gather_or_cancel_siblings(
    *awaitables: Awaitable[_TResult],
) -> tuple[_TResult, ...]:
    tasks = tuple(asyncio.ensure_future(awaitable) for awaitable in awaitables)
    try:
        return tuple(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class _InvocationBudget:
    def __init__(self, limit: int) -> None:
        self._remaining = limit
        self._lock = asyncio.Lock()

    async def reserve(self, count: int) -> bool:
        async with self._lock:
            if count > self._remaining:
                return False
            self._remaining -= count
            return True

    async def remaining_count(self) -> int:
        async with self._lock:
            return self._remaining


__all__ = [
    "DomainTweakBatchGenerationPipeline",
    "_question_attempt_windows",
]
