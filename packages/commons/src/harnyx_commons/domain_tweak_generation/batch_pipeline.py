"""Batch-level orchestration for domain-tweak generation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from math import ceil
from typing import TypeVar
from uuid import UUID, uuid4

from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries
from harnyx_commons.domain_tweak_generation.adk_runner import DomainTweakAdkRunner
from harnyx_commons.domain_tweak_generation.pipeline import (
    DomainTweakGenerationPipeline,
    DomainTweakPairRunResult,
)
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkPhaseResult,
    DomainTweakAdkRunConfig,
    DomainTweakBatchGenerationConfig,
    DomainTweakBatchGenerationResult,
    DomainTweakFailedFinalization,
    DomainTweakFinalizedTask,
    DomainTweakQuestionPhasePolicy,
    DomainTweakReferenceAnswerPhasePolicy,
    DomainTweakRejectedQuestionAttempt,
    DomainTweakReviewedQuestion,
)
from harnyx_commons.miner_task_generation import (
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionCandidate,
)

_RETRYABLE_A2_INVOCATION_STATUSES = {"timeout", "invocation_error"}
_MAX_DOMAIN_TWEAK_GENERATION_CONCURRENCY = 8
_T = TypeVar("_T")


class DomainTweakBatchGenerationPipeline:
    """Fills selected questions to N, then finalizes selected questions with A2."""

    def __init__(
        self,
        *,
        base_config: DomainTweakAdkRunConfig,
        runner: DomainTweakAdkRunner | None = None,
        task_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._base_config = base_config
        self._runner = runner
        self._task_id_factory = task_id_factory

    async def generate_batch(
        self,
        pair_inputs: Sequence[DomainTweakPairInput],
        config: DomainTweakBatchGenerationConfig,
    ) -> DomainTweakBatchGenerationResult:
        selected: list[DomainTweakReviewedQuestion] = []
        selected_question_texts: set[str] = set()
        rejected: list[DomainTweakRejectedQuestionAttempt] = []
        total_usage = ToolUsageSummary.zero()

        for window in _question_attempt_windows(pair_inputs, config.question_policy, config.target_count):
            if len(selected) >= config.target_count:
                break
            window_index = 0
            while window_index < len(window):
                if len(selected) >= config.target_count:
                    break
                remaining_slots = config.target_count - len(selected)
                chunk_size = min(
                    _MAX_DOMAIN_TWEAK_GENERATION_CONCURRENCY,
                    remaining_slots,
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
                    question_text = _canonical_question_text(reviewed_question.question_candidate.question)
                    if question_text in selected_question_texts:
                        rejected.append(_rejected_attempt_from_result(pair_input, pair_result))
                        continue
                    selected_question_texts.add(question_text)
                    selected.append(reviewed_question)
                    if len(selected) >= config.target_count:
                        break

        frozen_selected = tuple(selected[: config.target_count])
        if len(frozen_selected) < config.target_count:
            return DomainTweakBatchGenerationResult(
                target_count=config.target_count,
                selected_questions=frozen_selected,
                finalized_tasks=(),
                rejected_attempts=tuple(rejected),
                failed_finalizations=(),
                reference_answer_finalization_attempt_count=0,
                reference_answer_retry_attempt_count=0,
                reference_answer_retry_round_count=0,
                underfilled=True,
                tool_usage=total_usage,
            )

        finalized_by_question: dict[str, DomainTweakFinalizedTask] = {}
        failed_by_question: dict[str, DomainTweakFailedFinalization] = {}
        pending = tuple(frozen_selected)
        finalization_attempt_count = 0
        retry_attempt_count = 0
        retry_round_count = 0
        answer_invocation_cap = config.reference_answer_policy.invocation_attempt_cap(config.target_count)
        answer_invocation_budget = _InvocationBudget(answer_invocation_cap)
        for round_index in range(config.reference_answer_policy.failed_finalization_retries_per_batch_item + 1):
            if not pending:
                break
            if round_index > 0:
                retry_round_count += 1
            fresh_invocation_retries = (
                config.reference_answer_policy.invocation_retries_per_answer if round_index == 0 else 0
            )
            next_pending: list[DomainTweakReviewedQuestion] = []
            pending_index = 0
            while pending_index < len(pending):
                remaining_invocations = await answer_invocation_budget.remaining_count()
                if remaining_invocations <= 0:
                    break
                chunk_size = min(
                    _MAX_DOMAIN_TWEAK_GENERATION_CONCURRENCY,
                    remaining_invocations,
                    len(pending) - pending_index,
                )
                chunk = pending[pending_index : pending_index + chunk_size]
                pending_index += chunk_size
                chunk_finalizations = await asyncio.gather(
                    *(
                        self._finalize_with_retries(
                            reviewed_question,
                            policy=config.reference_answer_policy,
                            invocation_budget=answer_invocation_budget,
                            invocation_retries=fresh_invocation_retries,
                        )
                        for reviewed_question in chunk
                    )
                )
                finalization_attempt_count += len(chunk_finalizations)
                if round_index > 0:
                    retry_attempt_count += len(chunk_finalizations)
                for reviewed_question, finalization in zip(chunk, chunk_finalizations, strict=True):
                    total_usage = merge_tool_usage_summaries(total_usage, finalization.tool_usage)
                    question_key = _canonical_question_text(reviewed_question.question_candidate.question)
                    if isinstance(finalization, DomainTweakFinalizedTask):
                        finalized_by_question[question_key] = finalization
                        failed_by_question.pop(question_key, None)
                        continue
                    failed_by_question[question_key] = _merge_failed_finalization_attempts(
                        failed_by_question.get(question_key),
                        finalization,
                    )
                    next_pending.append(reviewed_question)
            pending = tuple(next_pending)

        finalized = tuple(
            finalized_by_question[question_key]
            for item in frozen_selected
            if (question_key := _canonical_question_text(item.question_candidate.question)) in finalized_by_question
        )
        failed_finalizations = tuple(
            failed_by_question[question_key]
            for item in frozen_selected
            if (question_key := _canonical_question_text(item.question_candidate.question)) in failed_by_question
        )

        return DomainTweakBatchGenerationResult(
            target_count=config.target_count,
            selected_questions=frozen_selected,
            finalized_tasks=finalized,
            rejected_attempts=tuple(rejected),
            failed_finalizations=failed_finalizations,
            reference_answer_finalization_attempt_count=finalization_attempt_count,
            reference_answer_retry_attempt_count=retry_attempt_count,
            reference_answer_retry_round_count=retry_round_count,
            underfilled=len(finalized) < config.target_count,
            tool_usage=total_usage,
        )

    async def _finalize_with_retries(
        self,
        reviewed_question: DomainTweakReviewedQuestion,
        *,
        policy: DomainTweakReferenceAnswerPhasePolicy,
        invocation_budget: _InvocationBudget,
        invocation_retries: int,
    ) -> DomainTweakFinalizedTask | DomainTweakFailedFinalization:
        reference_results: list[DomainTweakAdkPhaseResult] = []
        answer_usage = ToolUsageSummary.zero()
        max_invocations = invocation_retries + 1
        for _ in range(max_invocations):
            if not await invocation_budget.acquire():
                break
            finalization = await self._reference_pipeline(policy).finalize_task(
                reviewed_question,
                task_id_factory=self._task_id_factory,
            )
            answer_usage = merge_tool_usage_summaries(answer_usage, finalization.tool_usage)
            if isinstance(finalization, DomainTweakFinalizedTask):
                return DomainTweakFinalizedTask(
                    reviewed_question=finalization.reviewed_question,
                    reference_answer_result=finalization.reference_answer_result,
                    task=finalization.task,
                    tool_usage=answer_usage,
                )
            reference_results += list(finalization.reference_answer_results)
            if not _should_retry_a2_invocation(finalization.reference_answer_results):
                break
        return DomainTweakFailedFinalization(
            reviewed_question=reviewed_question,
            reference_answer_results=tuple(reference_results),
            tool_usage=answer_usage,
        )

    async def _generate_question_chunk(
        self,
        pair_inputs: Sequence[DomainTweakPairInput],
        policy: DomainTweakQuestionPhasePolicy,
    ) -> tuple[DomainTweakPairRunResult, ...]:
        return tuple(
            await asyncio.gather(
                *(
                    self._question_pipeline(policy).generate_reviewed_question(pair_input)
                    for pair_input in pair_inputs
                )
            )
        )

    def _question_pipeline(self, policy: DomainTweakQuestionPhasePolicy) -> DomainTweakGenerationPipeline:
        return DomainTweakGenerationPipeline(
            config=self._base_config.model_copy(
                update={
                    "max_retries": policy.validation_retries_per_pair,
                    "phase_timeout_seconds": policy.timeout_seconds,
                }
            ),
            runner=self._runner,
            form_review_retries=policy.form_review_retries_per_pair,
        )

    def _reference_pipeline(self, policy: DomainTweakReferenceAnswerPhasePolicy) -> DomainTweakGenerationPipeline:
        return DomainTweakGenerationPipeline(
            config=self._base_config.model_copy(
                update={
                    "max_retries": policy.validation_retries_per_answer,
                    "phase_timeout_seconds": policy.timeout_seconds,
                    "soft_timeout_seconds": policy.soft_timeout_seconds,
                    "soft_timeout_interval_seconds": policy.soft_timeout_interval_seconds,
                }
            ),
            runner=self._runner,
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
    if not isinstance(result.question_generation.parsed_output, DomainTweakQuestionCandidate):
        return None
    if not isinstance(result.form_review, DomainTweakAdkPhaseResult):
        return None
    if result.form_review.terminal_status != "validated":
        return None
    if not isinstance(result.form_review.parsed_output, DomainTweakFormReview):
        return None
    return DomainTweakReviewedQuestion(
        pair_input=pair_input,
        question_candidate=result.question_generation.parsed_output,
        form_review=result.form_review.parsed_output,
        question_generation_result=result.question_generation,
        form_review_result=result.form_review,
        tool_usage=result.tool_usage,
    )


def _rejected_attempt_from_result(
    pair_input: DomainTweakPairInput,
    result: DomainTweakPairRunResult,
) -> DomainTweakRejectedQuestionAttempt:
    return DomainTweakRejectedQuestionAttempt(
        pair_input=pair_input,
        question_generation_result=result.question_generation,
        form_review_result=result.form_review,
        tool_usage=result.tool_usage,
    )


def _should_retry_a2_invocation(results: tuple[DomainTweakAdkPhaseResult, ...]) -> bool:
    return bool(results and results[-1].terminal_status in _RETRYABLE_A2_INVOCATION_STATUSES)


def _merge_failed_finalization_attempts(
    existing: DomainTweakFailedFinalization | None,
    current: DomainTweakFailedFinalization,
) -> DomainTweakFailedFinalization:
    if existing is None:
        return current
    return DomainTweakFailedFinalization(
        reviewed_question=existing.reviewed_question,
        reference_answer_results=(*existing.reference_answer_results, *current.reference_answer_results),
        tool_usage=merge_tool_usage_summaries(existing.tool_usage, current.tool_usage),
    )


def _canonical_question_text(question: str) -> str:
    return " ".join(question.split()).casefold()


def _chunks(items: Sequence[_T], size: int) -> tuple[Sequence[_T], ...]:
    return tuple(items[index : index + size] for index in range(0, len(items), size))


class _InvocationBudget:
    def __init__(self, limit: int) -> None:
        self._remaining = limit
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    async def remaining_count(self) -> int:
        async with self._lock:
            return self._remaining


__all__ = [
    "DomainTweakBatchGenerationPipeline",
    "_question_attempt_windows",
]
