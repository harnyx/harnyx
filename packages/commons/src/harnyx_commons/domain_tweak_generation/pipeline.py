"""Source-aware pipeline for one domain-tweak pair and one candidate finalization."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from uuid import UUID, uuid4

from pydantic import BaseModel

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries
from harnyx_commons.domain_tweak_generation.adk_runner import DomainTweakAdkRunner
from harnyx_commons.domain_tweak_generation.prompts import (
    form_blueprint_prompt,
    form_review_prompt,
    phase_instruction,
    question_generation_prompt,
    reference_answer_prompt,
    semantic_support_prompt,
)
from harnyx_commons.domain_tweak_generation.source_evidence import (
    BatchSourceEvidence,
    SourceEvidenceLimitError,
    SourceEvidenceSession,
)
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkPhase,
    DomainTweakAdkPhaseResult,
    DomainTweakAdkRunConfig,
    DomainTweakDiscardedCandidate,
    DomainTweakDiscardReason,
    DomainTweakFinalizationStage,
    DomainTweakFinalizedTask,
    DomainTweakPipelineStage,
    DomainTweakReviewedQuestion,
    DomainTweakSourceEvidenceSummary,
    DomainTweakStageSummary,
    DomainTweakValidationOutcome,
)
from harnyx_commons.domain_tweak_generation.validation import (
    reference_delivery_feedback,
    semantic_support_coverage_feedback,
    validate_form_blueprint_output,
    validate_form_review_output,
    validate_question_generation_output,
    validate_reference_answer_output,
    validate_semantic_support_output,
)
from harnyx_commons.miner_task_generation import (
    DomainTweakFormBlueprint,
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionPacket,
    DomainTweakReferenceAnswerOutput,
    DomainTweakSemanticSupportReview,
)


class DomainTweakPairRunResult(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    form_blueprint: DomainTweakAdkPhaseResult
    question_generation: DomainTweakAdkPhaseResult | None = None
    form_review: DomainTweakAdkPhaseResult | None = None
    stage_summaries: tuple[DomainTweakStageSummary, ...] = ()
    tool_usage: ToolUsageSummary


class DomainTweakGenerationPipeline:
    """Runs pair stages without owning batch selection or replacement policy."""

    def __init__(
        self,
        *,
        config: DomainTweakAdkRunConfig,
        runner: DomainTweakAdkRunner | None = None,
        source_evidence: BatchSourceEvidence | None = None,
    ) -> None:
        self._config = config
        self._runner = runner or DomainTweakAdkRunner()
        self._source_evidence = source_evidence

    async def generate_reviewed_question(self, pair_input: DomainTweakPairInput) -> DomainTweakPairRunResult:
        blueprint_result = await self._run_phase(
            phase="form_blueprint",
            prompt=form_blueprint_prompt(pair_input),
            search_enabled=False,
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        )
        stage_summaries = [_agent_stage_summary(blueprint_result)]
        total_usage = blueprint_result.tool_usage
        blueprint = blueprint_result.parsed_output
        if not isinstance(blueprint, DomainTweakFormBlueprint) or blueprint.status != "proceed":
            return DomainTweakPairRunResult(
                form_blueprint=blueprint_result,
                stage_summaries=tuple(stage_summaries),
                tool_usage=total_usage,
            )

        question_result = await self._run_phase(
            phase="question_generation",
            prompt=question_generation_prompt(pair_input, blueprint),
            search_enabled=True,
            output_schema=DomainTweakQuestionPacket,
            validate=validate_question_generation_output,
        )
        stage_summaries.append(_agent_stage_summary(question_result))
        total_usage = merge_tool_usage_summaries(total_usage, question_result.tool_usage)
        packet = question_result.parsed_output
        if not isinstance(packet, DomainTweakQuestionPacket) or packet.status != "ready":
            return DomainTweakPairRunResult(
                form_blueprint=blueprint_result,
                question_generation=question_result,
                stage_summaries=tuple(stage_summaries),
                tool_usage=total_usage,
            )

        form_result = await self._run_phase(
            phase="form_review",
            prompt=form_review_prompt(blueprint, packet),
            search_enabled=False,
            output_schema=DomainTweakFormReview,
            validate=validate_form_review_output,
        )
        stage_summaries.append(_agent_stage_summary(form_result))
        total_usage = merge_tool_usage_summaries(total_usage, form_result.tool_usage)
        return DomainTweakPairRunResult(
            form_blueprint=blueprint_result,
            question_generation=question_result,
            form_review=form_result,
            stage_summaries=tuple(stage_summaries),
            tool_usage=total_usage,
        )

    async def finalize_task(
        self,
        reviewed_question: DomainTweakReviewedQuestion,
        *,
        task_id_factory: Callable[[], UUID] = uuid4,
    ) -> DomainTweakFinalizedTask | DomainTweakDiscardedCandidate:
        if self._source_evidence is None:
            raise RuntimeError("source evidence is required for candidate finalization")
        question = reviewed_question.question_packet.question
        if question is None:
            raise ValueError("reviewed question packet must contain a question")
        session = self._source_evidence.new_session(
            objective=question,
            claim_ids=tuple(claim.claim_id for claim in reviewed_question.question_packet.claims),
        )
        stage_summaries: list[DomainTweakStageSummary] = []

        acquisition_started = time.perf_counter()
        try:
            await asyncio.wait_for(
                session.acquire_declared(reviewed_question.question_packet.evidence_declarations),
                timeout=self._config.phase_timeout_seconds,
            )
        except TimeoutError:
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="initial_evidence_acquisition",
                reason="timeout",
                stage_summaries=(
                    _deterministic_stage_summary(
                        "initial_evidence_acquisition",
                        "timeout",
                        acquisition_started,
                    ),
                ),
                source_evidence=session.summary(),
                tool_usage=session.tool_usage,
            )
        except SourceEvidenceLimitError as exc:
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="initial_evidence_acquisition",
                reason=exc.reason,
                stage_summaries=(
                    _deterministic_stage_summary(
                        "initial_evidence_acquisition",
                        exc.reason,
                        acquisition_started,
                    ),
                ),
                source_evidence=session.summary(),
                tool_usage=session.tool_usage,
            )
        source_summary = session.summary()
        acquisition_outcome = session.terminal_reason or (
            "completed_with_source_errors" if source_summary.failed_source_count else "completed"
        )
        stage_summaries.append(
            _deterministic_stage_summary(
                "initial_evidence_acquisition",
                acquisition_outcome,
                acquisition_started,
            )
        )
        if session.terminal_reason is not None:
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="initial_evidence_acquisition",
                reason=session.terminal_reason,
                stage_summaries=tuple(stage_summaries),
                source_evidence=session.summary(),
                tool_usage=session.tool_usage,
            )

        reference_result = await self._run_reference_stage(reviewed_question, session)
        stage_summaries.append(_agent_stage_summary(reference_result))
        total_usage = merge_tool_usage_summaries(session.tool_usage, reference_result.tool_usage)
        if session.terminal_reason is not None:
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="reference_answer_generation",
                reason=session.terminal_reason,
                reference_answer_result=reference_result,
                stage_summaries=tuple(stage_summaries),
                source_evidence=session.summary(),
                tool_usage=total_usage,
            )
        reference_output = reference_result.parsed_output
        if not isinstance(reference_output, DomainTweakReferenceAnswerOutput):
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="reference_answer_generation",
                reason=_phase_failure_reason(reference_result),
                reference_answer_result=reference_result,
                stage_summaries=tuple(stage_summaries),
                source_evidence=session.summary(),
                tool_usage=total_usage,
            )
        if reference_output.status == "abandon":
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="reference_answer_generation",
                reason="reference_abandon",
                reference_answer_result=reference_result,
                stage_summaries=tuple(stage_summaries),
                source_evidence=session.summary(),
                tool_usage=total_usage,
            )

        delivery_started = time.perf_counter()
        delivery_feedback = reference_delivery_feedback(
            reference_output,
            question_packet=reviewed_question.question_packet,
            allowed_window_ids=session.allowed_window_ids,
            allowed_claim_ids_by_window_id=session.allowed_claim_ids_by_window_id,
        )
        delivery_outcome = "pass" if not delivery_feedback else "delivery_validation_failed"
        stage_summaries.append(
            _deterministic_stage_summary(
                "deterministic_evidence_delivery",
                delivery_outcome,
                delivery_started,
            )
        )
        if delivery_feedback:
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="deterministic_evidence_delivery",
                reason="delivery_validation_failed",
                reference_answer_result=reference_result,
                stage_summaries=tuple(stage_summaries),
                source_evidence=session.summary(),
                tool_usage=total_usage,
            )

        evidence_window_ids = tuple(
            dict.fromkeys(
                (
                    *reference_output.citation_window_ids,
                    *(window_id for claim in reference_output.claims for window_id in claim.evidence_window_ids),
                )
            )
        )
        semantic_result = await self._run_phase(
            phase="semantic_support_gate",
            prompt=semantic_support_prompt(
                reviewed_question,
                reference_output,
                evidence_windows=session.gate_windows(evidence_window_ids),
            ),
            search_enabled=False,
            output_schema=DomainTweakSemanticSupportReview,
            validate=validate_semantic_support_output,
        )
        stage_summaries.append(_agent_stage_summary(semantic_result))
        total_usage = merge_tool_usage_summaries(total_usage, semantic_result.tool_usage)
        semantic_review = semantic_result.parsed_output
        if not isinstance(semantic_review, DomainTweakSemanticSupportReview):
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="semantic_support_gate",
                reason=_phase_failure_reason(semantic_result),
                reference_answer_result=reference_result,
                semantic_support_result=semantic_result,
                stage_summaries=tuple(stage_summaries),
                source_evidence=session.summary(),
                tool_usage=total_usage,
            )
        coverage_feedback = semantic_support_coverage_feedback(
            semantic_review,
            form_review=reviewed_question.form_review,
            reference_output=reference_output,
            allowed_window_ids=frozenset(evidence_window_ids),
        )
        if coverage_feedback:
            stage_summaries[-1] = stage_summaries[-1].model_copy(update={"outcome": "validation_failed"})
        if coverage_feedback or semantic_review.status == "abandon":
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="semantic_support_gate",
                reason=("validation_failed" if coverage_feedback else "semantic_support_abandon"),
                reference_answer_result=reference_result,
                semantic_support_result=semantic_result,
                stage_summaries=tuple(stage_summaries),
                source_evidence=session.summary(),
                tool_usage=total_usage,
            )

        hydration_started = time.perf_counter()
        try:
            if reference_output.reference_answer_text is None:
                raise ValueError("finalized reference output is missing answer text")
            citations = session.hydrate_citations(
                reference_output.citation_window_ids,
                reference_output.claims,
            )
            reference_answer = ReferenceAnswer(
                text=reference_output.reference_answer_text,
                citations=citations,
            )
        except ValueError:
            stage_summaries.append(
                _deterministic_stage_summary(
                    "citation_hydration",
                    "citation_hydration_failed",
                    hydration_started,
                )
            )
            return _discarded_candidate(
                reviewed_question,
                terminal_stage="citation_hydration",
                reason="citation_hydration_failed",
                reference_answer_result=reference_result,
                semantic_support_result=semantic_result,
                stage_summaries=tuple(stage_summaries),
                source_evidence=session.summary(),
                tool_usage=total_usage,
            )
        stage_summaries.append(_deterministic_stage_summary("citation_hydration", "completed", hydration_started))
        return DomainTweakFinalizedTask(
            reviewed_question=reviewed_question,
            reference_answer_result=reference_result,
            semantic_support_result=semantic_result,
            task=MinerTask(
                task_id=task_id_factory(),
                query=Query(text=question),
                reference_answer=reference_answer,
            ),
            stage_summaries=tuple(stage_summaries),
            source_evidence=session.summary(),
            tool_usage=total_usage,
        )

    async def _run_reference_stage(
        self,
        reviewed_question: DomainTweakReviewedQuestion,
        session: SourceEvidenceSession,
    ) -> DomainTweakAdkPhaseResult:
        try:
            return await self._run_phase(
                phase="reference_answer_generation",
                prompt=reference_answer_prompt(
                    reviewed_question,
                    source_packet=session.reference_prompt_sources(),
                    question_generation_trajectory=_question_generation_trajectory(reviewed_question),
                ),
                search_enabled=True,
                function_tools=(session.read_cached_source, session.acquire_sources),
                output_schema=DomainTweakReferenceAnswerOutput,
                validate=validate_reference_answer_output,
            )
        except SourceEvidenceLimitError as exc:
            return DomainTweakAdkPhaseResult(
                phase="reference_answer_generation",
                terminal_status="invocation_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def _run_phase(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        search_enabled: bool,
        output_schema: type[BaseModel],
        validate: Callable[[str], DomainTweakValidationOutcome],
        function_tools: tuple[Callable[..., object], ...] = (),
    ) -> DomainTweakAdkPhaseResult:
        return await self._runner.run_phase(
            phase=phase,
            prompt=prompt,
            config=self._config,
            agent_instruction=phase_instruction(phase),
            search_enabled=search_enabled,
            function_tools=function_tools,
            output_schema=output_schema,
            validate=validate,
        )


def _question_generation_trajectory(
    reviewed_question: DomainTweakReviewedQuestion,
) -> dict[str, object]:
    attempts = reviewed_question.question_generation_result.attempts
    events = tuple(event for attempt in attempts for event in attempt.event_summaries)
    search_queries = tuple(dict.fromkeys(query for event in events for query in event.web_search_queries))
    return {
        "search_queries": search_queries,
        "search_query_count": sum(event.web_search_query_count for event in events),
        "function_call_names": tuple(name for event in events for name in event.function_call_names),
        "function_response_names": tuple(name for event in events for name in event.function_response_names),
        "attempt_count": len(attempts),
        "elapsed_ms": reviewed_question.question_generation_result.elapsed_ms,
        "event_summary_count": sum(attempt.event_summary_count for attempt in attempts),
        "event_summaries_truncated": any(attempt.event_summaries_truncated for attempt in attempts),
    }


def _agent_stage_summary(result: DomainTweakAdkPhaseResult) -> DomainTweakStageSummary:
    return DomainTweakStageSummary(
        stage=result.phase,
        outcome=result.terminal_status,
        elapsed_ms=result.elapsed_ms,
    )


def _deterministic_stage_summary(
    stage: DomainTweakPipelineStage,
    outcome: str,
    started: float,
) -> DomainTweakStageSummary:
    return DomainTweakStageSummary(
        stage=stage,
        outcome=outcome,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def _phase_failure_reason(result: DomainTweakAdkPhaseResult) -> DomainTweakDiscardReason:
    if result.terminal_status == "timeout":
        return "timeout"
    if result.terminal_status == "invocation_error":
        return "invocation_error"
    return "validation_failed"


def _discarded_candidate(
    reviewed_question: DomainTweakReviewedQuestion,
    *,
    terminal_stage: DomainTweakFinalizationStage,
    reason: DomainTweakDiscardReason,
    stage_summaries: tuple[DomainTweakStageSummary, ...],
    source_evidence: DomainTweakSourceEvidenceSummary,
    tool_usage: ToolUsageSummary,
    reference_answer_result: DomainTweakAdkPhaseResult | None = None,
    semantic_support_result: DomainTweakAdkPhaseResult | None = None,
) -> DomainTweakDiscardedCandidate:
    return DomainTweakDiscardedCandidate(
        reviewed_question=reviewed_question,
        terminal_stage=terminal_stage,
        reason=reason,
        reference_answer_result=reference_answer_result,
        semantic_support_result=semantic_support_result,
        stage_summaries=stage_summaries,
        source_evidence=source_evidence,
        tool_usage=tool_usage,
    )


__all__ = [
    "DomainTweakGenerationPipeline",
    "DomainTweakPairRunResult",
]
