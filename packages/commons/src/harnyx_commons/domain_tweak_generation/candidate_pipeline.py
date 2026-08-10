"""One fresh candidate's complete dossier-to-audited-reference lifecycle."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar
from uuid import UUID, uuid4

from harnyx_commons.domain.miner_task import MinerTask, Query
from harnyx_commons.domain.tool_usage_accounting import (
    known_zero_actual_cost_tool_usage,
    merge_complete_actual_cost_usage,
)
from harnyx_commons.domain_tweak_generation.agent_runner import DomainTweakAgentRunner
from harnyx_commons.domain_tweak_generation.contracts import (
    AcceptedRouteContext,
    AuditResult,
    BatchTerminalGenerationError,
    CandidateFailure,
    CandidateOutcome,
    CandidateStageError,
    DomainTweakFinalizedTask,
    DomainTweakStageSummary,
    GenerationForm,
    PortfolioAllocation,
    QuestionPacket,
    ReferenceProof,
    SourceDossier,
    StageName,
    StageRunResult,
)
from harnyx_commons.domain_tweak_generation.prompts import (
    AUDIT_SYSTEM,
    DOSSIER_SYSTEM,
    QUESTION_SYSTEM,
    REFERENCE_SYSTEM,
    audit_prompt,
    dossier_prompt,
    question_prompt,
    reference_prompt,
    reference_repair_prompt,
)
from harnyx_commons.domain_tweak_generation.proof_validation import (
    ProofValidationError,
    ValidatedReference,
    reference_contract_defects,
    validate_and_render_reference,
)
from harnyx_commons.domain_tweak_generation.source_workspace import SourceFetcherPort, SourceWorkspace

DOSSIER_TIMEOUT_SECONDS = 360.0
QUESTION_TIMEOUT_SECONDS = 180.0
REFERENCE_TIMEOUT_SECONDS = 240.0
_OutputT = TypeVar("_OutputT", bound=SourceDossier | QuestionPacket | ReferenceProof | AuditResult)
AUDIT_TIMEOUT_SECONDS = 180.0


class CandidatePipeline:
    def __init__(
        self,
        *,
        runner: DomainTweakAgentRunner,
        source_fetcher: SourceFetcherPort,
        task_id_factory: Callable[[], UUID] = uuid4,
        workspace_factory: Callable[[], SourceWorkspace] = SourceWorkspace,
    ) -> None:
        self._runner = runner
        self._source_fetcher = source_fetcher
        self._task_id_factory = task_id_factory
        self._workspace_factory = workspace_factory

    async def run(
        self,
        form: GenerationForm,
        allocation: PortfolioAllocation,
    ) -> CandidateOutcome:
        started = time.perf_counter()
        workspace = self._workspace_factory()
        summaries: list[DomainTweakStageSummary] = []
        usage = known_zero_actual_cost_tool_usage()
        try:
            dossier_result = await self._runner.run_stage(
                stage="dossier",
                system_prompt=DOSSIER_SYSTEM,
                prompt=dossier_prompt(allocation),
                output_model=SourceDossier,
                timeout_seconds=DOSSIER_TIMEOUT_SECONDS,
                web_search=True,
                tool_set=workspace.dossier_tools(self._source_fetcher),
                output_validator=lambda output: _dossier_contract_defects(output, workspace),
            )
            usage = merge_complete_actual_cost_usage(usage, dossier_result.tool_usage)
            dossier = _typed_output(dossier_result, SourceDossier)
            summaries.append(_summary("dossier", dossier.status, dossier_result))
            if dossier.status == "no_generate":
                assert dossier.failure_class is not None
                return CandidateFailure(
                    dossier.failure_class,
                    "dossier",
                    tuple(summaries),
                    usage,
                )

            question_result = await self._runner.run_stage(
                stage="question",
                system_prompt=QUESTION_SYSTEM,
                prompt=question_prompt(form, allocation, dossier, workspace.evidence_identities()),
                output_model=QuestionPacket,
                timeout_seconds=QUESTION_TIMEOUT_SECONDS,
                output_validator=_question_contract_defects,
            )
            usage = merge_complete_actual_cost_usage(usage, question_result.tool_usage)
            question_packet = _typed_output(question_result, QuestionPacket)
            summaries.append(_summary("question", question_packet.status, question_result))
            if question_packet.status == "giveup":
                return CandidateFailure(
                    "reasoning_no_generate",
                    "question",
                    tuple(summaries),
                    usage,
                )
            assert question_packet.question is not None

            proof_result = await self._runner.run_stage(
                stage="reference",
                system_prompt=REFERENCE_SYSTEM,
                prompt=reference_prompt(
                    question=question_packet.question,
                    dossier=dossier,
                    evidence_identities=workspace.evidence_identities(),
                ),
                output_model=ReferenceProof,
                timeout_seconds=REFERENCE_TIMEOUT_SECONDS,
                web_search=True,
                tool_set=workspace.reference_tools(self._source_fetcher),
                output_validator=lambda output: reference_contract_defects(
                    output,
                    workspace=workspace,
                    dossier_answers=dossier.answers,
                ),
            )
            usage = merge_complete_actual_cost_usage(usage, proof_result.tool_usage)
            proof = _typed_output(proof_result, ReferenceProof)
            summaries.append(_summary("reference", proof.status, proof_result))
            if proof.status == "giveup":
                return CandidateFailure(
                    "reasoning_no_generate",
                    "reference",
                    tuple(summaries),
                    usage,
                )
            validated = _validate_reference(question_packet, dossier, proof, workspace)

            audit_result = await self._audit(validated)
            usage = merge_complete_actual_cost_usage(usage, audit_result.tool_usage)
            audit = _typed_output(audit_result, AuditResult)
            summaries.append(_summary("audit", audit.status, audit_result))
            repaired = False
            if audit.status == "reject":
                repaired = True
                repair_result = await self._runner.run_stage(
                    stage="reference_repair",
                    system_prompt=REFERENCE_SYSTEM,
                    prompt=reference_repair_prompt(
                        question=question_packet.question,
                        prior_proof=proof,
                        defects=audit.defects,
                    ),
                    output_model=ReferenceProof,
                    timeout_seconds=REFERENCE_TIMEOUT_SECONDS,
                    web_search=True,
                    tool_set=workspace.reference_tools(self._source_fetcher),
                    output_validator=lambda output: reference_contract_defects(
                        output,
                        workspace=workspace,
                        dossier_answers=dossier.answers,
                    ),
                )
                usage = merge_complete_actual_cost_usage(usage, repair_result.tool_usage)
                repaired_proof = _typed_output(repair_result, ReferenceProof)
                summaries.append(_summary("reference_repair", repaired_proof.status, repair_result))
                if repaired_proof.status == "giveup":
                    return CandidateFailure(
                        "audit_rejected",
                        "reference_repair",
                        tuple(summaries),
                        usage,
                    )
                validated = _validate_reference(question_packet, dossier, repaired_proof, workspace)
                second_audit_result = await self._audit(validated)
                usage = merge_complete_actual_cost_usage(usage, second_audit_result.tool_usage)
                second_audit = _typed_output(second_audit_result, AuditResult)
                summaries.append(_summary("audit", second_audit.status, second_audit_result))
                if second_audit.status == "reject":
                    return CandidateFailure(
                        "audit_rejected",
                        "audit",
                        tuple(summaries),
                        usage,
                    )

            return DomainTweakFinalizedTask(
                form_identity=form.form_identity,
                task=MinerTask(
                    task_id=self._task_id_factory(),
                    query=Query(text=question_packet.question),
                    reference_answer=validated.reference_answer,
                ),
                stage_summaries=tuple(summaries),
                tool_usage=usage,
                repaired=repaired,
                route_context=AcceptedRouteContext(
                    subject=dossier.subject or "",
                    route_summary=dossier.route_summary or "",
                    source_urls=validated.selected_source_urls,
                ),
            )
        except BatchTerminalGenerationError as exc:
            terminal_usage = merge_complete_actual_cost_usage(usage, exc.tool_usage)
            terminal_summaries = (
                *summaries,
                DomainTweakStageSummary(
                    stage=exc.stage,
                    outcome=exc.failure_class,
                    elapsed_ms=exc.elapsed_ms,
                ),
            )
            raise BatchTerminalGenerationError(
                exc.failure_class,
                str(exc),
                stage=exc.stage,
                tool_usage=terminal_usage,
                stage_summaries=terminal_summaries,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                actual_llm_cost_usd=terminal_usage.llm.actual_cost,
            ) from exc
        except CandidateStageError as exc:
            usage = merge_complete_actual_cost_usage(usage, exc.tool_usage)
            summaries.append(
                DomainTweakStageSummary(
                    stage=exc.stage,
                    outcome=exc.failure_class,
                    elapsed_ms=exc.elapsed_ms,
                )
            )
            return CandidateFailure(
                exc.failure_class,
                exc.stage,
                tuple(summaries),
                usage,
                exc.retry_after_seconds,
            )
        except ProofValidationError:
            return CandidateFailure(
                "proof_invalid",
                "reference",
                tuple(summaries),
                usage,
            )
        except Exception as exc:
            stage = summaries[-1].stage if summaries else "dossier"
            raise BatchTerminalGenerationError(
                "unexpected_pipeline_failure",
                f"{type(exc).__name__}: {exc}",
                stage=stage,
                tool_usage=usage,
                stage_summaries=tuple(summaries),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                actual_llm_cost_usd=usage.llm.actual_cost,
            ) from exc

    async def _audit(self, validated: ValidatedReference) -> StageRunResult:
        return await self._runner.run_stage(
            stage="audit",
            system_prompt=AUDIT_SYSTEM,
            prompt=audit_prompt(validated.audit_packet),
            output_model=AuditResult,
            timeout_seconds=AUDIT_TIMEOUT_SECONDS,
        )


def _typed_output(result: StageRunResult, expected: type[_OutputT]) -> _OutputT:
    if not isinstance(result.output, expected):
        raise ValueError(f"stage returned {type(result.output).__name__}, expected {expected.__name__}")
    return result.output


def _summary(stage: StageName, outcome: str, result: StageRunResult) -> DomainTweakStageSummary:
    return DomainTweakStageSummary(stage=stage, outcome=outcome, elapsed_ms=result.elapsed_ms)


def _dossier_contract_defects(
    dossier: SourceDossier,
    workspace: SourceWorkspace,
) -> tuple[str, ...]:
    if dossier.status == "no_generate":
        if dossier.source_failure_id is not None:
            source_failure = workspace.source_failure(dossier.source_failure_id)
            if source_failure is None:
                return ("dossier source_failure_id was not observed by the workspace",)
            if source_failure.failure_class != dossier.failure_class:
                return ("dossier source_failure_id does not match its declared failure_class",)
        return ()
    defects: list[str] = []
    evidence_ids = {item.evidence_id for item in workspace.evidence}
    if not evidence_ids:
        defects.append("ready dossier has no registered evidence")
    referenced = {evidence_id for fact in dossier.source_facts for evidence_id in fact.evidence_ids}
    if not referenced or not referenced <= evidence_ids:
        defects.append("dossier references missing evidence IDs")
    return tuple(defects)


def _question_contract_defects(question: QuestionPacket) -> tuple[str, ...]:
    if question.status == "giveup":
        return ()
    if question.question is None:
        return ("generated question omitted question text",)
    if "[[" in question.question or "]]" in question.question:
        return ("generated question contains a model-authored citation marker",)
    return ()


def _validate_reference(
    question: QuestionPacket,
    dossier: SourceDossier,
    proof: ReferenceProof,
    workspace: SourceWorkspace,
) -> ValidatedReference:
    assert question.question is not None
    return validate_and_render_reference(
        question=question.question,
        dossier_answers=dossier.answers,
        proof=proof,
        workspace=workspace,
    )


__all__ = [
    "AUDIT_TIMEOUT_SECONDS",
    "CandidatePipeline",
    "DOSSIER_TIMEOUT_SECONDS",
    "QUESTION_TIMEOUT_SECONDS",
    "REFERENCE_TIMEOUT_SECONDS",
]
