"""Terminal evidence-backed reference using the existing source-first proof path."""

from __future__ import annotations

import json
from time import monotonic

from harnyx_commons.domain.tool_usage_accounting import merge_complete_actual_cost_usage

from .agent_runner import ACTIVE_CAPTURE, CallCapture, retry_request
from .contracts import (
    CandidateSession,
    DossierAnswer,
    QuestionDraft,
    ReferenceProof,
    ReferenceQuestion,
    ReviewDecision,
    StageRunResult,
    StructuredQuestionDraft,
    TerminalReference,
)
from .output_schema import same_json_value
from .prompts import REFERENCE_SYSTEM
from .proof_validation import reference_contract_defects, validate_and_render_reference
from .reference_runner import ReferenceAgentRunner
from .source_fetch import PublicSourceFetcher
from .source_workspace import SourceWorkspace


async def derive_reference(
    draft: QuestionDraft,
    verification: ReviewDecision | None,
    session: CandidateSession,
    deadline: float,
    *,
    project_id: str | None,
    fetcher: PublicSourceFetcher,
) -> TerminalReference:
    question = ReferenceQuestion(
        question=draft.question,
        answers=tuple(
            DossierAnswer(answer_id=f"A{index + 1}", value=value) for index, value in enumerate(draft.answer_components)
        ),
        response_mode=session.mode,
        output_schema_json=draft.output_schema_json if isinstance(draft, StructuredQuestionDraft) else None,
    )
    runner = ReferenceAgentRunner(project_id=project_id)
    capture = CallCapture(session, "reference", "claude-opus-5", "vertex")

    async def call() -> tuple[StageRunResult, SourceWorkspace]:
        workspace = SourceWorkspace()
        candidates = [
            workspace.register_source_candidate(url=source.url, title="Retained author source")
            for source in session.source_support
        ]
        prompt = json.dumps(
            {
                "question": draft.question,
                "dossier_hypothesis": question.model_dump(mode="json"),
                "current_draft": draft.model_dump(mode="json"),
                "source_support": [source.model_dump(mode="json") for source in session.source_support],
                "source_candidates": [
                    {"source_candidate_id": candidate.source_candidate_id, "url": candidate.url}
                    for candidate in candidates
                ],
                "verification": verification.model_dump(by_alias=True) if verification else None,
            },
            ensure_ascii=False,
        )
        try:
            result = await runner.run_stage(
                stage="reference",
                system_prompt=REFERENCE_SYSTEM,
                prompt=prompt,
                output_model=ReferenceProof,
                timeout_seconds=max(0, min(600, deadline - monotonic())),
                web_search=True,
                tool_set=workspace.reference_tools(fetcher),
                output_validator=lambda proof: reference_contract_defects(proof, workspace=workspace, dossier=question),
            )
        except Exception as exc:
            from .contracts import BatchTerminalGenerationError, CandidateStageError

            if isinstance(exc, (CandidateStageError, BatchTerminalGenerationError)):
                capture.tool_usage = merge_complete_actual_cost_usage(capture.tool_usage, exc.tool_usage)
                if exc.tool_usage.llm.call_count:
                    capture.usage_records += 1
                capture.available_cost_usd += exc.tool_usage.actual_total_cost_usd or 0
            raise
        capture.tool_usage = merge_complete_actual_cost_usage(capture.tool_usage, result.tool_usage)
        capture.usage_records += 1
        capture.available_cost_usd += result.tool_usage.actual_total_cost_usd or 0
        return result, workspace

    token = ACTIVE_CAPTURE.set(capture)
    try:
        result, workspace = await retry_request(call, capture, deadline, solver=False)
        proof = ReferenceProof.model_validate(result.output)
        if proof.status == "giveup":
            return TerminalReference(reference_answer=None, unsupported_reason=proof.giveup_reason)
        validated = validate_and_render_reference(dossier=question, proof=proof, workspace=workspace)
        structured_correction = isinstance(draft, StructuredQuestionDraft) and not same_json_value(
            validated.reference_answer.text, draft.structured_answer_json
        )
        return TerminalReference(
            reference_answer=validated.reference_answer,
            factual_correction=structured_correction
            or any(answer.corrected_value is not None for answer in proof.answers),
        )
    finally:
        ACTIVE_CAPTURE.reset(token)
        capture.finish()
