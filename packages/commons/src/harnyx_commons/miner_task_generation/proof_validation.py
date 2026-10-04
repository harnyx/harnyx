"""Deterministic proof validation and host-owned citation rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from harnyx_commons.application.miner_response_hydration import (
    MAX_TOTAL_CITATION_EVIDENCE_CHARS,
    MinerResponsePayloadError,
    materialize_citation_slices,
    validate_citation_references,
)
from harnyx_commons.domain.miner_task import AnswerCitation, ReferenceAnswer, Response
from harnyx_commons.miner_task_generation.contracts import ReferenceProof, ReferenceQuestion
from harnyx_commons.miner_task_generation.source_workspace import SourceWorkspace
from harnyx_miner_sdk.json_types import JsonObject, JsonValue
from harnyx_miner_sdk.structured_output import (
    compact_json,
    validate_output_size,
)

_CITATION_MARKER = re.compile(r"\[\[\s*\d+\s*\]\]")


class ProofValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedReference:
    proof: ReferenceProof
    reference_answer: ReferenceAnswer
    output_schema: JsonObject | None


def validate_and_render_reference(
    *,
    dossier: ReferenceQuestion,
    proof: ReferenceProof,
    workspace: SourceWorkspace,
) -> ValidatedReference:
    if dossier.status != "ready" or dossier.question is None or dossier.response_mode is None:
        raise ProofValidationError("reference validation requires a ready dossier")
    if proof.status != "finalized":
        raise ProofValidationError(proof.giveup_reason or "reference proof gave up")
    expected_answer_ids = tuple(item.answer_id for item in dossier.answers)
    if not expected_answer_ids:
        raise ProofValidationError("dossier answer IDs must be non-empty")
    answer_ids = tuple(item.answer_id for item in proof.answers)
    if answer_ids != expected_answer_ids:
        raise ProofValidationError("reference answer IDs differ from the dossier contract")
    dossier_values = {item.answer_id: item.value for item in dossier.answers}
    answer_values = tuple(
        dossier_values[item.answer_id] if item.corrected_value is None else item.corrected_value
        for item in proof.answers
    )
    private_proof_values = (*answer_values, *(step.statement for step in proof.proof_steps))
    if any(_CITATION_MARKER.search(value) for value in private_proof_values):
        raise ProofValidationError("private proof fields cannot contain public citation markers")

    evidence_by_id = {item.evidence_id: item for item in workspace.evidence}
    certificates_by_id = {item.certificate_id: item for item in workspace.certificates}
    known_steps: set[str] = set()
    step_ids = [step.step_id for step in proof.proof_steps]
    if len(step_ids) != len(set(step_ids)):
        raise ProofValidationError("proof step IDs must be unique")
    for step in proof.proof_steps:
        unknown_evidence = sorted(set(step.evidence_ids) - set(evidence_by_id))
        if unknown_evidence:
            raise ProofValidationError(f"proof step references unknown evidence IDs: {unknown_evidence}")
        unknown_certificates = sorted(set(step.scan_certificate_ids) - set(certificates_by_id))
        if unknown_certificates:
            raise ProofValidationError(f"proof step references unknown scan certificates: {unknown_certificates}")
        unknown_dependencies = sorted(set(step.depends_on_step_ids) - known_steps)
        if unknown_dependencies:
            raise ProofValidationError(f"proof derivation references unavailable prior steps: {unknown_dependencies}")
        if step.kind == "supported" and not step.evidence_ids:
            raise ProofValidationError("supported proof steps require registered evidence")
        if step.kind == "derived" and not step.depends_on_step_ids:
            raise ProofValidationError("derived proof steps require prior step dependencies")
        known_steps.add(step.step_id)

    proof_evidence_ids = tuple(
        dict.fromkeys(evidence_id for step in proof.proof_steps for evidence_id in step.evidence_ids)
    )
    if not proof_evidence_ids:
        raise ProofValidationError("reference proof contains no selected evidence")
    citations: list[AnswerCitation | None] = []
    total_source_characters = 0
    try:
        for evidence_id in proof.citation_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                citations.append(None)
                continue
            source = workspace.get_source(evidence.source_id)
            materialized = materialize_citation_slices(source.content, workspace.citation_slices(evidence))
            total_source_characters += materialized.char_count
            citations.append(
                AnswerCitation(
                    url=source.final_url,
                    title=None,
                    excerpts=materialized.excerpts,
                )
            )
    except MinerResponsePayloadError as exc:
        raise ProofValidationError(str(exc)) from exc
    if total_source_characters > MAX_TOTAL_CITATION_EVIDENCE_CHARS:
        raise ProofValidationError("reference citations exceed 120000 materialized source-text characters")
    output_schema: JsonObject | None = None
    structured_value: JsonValue | None = None
    if dossier.response_mode == "structured":
        if proof.answer_text is not None:
            raise ProofValidationError("structured reference proof cannot contain answer_text")
        if proof.structured_answer_json is None:
            raise ProofValidationError("structured reference proof omitted structured_answer_json")
        from .output_schema import validate_structured_draft

        assert dossier.output_schema_json is not None
        try:
            output_schema, structured_value = validate_structured_draft(
                dossier.output_schema_json, proof.structured_answer_json
            )
            validate_output_size(structured_value)
        except (ValueError, RecursionError) as exc:
            raise ProofValidationError(str(exc)) from exc
        reference_text = compact_json(structured_value)
    else:
        if proof.answer_text is None:
            raise ProofValidationError("plain_text reference proof omitted answer_text")
        if proof.structured_answer_json is not None:
            raise ProofValidationError("plain_text reference proof cannot contain structured_answer_json")
        reference_text = proof.answer_text
    try:
        if structured_value is None:
            public_response = Response(text=reference_text, note=proof.note)
        else:
            public_response = Response(output=structured_value, note=proof.note)
    except ValidationError as exc:
        raise ProofValidationError(f"reference answer violates the public miner response contract: {exc}") from exc
    try:
        validate_citation_references((public_response.text, public_response.output, public_response.note), citations)
    except MinerResponsePayloadError as exc:
        raise ProofValidationError(str(exc)) from exc
    reference = ReferenceAnswer(
        text=reference_text,
        note=public_response.note,
        citations=tuple(citations) or None,
    )
    return ValidatedReference(proof=proof, reference_answer=reference, output_schema=output_schema)


def reference_contract_defects(
    proof: ReferenceProof,
    *,
    workspace: SourceWorkspace,
    dossier: ReferenceQuestion,
) -> tuple[str, ...]:
    if proof.status == "giveup":
        return ()
    try:
        validate_and_render_reference(
            dossier=dossier,
            proof=proof,
            workspace=workspace,
        )
    except ProofValidationError as exc:
        return (str(exc),)
    return ()


__all__ = ["ProofValidationError", "ValidatedReference", "reference_contract_defects", "validate_and_render_reference"]
