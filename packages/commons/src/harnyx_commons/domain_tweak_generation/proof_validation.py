"""Deterministic proof validation and host-owned citation rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass

from harnyx_commons.domain.miner_task import AnswerCitation, ReferenceAnswer
from harnyx_commons.domain_tweak_generation.contracts import DossierAnswer, ReferenceProof
from harnyx_commons.domain_tweak_generation.source_workspace import SourceWorkspace, _ProofPacketSizeError

_CITATION_MARKER = re.compile(r"\[\[\s*\d+\s*\]\]")


class ProofValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedReference:
    proof: ReferenceProof
    reference_answer: ReferenceAnswer
    audit_packet: dict[str, object]
    selected_source_urls: tuple[str, ...]


def validate_and_render_reference(
    *,
    question: str,
    dossier_answers: tuple[DossierAnswer, ...],
    proof: ReferenceProof,
    workspace: SourceWorkspace,
) -> ValidatedReference:
    if proof.status != "finalized":
        raise ProofValidationError(proof.giveup_reason or "reference proof gave up")
    expected_answer_ids = tuple(item.answer_id for item in dossier_answers)
    if not expected_answer_ids:
        raise ProofValidationError("dossier answer IDs must be non-empty")
    answer_ids = tuple(item.answer_id for item in proof.answers)
    if answer_ids != expected_answer_ids:
        raise ProofValidationError("reference answer IDs differ from the dossier contract")
    dossier_values = {item.answer_id: item.value for item in dossier_answers}
    answer_values = tuple(
        dossier_values[item.answer_id] if item.corrected_value is None else item.corrected_value
        for item in proof.answers
    )
    authored_values = (*answer_values, *(step.statement for step in proof.proof_steps))
    if any(_CITATION_MARKER.search(value) for value in authored_values):
        raise ProofValidationError("model-authored citation markers are forbidden")

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

    ordered_evidence_ids: list[str] = []
    for step in proof.proof_steps:
        for evidence_id in step.evidence_ids:
            if evidence_id not in ordered_evidence_ids:
                ordered_evidence_ids.append(evidence_id)
    if not ordered_evidence_ids:
        raise ProofValidationError("reference proof contains no selected evidence")
    number_by_id = {evidence_id: index for index, evidence_id in enumerate(ordered_evidence_ids, start=1)}
    answer_line = "Short answers: " + "; ".join(answer_values)
    rendered_steps = []
    for step in proof.proof_steps:
        markers = "".join(f"[[{number_by_id[item]}]]" for item in step.evidence_ids)
        rendered_steps.append(f"- {step.statement}{markers}")
    citations = tuple(
        AnswerCitation(
            url=evidence_by_id[evidence_id].url,
            title=evidence_by_id[evidence_id].claim,
            note=evidence_by_id[evidence_id].excerpt,
        )
        for evidence_id in ordered_evidence_ids
    )
    reference = ReferenceAnswer(text="\n\n".join((answer_line, "\n".join(rendered_steps))), citations=citations)
    try:
        audit_packet = workspace.proof_packet(
            question=question,
            short_answers=answer_values,
            steps=proof.proof_steps,
        )
    except _ProofPacketSizeError as exc:
        raise ProofValidationError(str(exc)) from exc
    selected_source_urls = tuple(dict.fromkeys(evidence_by_id[evidence_id].url for evidence_id in ordered_evidence_ids))
    return ValidatedReference(
        proof=proof,
        reference_answer=reference,
        audit_packet=audit_packet,
        selected_source_urls=selected_source_urls,
    )


def reference_contract_defects(
    proof: ReferenceProof,
    *,
    workspace: SourceWorkspace,
    dossier_answers: tuple[DossierAnswer, ...],
) -> tuple[str, ...]:
    if proof.status == "giveup":
        return ()
    try:
        validate_and_render_reference(
            question="contract validation only",
            dossier_answers=dossier_answers,
            proof=proof,
            workspace=workspace,
        )
    except ProofValidationError as exc:
        return (str(exc),)
    return ()


__all__ = [
    "ProofValidationError",
    "ValidatedReference",
    "reference_contract_defects",
    "validate_and_render_reference",
]
