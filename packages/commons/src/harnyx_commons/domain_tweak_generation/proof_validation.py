"""Deterministic proof validation and host-owned citation rendering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from pydantic import ValidationError

from harnyx_commons.application.miner_response_hydration import (
    MAX_TOTAL_CITATION_EVIDENCE_CHARS,
    MinerResponsePayloadError,
    materialize_citation_slices,
)
from harnyx_commons.domain.miner_task import AnswerCitation, ReferenceAnswer, Response
from harnyx_commons.domain_tweak_generation.contracts import GroundedQuestionDossier, ReferenceProof
from harnyx_commons.domain_tweak_generation.source_workspace import SourceWorkspace
from harnyx_commons.miner_task_generation import validate_generated_output_schema
from harnyx_miner_sdk.json_types import JsonObject, JsonValue
from harnyx_miner_sdk.structured_output import (
    compact_json,
    validate_output_against_schema,
    validate_output_schema,
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
    audit_packet: dict[str, object]
    selected_source_urls: tuple[str, ...]


def validate_and_render_reference(
    *,
    dossier: GroundedQuestionDossier,
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
                    note=materialized.text,
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
        output_schema, structured_value = validate_structured_payload(
            dossier.output_schema_json,
            proof.structured_answer_json,
        )
        reference_text = compact_json(structured_value)
    else:
        if proof.answer_text is None:
            raise ProofValidationError("plain_text reference proof omitted answer_text")
        if proof.structured_answer_json is not None:
            raise ProofValidationError("plain_text reference proof cannot contain structured_answer_json")
        reference_text = proof.answer_text
    try:
        if structured_value is None:
            Response(text=reference_text)
        else:
            Response(output=structured_value)
    except ValidationError as exc:
        raise ProofValidationError(f"reference answer violates the public miner response contract: {exc}") from exc
    reference = ReferenceAnswer(text=reference_text, citations=tuple(citations) or None)
    validated_citations = [
        None if citation is None else citation.model_dump(mode="json", exclude_none=True) for citation in citations
    ]
    audit_packet = workspace.proof_packet(
        question=dossier.question,
        short_answers=answer_values,
        steps=proof.proof_steps,
        answer_text=proof.answer_text,
        validated_citations=validated_citations,
        response_mode=dossier.response_mode,
        output_schema=output_schema,
        structured_answer=structured_value,
    )
    selected_source_urls = tuple(dict.fromkeys(evidence_by_id[evidence_id].url for evidence_id in proof_evidence_ids))
    return ValidatedReference(
        proof=proof,
        reference_answer=reference,
        output_schema=output_schema,
        audit_packet=audit_packet,
        selected_source_urls=selected_source_urls,
    )


def reference_contract_defects(
    proof: ReferenceProof,
    *,
    workspace: SourceWorkspace,
    dossier: GroundedQuestionDossier,
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


def validate_structured_payload(
    output_schema_json: str | None,
    structured_answer_json: str | None,
) -> tuple[JsonObject, JsonValue]:
    if output_schema_json is None or structured_answer_json is None:
        raise ProofValidationError("structured response requires schema and answer JSON")
    try:
        schema_value = json.loads(output_schema_json)
        answer_value = json.loads(structured_answer_json)
    except (ValueError, RecursionError) as exc:
        raise ProofValidationError(f"structured response JSON could not be parsed: {exc}") from exc
    if not isinstance(schema_value, dict):
        raise ProofValidationError("structured output schema must be a JSON object")
    schema = cast(JsonObject, schema_value)
    answer = cast(JsonValue, answer_value)
    try:
        validate_generated_output_schema(schema)
        validate_output_schema(schema)
        validate_output_size(answer)
        validate_output_against_schema(answer, schema)
    except ValueError as exc:
        raise ProofValidationError(str(exc)) from exc
    return schema, answer


__all__ = [
    "ProofValidationError",
    "ValidatedReference",
    "reference_contract_defects",
    "validate_structured_payload",
    "validate_and_render_reference",
]
