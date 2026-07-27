"""Strict structural validation for source-aware domain-tweak stage outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TypeVar, cast

from pydantic import BaseModel, ValidationError

from harnyx_commons.domain.miner_task import Query
from harnyx_commons.domain_tweak_generation.source_evidence import (
    DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID,
)
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakValidationOutcome,
)
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.miner_task_generation import (
    DomainTweakFormBlueprint,
    DomainTweakFormReview,
    DomainTweakQuestionPacket,
    DomainTweakReferenceAnswerOutput,
    DomainTweakRequiredRelation,
    DomainTweakSemanticSupportReview,
    DomainTweakStructuredOutputMaterialization,
    DomainTweakStructuredOutputMaterializationWire,
    validate_generated_output_schema,
)
from harnyx_miner_sdk.structured_output import (
    JSON_SCHEMA_DRAFT_2020_12,
    validate_output_against_schema,
    validate_output_size,
)

TModel = TypeVar("TModel", bound=BaseModel)
_SOURCE_BACKED_REQUIREMENT_RELATIONS: frozenset[DomainTweakRequiredRelation] = frozenset(
    (
        "direct_fact",
        "field_equivalence",
        "exhaustive_set",
        "exact_cardinality",
        "absence",
        "upper_or_lower_bound",
    )
)


def validate_form_blueprint_output(text: str) -> DomainTweakValidationOutcome:
    parsed, error = _parse_model(text, DomainTweakFormBlueprint)
    if error is not None:
        return error
    assert parsed is not None
    return DomainTweakValidationOutcome(
        ok=True,
        terminal_status="no_generate" if parsed.status == "no_generate" else "validated",
        parsed_output=parsed,
    )


def validate_question_generation_output(text: str) -> DomainTweakValidationOutcome:
    parsed, error = _parse_model(text, DomainTweakQuestionPacket)
    if error is not None:
        return error
    assert parsed is not None
    return DomainTweakValidationOutcome(
        ok=True,
        terminal_status="no_generate" if parsed.status == "no_generate" else "validated",
        parsed_output=parsed,
    )


def validate_form_review_output(text: str) -> DomainTweakValidationOutcome:
    parsed, error = _parse_model(text, DomainTweakFormReview)
    if error is not None:
        return error
    assert parsed is not None
    return DomainTweakValidationOutcome(
        ok=True,
        terminal_status="validated" if parsed.form_match else "form_rejected",
        parsed_output=parsed,
        feedback=() if parsed.form_match else (parsed.reviewer_feedback,),
    )


def validate_reference_answer_output(text: str) -> DomainTweakValidationOutcome:
    parsed, error = _parse_model(text, DomainTweakReferenceAnswerOutput)
    if error is not None:
        return error
    assert parsed is not None
    return DomainTweakValidationOutcome(
        ok=True,
        terminal_status="validated" if parsed.status == "finalized" else "abandoned",
        parsed_output=parsed,
        feedback=() if parsed.status == "finalized" else (parsed.abandon_reason or "abandon",),
    )


def validate_structured_output_materialization(
    text: str,
    *,
    question: str,
    form_review: DomainTweakFormReview,
    reference_output: DomainTweakReferenceAnswerOutput,
) -> DomainTweakValidationOutcome:
    wire, error = _parse_model(text, DomainTweakStructuredOutputMaterializationWire)
    if error is not None:
        return error
    assert wire is not None
    if wire.disposition == "not_materializable":
        return DomainTweakValidationOutcome(
            ok=True,
            terminal_status="abandoned",
            parsed_output=DomainTweakStructuredOutputMaterialization(
                disposition=wire.disposition,
                rationale=wire.rationale,
                output_schema=None,
                structured_output=None,
                field_bindings=(),
            ),
            feedback=(wire.rationale,),
        )

    try:
        raw_schema = json.loads(
            wire.output_schema_json or "",
            parse_constant=_reject_nonfinite_json_constant,
        )
        structured_output = json.loads(
            wire.structured_output_json or "",
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return _validation_failure("schema and output fields must contain strict JSON", exc)
    if not isinstance(raw_schema, dict):
        return _validation_failure("output_schema_json must encode a JSON object")
    if not isinstance(structured_output, dict):
        return _validation_failure("structured_output_json must encode a JSON object")

    schema = cast(JsonObject, raw_schema)
    dialect = schema.get("$schema")
    if dialect is not None and dialect != JSON_SCHEMA_DRAFT_2020_12:
        return _validation_failure("generated schema uses an unsupported $schema dialect")
    schema["$schema"] = JSON_SCHEMA_DRAFT_2020_12
    try:
        leaf_paths = validate_generated_output_schema(schema)
        Query(text=question, output_schema=schema)
        validate_output_size(cast(JsonValue, structured_output))
        validate_output_against_schema(cast(JsonValue, structured_output), schema)
    except (ValueError, ValidationError) as exc:
        return _validation_failure(str(exc), exc)

    feedback = _materialization_binding_feedback(
        wire,
        leaf_paths=leaf_paths,
        form_review=form_review,
        reference_output=reference_output,
    )
    if feedback:
        return DomainTweakValidationOutcome(
            ok=False,
            terminal_status="validation_failed",
            feedback=feedback,
            error_type="MaterializationBindingError",
            error="; ".join(feedback),
        )
    return DomainTweakValidationOutcome(
        ok=True,
        terminal_status="validated",
        parsed_output=DomainTweakStructuredOutputMaterialization(
            disposition=wire.disposition,
            rationale=wire.rationale,
            output_schema=schema,
            structured_output=cast(JsonValue, structured_output),
            field_bindings=wire.field_bindings,
        ),
    )


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def validate_semantic_support_output(text: str) -> DomainTweakValidationOutcome:
    parsed, error = _parse_model(text, DomainTweakSemanticSupportReview)
    if error is not None:
        return error
    assert parsed is not None
    return DomainTweakValidationOutcome(
        ok=True,
        terminal_status="validated" if parsed.status == "pass" else "semantic_rejected",
        parsed_output=parsed,
        feedback=() if parsed.status == "pass" else (parsed.abandon_reason or "abandon",),
    )


def validate_structured_semantic_support_output(
    text: str,
    *,
    form_review: DomainTweakFormReview,
    reference_output: DomainTweakReferenceAnswerOutput,
    allowed_window_ids: frozenset[str],
    materialization: DomainTweakStructuredOutputMaterialization,
) -> DomainTweakValidationOutcome:
    outcome = validate_semantic_support_output(text)
    review = outcome.parsed_output
    if not outcome.ok or not isinstance(review, DomainTweakSemanticSupportReview):
        return outcome
    feedback = semantic_support_coverage_feedback(
        review,
        form_review=form_review,
        reference_output=reference_output,
        allowed_window_ids=allowed_window_ids,
        materialization=materialization,
    )
    if not feedback:
        return outcome
    return DomainTweakValidationOutcome(
        ok=False,
        terminal_status="validation_failed",
        feedback=feedback,
        error_type="SemanticSupportCoverageError",
        error="; ".join(feedback),
    )


def _materialization_binding_feedback(
    wire: DomainTweakStructuredOutputMaterializationWire,
    *,
    leaf_paths: tuple[str, ...],
    form_review: DomainTweakFormReview,
    reference_output: DomainTweakReferenceAnswerOutput,
) -> tuple[str, ...]:
    feedback: list[str] = []
    binding_paths = [binding.schema_path for binding in wire.field_bindings]
    if len(binding_paths) != len(set(binding_paths)):
        feedback.append("structured field binding paths must be unique")
    if set(binding_paths) != set(leaf_paths):
        feedback.append("structured field bindings must exactly match schema leaf paths")

    known_requirement_ids = {requirement.requirement_id for requirement in form_review.question_requirements}
    reference_claims = {claim.claim_id: claim for claim in reference_output.claims}
    allowed_window_ids = {window_id for claim in reference_output.claims for window_id in claim.evidence_window_ids}
    for binding in wire.field_bindings:
        unknown_requirements = sorted(set(binding.requirement_ids) - known_requirement_ids)
        if unknown_requirements:
            feedback.append(f"{binding.schema_path} references unknown requirement IDs: {unknown_requirements}")
        unknown_claims = sorted(set(binding.claim_ids) - set(reference_claims))
        if unknown_claims:
            feedback.append(f"{binding.schema_path} references unknown claim IDs: {unknown_claims}")
        unknown_windows = sorted(set(binding.evidence_window_ids) - allowed_window_ids)
        if unknown_windows:
            feedback.append(f"{binding.schema_path} references unknown evidence windows: {unknown_windows}")
        manifested_windows = {
            window_id
            for claim_id in binding.claim_ids
            if (claim := reference_claims.get(claim_id)) is not None
            for window_id in claim.evidence_window_ids
        }
        mismatched_windows = sorted(set(binding.evidence_window_ids) - manifested_windows)
        if mismatched_windows:
            feedback.append(f"{binding.schema_path} uses windows not manifested by its claims: {mismatched_windows}")
    return tuple(feedback)


def reference_delivery_feedback(
    output: DomainTweakReferenceAnswerOutput,
    *,
    question_packet: DomainTweakQuestionPacket,
    allowed_window_ids: frozenset[str],
    allowed_claim_ids_by_window_id: Mapping[str, frozenset[str]],
) -> tuple[str, ...]:
    """Check only mechanically defined claim and evidence identities."""
    if output.status != "finalized":
        return ("reference output was not finalized",)
    feedback: list[str] = []
    qg_claims = {claim.claim_id: claim for claim in question_packet.claims}
    reference_claims = {claim.claim_id: claim for claim in output.claims}
    reference_claim_ids = [claim.claim_id for claim in output.claims]
    if len(reference_claim_ids) != len(reference_claims):
        feedback.append("reference claim IDs must be unique")
    valid_claim_ids = {*qg_claims, DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID}
    unknown_claim_ids = sorted(set(reference_claims) - valid_claim_ids)
    if unknown_claim_ids:
        feedback.append(f"reference output contains unknown claim IDs: {unknown_claim_ids}")
    expected_answer_claim_ids = {
        claim.claim_id for claim in question_packet.claims if claim.role == "answer_determining"
    }
    actual_answer_claim_ids = {claim.claim_id for claim in output.claims if claim.role == "answer_determining"}
    if actual_answer_claim_ids != expected_answer_claim_ids:
        feedback.append("reference answer-determining claim IDs must exactly match the QG packet")
    additional_claim = reference_claims.get(DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID)
    if additional_claim is not None and additional_claim.role != "explanatory":
        feedback.append("the additional-explanation claim ID cannot determine the answer")
    if additional_claim is not None and not additional_claim.evidence_window_ids:
        feedback.append("additional-explanation claims require acquired windows")
    external_claims_without_windows = sorted(
        claim_id
        for claim_id, reference_claim in reference_claims.items()
        if (qg_claim := qg_claims.get(claim_id)) is not None
        and qg_claim.support_mode == "external_source"
        and not reference_claim.evidence_window_ids
    )
    if external_claims_without_windows:
        feedback.append(f"QG external-source claims require acquired windows: {external_claims_without_windows}")
    referenced_window_ids = {window_id for claim in output.claims for window_id in claim.evidence_window_ids}
    selected_window_ids = set(output.citation_window_ids)
    unknown_window_ids = sorted((referenced_window_ids | selected_window_ids) - allowed_window_ids)
    if unknown_window_ids:
        feedback.append(f"reference output uses unknown source windows: {unknown_window_ids}")
    claim_mismatched_windows = {
        claim.claim_id: sorted(
            window_id
            for window_id in claim.evidence_window_ids
            if window_id in allowed_window_ids
            and claim.claim_id not in allowed_claim_ids_by_window_id.get(window_id, frozenset())
        )
        for claim in output.claims
    }
    claim_mismatched_windows = {
        claim_id: window_ids for claim_id, window_ids in claim_mismatched_windows.items() if window_ids
    }
    if claim_mismatched_windows:
        feedback.append(f"source windows were not acquired for claim IDs: {claim_mismatched_windows}")
    unmanifested_citations = sorted(selected_window_ids - referenced_window_ids)
    if unmanifested_citations:
        feedback.append(f"citation windows must be attached to a reference claim: {unmanifested_citations}")
    if selected_window_ids != referenced_window_ids:
        feedback.append("citation window IDs must exactly match claim evidence windows")
    if not output.citation_window_ids:
        feedback.append("finalized reference output requires at least one citation window")
    return tuple(feedback)


def semantic_support_coverage_feedback(
    review: DomainTweakSemanticSupportReview,
    *,
    form_review: DomainTweakFormReview,
    reference_output: DomainTweakReferenceAnswerOutput,
    allowed_window_ids: frozenset[str],
    materialization: DomainTweakStructuredOutputMaterialization | None = None,
) -> tuple[str, ...]:
    """Require the gate to adjudicate the exact supplied requirement and claim sets."""
    feedback: list[str] = []
    expected_requirement_ids = {item.requirement_id for item in form_review.question_requirements}
    actual_requirement_ids = [item.requirement_id for item in review.requirement_findings]
    if len(actual_requirement_ids) != len(set(actual_requirement_ids)):
        feedback.append("semantic gate requirement finding IDs must be unique")
    if set(actual_requirement_ids) != expected_requirement_ids:
        feedback.append("semantic gate requirement finding IDs must exactly match form review")
    requirements_by_id = {item.requirement_id: item for item in form_review.question_requirements}
    source_backed_requirements_without_windows = sorted(
        finding.requirement_id
        for finding in review.requirement_findings
        if finding.support_status == "supported"
        and (requirement := requirements_by_id.get(finding.requirement_id)) is not None
        and requirement.required_relation in _SOURCE_BACKED_REQUIREMENT_RELATIONS
        and not finding.evidence_window_ids
    )
    if source_backed_requirements_without_windows:
        feedback.append(
            "source-backed requirement findings require acquired evidence windows: "
            f"{source_backed_requirements_without_windows}"
        )
    expected_claim_ids = {item.claim_id for item in reference_output.claims}
    actual_claim_ids = [item.claim_id for item in review.claim_findings]
    if len(actual_claim_ids) != len(set(actual_claim_ids)):
        feedback.append("semantic gate claim finding IDs must be unique")
    if set(actual_claim_ids) != expected_claim_ids:
        feedback.append("semantic gate claim finding IDs must exactly match reference claims")
    referenced_window_ids = {
        window_id
        for finding in (*review.requirement_findings, *review.claim_findings)
        for window_id in finding.evidence_window_ids
    }
    unknown_window_ids = sorted(referenced_window_ids - allowed_window_ids)
    if unknown_window_ids:
        feedback.append(f"semantic gate uses unknown source windows: {unknown_window_ids}")
    reference_windows_by_claim = {
        claim.claim_id: frozenset(claim.evidence_window_ids) for claim in reference_output.claims
    }
    claim_mismatched_windows = {
        finding.claim_id: sorted(
            window_id
            for window_id in finding.evidence_window_ids
            if window_id in allowed_window_ids
            and window_id not in reference_windows_by_claim.get(finding.claim_id, frozenset())
        )
        for finding in review.claim_findings
    }
    claim_mismatched_windows = {
        claim_id: window_ids for claim_id, window_ids in claim_mismatched_windows.items() if window_ids
    }
    if claim_mismatched_windows:
        feedback.append(f"semantic gate uses source windows not manifested for claim IDs: {claim_mismatched_windows}")
    missing_manifested_evidence = sorted(
        finding.claim_id
        for finding in review.claim_findings
        if finding.support_status == "supported"
        and reference_windows_by_claim.get(finding.claim_id)
        and not finding.evidence_window_ids
    )
    if missing_manifested_evidence:
        feedback.append(f"supported claim findings require manifested evidence windows: {missing_manifested_evidence}")
    if materialization is None:
        if review.structured_field_findings:
            feedback.append("plain semantic review must not contain structured field findings")
        return tuple(feedback)

    bindings = {binding.schema_path: binding for binding in materialization.field_bindings}
    finding_paths = [finding.schema_path for finding in review.structured_field_findings]
    if len(finding_paths) != len(set(finding_paths)):
        feedback.append("semantic gate structured field paths must be unique")
    if set(finding_paths) != set(bindings):
        feedback.append("semantic gate structured field paths must exactly match materialization bindings")
    for finding in review.structured_field_findings:
        binding = bindings.get(finding.schema_path)
        if binding is None:
            continue
        if set(finding.requirement_ids) != set(binding.requirement_ids):
            feedback.append(f"{finding.schema_path} requirement IDs must exactly match its binding")
        if set(finding.claim_ids) != set(binding.claim_ids):
            feedback.append(f"{finding.schema_path} claim IDs must exactly match its binding")
        if set(finding.evidence_window_ids) != set(binding.evidence_window_ids):
            feedback.append(f"{finding.schema_path} evidence windows must exactly match its binding")
    return tuple(feedback)


def _parse_model(
    text: str,
    model: type[TModel],
) -> tuple[TModel | None, DomainTweakValidationOutcome | None]:
    try:
        return model.model_validate_json(text), None
    except ValidationError as exc:
        return None, _validation_error(exc)


def _validation_error(exc: ValidationError) -> DomainTweakValidationOutcome:
    return DomainTweakValidationOutcome(
        ok=False,
        terminal_status="validation_failed",
        feedback=tuple(_format_validation_error(error) for error in exc.errors()[:5]),
        error_type=type(exc).__name__,
        error=str(exc),
    )


def _validation_failure(
    feedback: str,
    exc: Exception | None = None,
) -> DomainTweakValidationOutcome:
    return DomainTweakValidationOutcome(
        ok=False,
        terminal_status="validation_failed",
        feedback=(feedback,),
        error_type=type(exc).__name__ if exc is not None else "ValueError",
        error=str(exc) if exc is not None else feedback,
    )


def _format_validation_error(error: Mapping[str, object]) -> str:
    location = ".".join(str(item) for item in _as_sequence(error.get("loc")))
    message = str(error.get("msg") or "invalid output")
    return f"{location}: {message}" if location else message


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, tuple | list):
        return value
    return ()


__all__ = [
    "reference_delivery_feedback",
    "semantic_support_coverage_feedback",
    "validate_form_blueprint_output",
    "validate_form_review_output",
    "validate_question_generation_output",
    "validate_reference_answer_output",
    "validate_semantic_support_output",
    "validate_structured_semantic_support_output",
    "validate_structured_output_materialization",
]
