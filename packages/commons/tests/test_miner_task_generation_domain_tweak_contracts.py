from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_MAX_DECLARED_EVIDENCE_URLS,
    DOMAIN_TWEAK_REQUIREMENT_CATEGORIES,
    DomainTweakClaim,
    DomainTweakEvidenceDeclaration,
    DomainTweakFormBlueprint,
    DomainTweakFormReview,
    DomainTweakQuestionPacket,
    DomainTweakQuestionRequirement,
    DomainTweakReferenceAnswerOutput,
    DomainTweakReferenceClaim,
    DomainTweakRequirementCategoryAudit,
)


def test_question_packet_caps_declared_evidence_at_one_extract_batch() -> None:
    declarations = tuple(
        DomainTweakEvidenceDeclaration(
            evidence_id=f"evidence-{index}",
            source_url=f"https://example.com/source-{index}",
            source_title="Example source",
            source_locator=None,
            claimed_excerpt="A satisfies both predicates.",
            supported_claim_ids=("answer",),
            support_explanation="The excerpt directly records the answer.",
        )
        for index in range(DOMAIN_TWEAK_MAX_DECLARED_EVIDENCE_URLS + 1)
    )

    with pytest.raises(ValidationError, match="at most 20 items"):
        _question_packet(evidence_declarations=declarations)


def test_question_packet_schema_exposes_declared_evidence_limit_to_adk() -> None:
    schema = DomainTweakQuestionPacket.model_json_schema()

    assert (
        schema["properties"]["evidence_declarations"]["maxItems"]
        == DOMAIN_TWEAK_MAX_DECLARED_EVIDENCE_URLS
    )


def test_ready_question_packet_requires_declared_evidence_for_external_claims() -> None:
    with pytest.raises(ValidationError, match="external_source claims require evidence declarations"):
        _question_packet(evidence_declarations=())


def test_logical_claim_can_omit_external_evidence() -> None:
    packet = _question_packet(
        claims=(
            DomainTweakClaim(
                claim_id="answer",
                claim="A is the only candidate satisfying both predicates.",
                role="answer_determining",
                support_mode="logical_or_mathematical_derivation",
                support_explanation="It follows by intersecting the two stated sets.",
            ),
        ),
        evidence_declarations=(),
    )

    assert packet.evidence_declarations == ()


def test_accepted_form_review_requires_complete_consistent_category_audit() -> None:
    requirements = (
        DomainTweakQuestionRequirement(
            requirement_id="metric",
            category="metric_or_field_relation",
            requirement="Apply both stated predicates.",
            required_relation="derived_calculation",
        ),
    )
    with pytest.raises(ValidationError, match="cover every category exactly once"):
        DomainTweakFormReview(
            form_match=True,
            reviewer_feedback="The operation is preserved.",
            question_requirements=requirements,
            requirement_category_audit=(),
        )

    audit = _category_audit(present={"scope"})
    with pytest.raises(ValidationError, match="disagrees with requirements"):
        DomainTweakFormReview(
            form_match=True,
            reviewer_feedback="The operation is preserved.",
            question_requirements=requirements,
            requirement_category_audit=audit,
        )


def test_reference_output_has_no_premise_self_assessment_field() -> None:
    payload = _reference_output().model_dump(mode="json")
    payload["premise_assessment"] = "valid"

    with pytest.raises(ValidationError, match="premise_assessment"):
        DomainTweakReferenceAnswerOutput.model_validate(payload)


def test_form_blueprint_no_generate_cannot_smuggle_analysis_fields() -> None:
    with pytest.raises(ValidationError, match="must not contain analysis fields"):
        DomainTweakFormBlueprint(
            status="no_generate",
            operation="filter",
            load_bearing_invariants=(),
            non_load_bearing_surface_features=(),
            retrieval_boundary=None,
            answer_shape=None,
            semantic_ambiguities=(),
            no_generate_reason="The source form is not reproducible.",
        )


def _question_packet(
    *,
    claims: tuple[DomainTweakClaim, ...] | None = None,
    evidence_declarations: tuple[DomainTweakEvidenceDeclaration, ...] | None = None,
) -> DomainTweakQuestionPacket:
    resolved_claims = claims or (
        DomainTweakClaim(
            claim_id="answer",
            claim="A satisfies both predicates.",
            role="answer_determining",
            support_mode="external_source",
            support_explanation="The source records both required values.",
        ),
    )
    resolved_evidence = (
        (
            DomainTweakEvidenceDeclaration(
                evidence_id="evidence-1",
                source_url="https://example.com/source",
                source_title="Example source",
                source_locator=None,
                claimed_excerpt="A satisfies both predicates.",
                supported_claim_ids=("answer",),
                support_explanation="The excerpt directly records the answer.",
            ),
        )
        if evidence_declarations is None
        else evidence_declarations
    )
    return DomainTweakQuestionPacket(
        status="ready",
        question="Which candidate satisfies both predicates?",
        short_answer="A",
        solution_steps=("Identify the candidate universe.", "Apply both predicates."),
        claims=resolved_claims,
        evidence_declarations=resolved_evidence,
        no_generate_reason=None,
    )


def _category_audit(*, present: set[str]) -> tuple[DomainTweakRequirementCategoryAudit, ...]:
    return tuple(
        DomainTweakRequirementCategoryAudit(
            category=category,
            present=category in present,
            explanation=("Present in the question." if category in present else "Not present."),
        )
        for category in DOMAIN_TWEAK_REQUIREMENT_CATEGORIES
    )


def _reference_output() -> DomainTweakReferenceAnswerOutput:
    return DomainTweakReferenceAnswerOutput(
        status="finalized",
        answer_disposition="unchanged",
        proposed_short_answer="A",
        reference_answer_text="A satisfies both predicates.",
        claims=(
            DomainTweakReferenceClaim(
                claim_id="answer",
                claim="A satisfies both predicates.",
                role="answer_determining",
                evidence_window_ids=("window-1",),
                support_explanation="The acquired window directly supports the claim.",
            ),
        ),
        citation_window_ids=("window-1",),
        abandon_reason=None,
    )
