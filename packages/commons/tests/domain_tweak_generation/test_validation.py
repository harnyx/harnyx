from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

import pytest
from pydantic import ValidationError

from harnyx_commons.domain_tweak_generation.source_evidence import (
    DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID,
)
from harnyx_commons.domain_tweak_generation.validation import (
    reference_delivery_feedback,
    semantic_support_coverage_feedback,
    validate_question_generation_output,
)
from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_REQUIREMENT_CATEGORIES,
    DomainTweakClaim,
    DomainTweakClaimFinding,
    DomainTweakEvidenceDeclaration,
    DomainTweakFormReview,
    DomainTweakQuestionPacket,
    DomainTweakQuestionRequirement,
    DomainTweakReferenceAnswerOutput,
    DomainTweakReferenceClaim,
    DomainTweakRequiredRelation,
    DomainTweakRequirementCategoryAudit,
    DomainTweakRequirementFinding,
    DomainTweakSemanticSupportReview,
)


def test_question_validation_is_strict_json_not_markdown_recovery() -> None:
    text = json.dumps(_question_packet().model_dump(mode="json"))

    assert validate_question_generation_output(text).ok
    rejected = validate_question_generation_output(f"```json\n{text}\n```")
    assert not rejected.ok
    assert rejected.terminal_status == "validation_failed"


def test_reference_delivery_requires_exact_qg_claim_ids_and_acquired_windows() -> None:
    output = _reference_output(claim_id="invented", window_id="window-missing")

    feedback = reference_delivery_feedback(
        output,
        question_packet=_question_packet(),
        allowed_window_ids=frozenset({"window-allowed"}),
        allowed_claim_ids_by_window_id={"window-allowed": frozenset({"answer"})},
    )

    assert any("claim IDs" in item for item in feedback)
    assert any("unknown source windows" in item for item in feedback)


def test_reference_delivery_requires_windows_for_qg_external_source_claims() -> None:
    feedback = reference_delivery_feedback(
        _reference_output(claim_id="answer", window_id=None),
        question_packet=_question_packet(),
        allowed_window_ids=frozenset(),
        allowed_claim_ids_by_window_id={},
    )

    assert any("external-source claims require acquired windows" in item for item in feedback)


def test_reference_delivery_allows_source_free_qg_derivations() -> None:
    feedback = reference_delivery_feedback(
        _reference_output(claim_id="answer", window_id=None),
        question_packet=_question_packet(support_mode="logical_or_mathematical_derivation"),
        allowed_window_ids=frozenset(),
        allowed_claim_ids_by_window_id={},
    )

    assert not any("external-source claims require acquired windows" in item for item in feedback)
    assert any("at least one citation window" in item for item in feedback)


def test_reference_delivery_rejects_window_acquired_for_a_different_claim() -> None:
    feedback = reference_delivery_feedback(
        _reference_output(claim_id="answer", window_id="window-other"),
        question_packet=_question_packet(),
        allowed_window_ids=frozenset({"window-other"}),
        allowed_claim_ids_by_window_id={"window-other": frozenset({"other"})},
    )

    assert any("not acquired for claim" in item for item in feedback)


def test_reference_delivery_requires_evidence_for_additional_explanation_claims() -> None:
    output = DomainTweakReferenceAnswerOutput(
        status="finalized",
        answer_disposition="unchanged",
        proposed_short_answer="A",
        reference_answer_text="A satisfies both predicates for the stated reason.",
        claims=(
            DomainTweakReferenceClaim(
                claim_id="answer",
                claim="A satisfies both predicates.",
                role="answer_determining",
                evidence_window_ids=("window-answer",),
                support_explanation="The acquired window supports the answer.",
            ),
            DomainTweakReferenceClaim(
                claim_id=DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID,
                claim="The predicates are independently meaningful.",
                role="explanatory",
                evidence_window_ids=(),
                support_explanation="This is additional factual context.",
            ),
        ),
        citation_window_ids=("window-answer",),
        abandon_reason=None,
    )

    feedback = reference_delivery_feedback(
        output,
        question_packet=_question_packet(),
        allowed_window_ids=frozenset({"window-answer"}),
        allowed_claim_ids_by_window_id={"window-answer": frozenset({"answer"})},
    )

    assert any("additional-explanation claims require acquired windows" in item for item in feedback)


def test_semantic_pass_requires_exact_requirement_and_reference_claim_id_sets() -> None:
    form_review = _form_review()
    output = _reference_output(claim_id="answer", window_id="window-allowed")
    review = DomainTweakSemanticSupportReview(
        status="pass",
        requirement_findings=(),
        claim_findings=(
            DomainTweakClaimFinding(
                claim_id="answer",
                support_status="supported",
                evidence_window_ids=("window-allowed",),
                explanation="The window supports the answer claim.",
            ),
        ),
        unmanifested_material_claims=(),
        abandon_reason=None,
    )

    feedback = semantic_support_coverage_feedback(
        review,
        form_review=form_review,
        reference_output=output,
        allowed_window_ids=frozenset({"window-allowed"}),
    )

    assert any("requirement finding IDs" in item for item in feedback)


def test_semantic_pass_accepts_complete_supported_findings() -> None:
    form_review = _form_review()
    output = _reference_output(claim_id="answer", window_id="window-allowed")
    review = DomainTweakSemanticSupportReview(
        status="pass",
        requirement_findings=(
            DomainTweakRequirementFinding(
                requirement_id="metric",
                support_status="supported",
                evidence_window_ids=("window-allowed",),
                explanation="The window establishes both predicates.",
            ),
        ),
        claim_findings=(
            DomainTweakClaimFinding(
                claim_id="answer",
                support_status="supported",
                evidence_window_ids=("window-allowed",),
                explanation="The window establishes the answer.",
            ),
        ),
        unmanifested_material_claims=(),
        abandon_reason=None,
    )

    assert not semantic_support_coverage_feedback(
        review,
        form_review=form_review,
        reference_output=output,
        allowed_window_ids=frozenset({"window-allowed"}),
    )


def test_semantic_gate_rejects_a_window_manifested_for_another_claim() -> None:
    form_review = _form_review()
    output = _reference_output(claim_id="answer", window_id="window-answer")
    review = DomainTweakSemanticSupportReview(
        status="pass",
        requirement_findings=(
            DomainTweakRequirementFinding(
                requirement_id="metric",
                support_status="supported",
                evidence_window_ids=("window-answer",),
                explanation="The answer window supports the requirement.",
            ),
        ),
        claim_findings=(
            DomainTweakClaimFinding(
                claim_id="answer",
                support_status="supported",
                evidence_window_ids=("window-other",),
                explanation="A different claim's window was used.",
            ),
        ),
        unmanifested_material_claims=(),
        abandon_reason=None,
    )

    feedback = semantic_support_coverage_feedback(
        review,
        form_review=form_review,
        reference_output=output,
        allowed_window_ids=frozenset({"window-answer", "window-other"}),
    )

    assert any("not manifested for claim" in item for item in feedback)


def test_semantic_gate_requires_evidence_for_a_manifested_sourced_claim() -> None:
    output = _reference_output(claim_id="answer", window_id="window-answer")
    review = DomainTweakSemanticSupportReview(
        status="pass",
        requirement_findings=(
            DomainTweakRequirementFinding(
                requirement_id="metric",
                support_status="supported",
                evidence_window_ids=("window-answer",),
                explanation="The window supports the requirement.",
            ),
        ),
        claim_findings=(
            DomainTweakClaimFinding(
                claim_id="answer",
                support_status="supported",
                evidence_window_ids=(),
                explanation="The claim is supported without naming its manifested evidence.",
            ),
        ),
        unmanifested_material_claims=(),
        abandon_reason=None,
    )

    feedback = semantic_support_coverage_feedback(
        review,
        form_review=_form_review(),
        reference_output=output,
        allowed_window_ids=frozenset({"window-answer"}),
    )

    assert any("require manifested evidence windows" in item for item in feedback)


@pytest.mark.parametrize(
    "required_relation",
    (
        "direct_fact",
        "field_equivalence",
        "exhaustive_set",
        "exact_cardinality",
        "absence",
        "upper_or_lower_bound",
    ),
)
def test_semantic_gate_requires_evidence_for_source_backed_requirements(
    required_relation: DomainTweakRequiredRelation,
) -> None:
    output = _reference_output(claim_id="answer", window_id="window-answer")
    review = DomainTweakSemanticSupportReview(
        status="pass",
        requirement_findings=(
            DomainTweakRequirementFinding(
                requirement_id="metric",
                support_status="supported",
                evidence_window_ids=(),
                explanation="The requirement is supported.",
            ),
        ),
        claim_findings=(
            DomainTweakClaimFinding(
                claim_id="answer",
                support_status="supported",
                evidence_window_ids=("window-answer",),
                explanation="The window supports the answer claim.",
            ),
        ),
        unmanifested_material_claims=(),
        abandon_reason=None,
    )

    feedback = semantic_support_coverage_feedback(
        review,
        form_review=_form_review(required_relation=required_relation),
        reference_output=output,
        allowed_window_ids=frozenset({"window-answer"}),
    )

    assert any(
        "source-backed requirement findings require acquired evidence windows" in item
        for item in feedback
    )


@pytest.mark.parametrize("required_relation", ("derived_calculation", "other"))
def test_semantic_gate_leaves_source_free_requirements_to_semantic_judgment(
    required_relation: DomainTweakRequiredRelation,
) -> None:
    output = _reference_output(claim_id="answer", window_id="window-answer")
    review = DomainTweakSemanticSupportReview(
        status="pass",
        requirement_findings=(
            DomainTweakRequirementFinding(
                requirement_id="metric",
                support_status="supported",
                evidence_window_ids=(),
                explanation="The requirement follows from supported premises.",
            ),
        ),
        claim_findings=(
            DomainTweakClaimFinding(
                claim_id="answer",
                support_status="supported",
                evidence_window_ids=("window-answer",),
                explanation="The window supports the answer claim.",
            ),
        ),
        unmanifested_material_claims=(),
        abandon_reason=None,
    )

    feedback = semantic_support_coverage_feedback(
        review,
        form_review=_form_review(required_relation=required_relation),
        reference_output=output,
        allowed_window_ids=frozenset({"window-answer"}),
    )

    assert not any(
        "source-backed requirement findings require acquired evidence windows" in item
        for item in feedback
    )


def test_reference_delivery_requires_every_claim_window_to_be_cited() -> None:
    output = DomainTweakReferenceAnswerOutput(
        status="finalized",
        answer_disposition="unchanged",
        proposed_short_answer="A",
        reference_answer_text="A satisfies both predicates.",
        claims=(
            DomainTweakReferenceClaim(
                claim_id="answer",
                claim="A satisfies both predicates.",
                role="answer_determining",
                evidence_window_ids=("window-one", "window-two"),
                support_explanation="The acquired windows jointly support the claim.",
            ),
        ),
        citation_window_ids=("window-one",),
        abandon_reason=None,
    )

    feedback = reference_delivery_feedback(
        output,
        question_packet=_question_packet(),
        allowed_window_ids=frozenset({"window-one", "window-two"}),
        allowed_claim_ids_by_window_id={
            "window-one": frozenset({"answer"}),
            "window-two": frozenset({"answer"}),
        },
    )

    assert any("must exactly match claim evidence windows" in item for item in feedback)


@pytest.mark.parametrize(
    "model",
    (
        lambda: DomainTweakReferenceClaim(
            claim_id="answer",
            claim="A satisfies both predicates.",
            role="answer_determining",
            evidence_window_ids=("window-one", "window-one"),
            support_explanation="The window supports the answer.",
        ),
        lambda: DomainTweakRequirementFinding(
            requirement_id="metric",
            support_status="supported",
            evidence_window_ids=("window-one", "window-one"),
            explanation="The window supports the requirement.",
        ),
        lambda: DomainTweakClaimFinding(
            claim_id="answer",
            support_status="supported",
            evidence_window_ids=("window-one", "window-one"),
            explanation="The window supports the claim.",
        ),
    ),
)
def test_claim_and_finding_contracts_reject_duplicate_window_ids(
    model: Callable[[], object],
) -> None:
    with pytest.raises(ValidationError, match="evidence window IDs must be unique"):
        model()


def _question_packet(
    *,
    support_mode: Literal[
        "external_source", "logical_or_mathematical_derivation"
    ] = "external_source",
) -> DomainTweakQuestionPacket:
    return DomainTweakQuestionPacket(
        status="ready",
        question="Which candidate satisfies both predicates?",
        short_answer="A",
        solution_steps=("Apply both predicates.",),
        claims=(
            DomainTweakClaim(
                claim_id="answer",
                claim="A satisfies both predicates.",
                role="answer_determining",
                support_mode=support_mode,
                support_explanation="The cited source records both values.",
            ),
        ),
        evidence_declarations=(
            DomainTweakEvidenceDeclaration(
                evidence_id="evidence-1",
                source_url="https://example.com/source",
                source_title="Source",
                source_locator=None,
                claimed_excerpt="A satisfies both predicates.",
                supported_claim_ids=("answer",),
                support_explanation="The excerpt records both values.",
            ),
        ),
        no_generate_reason=None,
    )


def _form_review(
    *,
    required_relation: DomainTweakRequiredRelation = "derived_calculation",
) -> DomainTweakFormReview:
    return DomainTweakFormReview(
        form_match=True,
        reviewer_feedback="The operation is preserved.",
        question_requirements=(
            DomainTweakQuestionRequirement(
                requirement_id="metric",
                category="metric_or_field_relation",
                requirement="Apply both predicates.",
                required_relation=required_relation,
            ),
        ),
        requirement_category_audit=tuple(
            DomainTweakRequirementCategoryAudit(
                category=category,
                present=category == "metric_or_field_relation",
                explanation=("Present." if category == "metric_or_field_relation" else "Absent."),
            )
            for category in DOMAIN_TWEAK_REQUIREMENT_CATEGORIES
        ),
    )


def _reference_output(*, claim_id: str, window_id: str | None) -> DomainTweakReferenceAnswerOutput:
    window_ids = (window_id,) if window_id is not None else ()
    return DomainTweakReferenceAnswerOutput(
        status="finalized",
        answer_disposition="unchanged",
        proposed_short_answer="A",
        reference_answer_text="A satisfies both predicates.",
        claims=(
            DomainTweakReferenceClaim(
                claim_id=claim_id,
                claim="A satisfies both predicates.",
                role="answer_determining",
                evidence_window_ids=window_ids,
                support_explanation="The acquired window supports the claim.",
            ),
        ),
        citation_window_ids=window_ids,
        abandon_reason=None,
    )
