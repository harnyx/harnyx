from __future__ import annotations

from datetime import UTC, datetime

from harnyx_commons.domain_tweak_generation.prompts import (
    form_blueprint_prompt,
    form_review_prompt,
    phase_instruction,
    question_generation_prompt,
    reference_answer_prompt,
    semantic_support_prompt,
)
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkPhaseResult,
    DomainTweakReviewedQuestion,
)
from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_REQUIREMENT_CATEGORIES,
    DomainTweakClaim,
    DomainTweakEvidenceDeclaration,
    DomainTweakFormBlueprint,
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionPacket,
    DomainTweakQuestionRequirement,
    DomainTweakReferenceAnswerOutput,
    DomainTweakReferenceClaim,
    DomainTweakRequirementCategoryAudit,
)


def test_stage_instructions_assign_search_and_judgment_boundaries() -> None:
    assert "Do not search" in phase_instruction("form_blueprint")
    assert "Do not assess truth" in phase_instruction("form_review")
    assert "no-search evidence-quality gate" in phase_instruction("semantic_support_gate")
    assert "bounded correction" in phase_instruction("reference_answer_generation")


def test_question_prompt_requires_claim_bound_evidence_and_no_self_assessment() -> None:
    prompt = question_generation_prompt(_pair(), _blueprint(semantic_ambiguities=("ambiguous term",)))

    assert "Search queries are trajectory, not evidence" in prompt
    assert "every answer-determining external claim" in prompt
    assert "proxy" in prompt
    assert "Do not emit a premise self-assessment" in prompt
    assert "Do not reproduce a source-question ambiguity" in prompt


def test_form_prompts_preserve_blueprint_and_audit_every_requirement_category() -> None:
    blueprint_prompt = form_blueprint_prompt(_pair())
    review_prompt = form_review_prompt(
        _blueprint(semantic_ambiguities=("ambiguous term",)),
        _packet(),
    )

    assert "do not solve it" in blueprint_prompt
    assert "not features that the new question must reproduce" in blueprint_prompt
    assert "Do not assess whether its facts" in review_prompt
    assert "Do not reject a clearer generated question" in review_prompt
    assert "Extra context is harmless" in review_prompt
    for category in DOMAIN_TWEAK_REQUIREMENT_CATEGORIES:
        assert category in review_prompt


def test_reference_prompt_uses_acquired_content_and_forbids_workflow_narration() -> None:
    prompt = reference_answer_prompt(
        _reviewed(),
        source_packet={"initial_evidence": [{"status": "acquired", "window_id": "window-1"}]},
        question_generation_trajectory={"search_queries": ["query"]},
    )

    assert "QG-authored excerpts as locators, never as fetched evidence" in prompt
    assert "read_cached_source" in prompt
    assert "acquire_sources" in prompt
    assert "claim-bound" in prompt
    assert "call `acquire_sources` again with that claim ID" in prompt
    assert "exact union" in prompt
    assert "claim's `evidence_window_ids`" in prompt
    assert "additional-explanation ID requires an acquired window" in prompt
    assert "materially new metric" in prompt
    assert "Never discuss the question generator" in prompt


def test_semantic_gate_prompt_requires_exact_findings_without_rewriting() -> None:
    prompt = semantic_support_prompt(
        _reviewed(),
        _reference_output(),
        evidence_windows=({"window_id": "window-1", "content": "A satisfies both."},),
    )

    assert "exactly one `requirement_finding`" in prompt
    assert "exactly one `claim_finding`" in prompt
    assert "source-backed required relations require" in prompt
    assert "at least one acquired window" in prompt
    assert "Do not improve, rewrite, or complete the answer" in prompt
    assert "Authority is not claim binding" in prompt


def _pair() -> DomainTweakPairInput:
    return DomainTweakPairInput(
        pair_id="pair-001",
        deepsearchqa_form_target="Which candidates satisfy both predicates?",
        deepresearch9k_domain_target="Public results tables",
        timestamp=datetime(2026, 7, 21, tzinfo=UTC),
    )


def _blueprint(
    *, semantic_ambiguities: tuple[str, ...] = ()
) -> DomainTweakFormBlueprint:
    return DomainTweakFormBlueprint(
        status="proceed",
        operation="Filter a closed candidate set by two retrieved predicates.",
        load_bearing_invariants=("closed candidate set", "two predicates"),
        non_load_bearing_surface_features=("entity names",),
        retrieval_boundary="The question states the set; sources provide predicate values.",
        answer_shape="Exhaustive list.",
        semantic_ambiguities=semantic_ambiguities,
        no_generate_reason=None,
    )


def _packet() -> DomainTweakQuestionPacket:
    return DomainTweakQuestionPacket(
        status="ready",
        question="Which candidates satisfy both predicates?",
        short_answer="A",
        solution_steps=("Read the values.", "Intersect the qualifying sets."),
        claims=(
            DomainTweakClaim(
                claim_id="answer",
                claim="A satisfies both predicates.",
                role="answer_determining",
                support_mode="external_source",
                support_explanation="The declared page records both values.",
            ),
        ),
        evidence_declarations=(
            DomainTweakEvidenceDeclaration(
                evidence_id="evidence-1",
                source_url="https://example.com/source",
                source_title="Source",
                source_locator="Results table",
                claimed_excerpt="A satisfies both predicates.",
                supported_claim_ids=("answer",),
                support_explanation="The table directly supports the answer.",
            ),
        ),
        no_generate_reason=None,
    )


def _review() -> DomainTweakFormReview:
    return DomainTweakFormReview(
        form_match=True,
        reviewer_feedback="The filtering operation is preserved.",
        question_requirements=(
            DomainTweakQuestionRequirement(
                requirement_id="metric",
                category="metric_or_field_relation",
                requirement="Both predicates must hold.",
                required_relation="derived_calculation",
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


def _reviewed() -> DomainTweakReviewedQuestion:
    return DomainTweakReviewedQuestion(
        pair_input=_pair(),
        form_blueprint=_blueprint(),
        question_packet=_packet(),
        form_review=_review(),
        form_blueprint_result=DomainTweakAdkPhaseResult(
            phase="form_blueprint", terminal_status="validated", parsed_output=_blueprint()
        ),
        question_generation_result=DomainTweakAdkPhaseResult(
            phase="question_generation", terminal_status="validated", parsed_output=_packet()
        ),
        form_review_result=DomainTweakAdkPhaseResult(
            phase="form_review", terminal_status="validated", parsed_output=_review()
        ),
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
                support_explanation="The window records both values.",
            ),
        ),
        citation_window_ids=("window-1",),
        abandon_reason=None,
    )
