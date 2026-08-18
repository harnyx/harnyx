import pytest
from pydantic import ValidationError

from harnyx_commons.domain_tweak_generation import (
    DomainTweakBatchGenerationResult,
    GroundedQuestionDossier,
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
)


def test_batch_result_rejects_partial_success_state() -> None:
    """Future failure: a partial refill must not escape as a successful batch result."""
    with pytest.raises(ValidationError, match="finalized task count must equal target_count"):
        DomainTweakBatchGenerationResult(
            target_count=1,
            portfolio_call_count=10_000,
            slot_attempt_count=10_000,
            round_count=10_000,
            failure_counts={"reasoning_no_generate": 10_000},
        )

    assert "discarded_candidates" not in DomainTweakBatchGenerationResult.model_fields
    assert "rejected_attempts" not in DomainTweakBatchGenerationResult.model_fields
    assert "completed" not in DomainTweakBatchGenerationResult.model_fields


def test_no_generate_dossier_requires_typed_terminal_cause() -> None:
    """Future failure: dossier attribution must not be reconstructed from unrelated workspace history."""
    with pytest.raises(ValidationError, match="requires failure_class"):
        GroundedQuestionDossier(
            status="no_generate",
            failure_reason="route was not viable",
        )

    dossier = GroundedQuestionDossier(
        status="no_generate",
        failure_reason="route was not viable",
        failure_class="reasoning_no_generate",
    )
    assert dossier.failure_class == "reasoning_no_generate"


def test_source_no_generate_requires_the_exact_failed_fetch_id() -> None:
    """Future failure: a prior failure class alone must not identify the dossier's terminal blocker."""
    with pytest.raises(ValidationError, match="requires source_failure_id"):
        GroundedQuestionDossier(
            status="no_generate",
            failure_reason="the required public document could not be fetched",
            failure_class="source_unavailable",
        )

    dossier = GroundedQuestionDossier(
        status="no_generate",
        failure_reason="the required public document could not be fetched",
        failure_class="source_unavailable",
        source_failure_id="source_failure:3",
    )
    assert dossier.source_failure_id == "source_failure:3"


def test_reasoning_no_generate_forbids_a_source_failure_id() -> None:
    """Future failure: a model-decided dead end must not be attributed to an incidental fetch attempt."""
    with pytest.raises(ValidationError, match="reasoning_no_generate cannot contain source_failure_id"):
        GroundedQuestionDossier(
            status="no_generate",
            failure_reason="the explored route cannot support the requested relationship",
            failure_class="reasoning_no_generate",
            source_failure_id="source_failure:1",
        )


def test_ready_question_dossier_requires_every_frozen_semantic_output() -> None:
    """Future failure: productization must not loss-compress the single ultra QG result."""
    required = {
        "subject",
        "route_summary",
        "question",
        "answers",
        "requirements",
        "source_facts",
        "derivation",
        "why_not_one_page",
        "substantive_final_condition",
    }
    assert required <= set(GroundedQuestionDossier.model_fields)

    with pytest.raises(ValidationError, match="one-page explanation"):
        GroundedQuestionDossier(
            status="ready",
            subject="Public roster",
            route_summary="Join the roster to status records",
            question="Which entry qualifies?",
            answers=[{"answer_id": "A1", "value": "Alpha"}],
            requirements=[{"description": "Check every roster entry"}],
            source_facts=[{"statement": "Alpha qualifies", "evidence_ids": ["E1"]}],
            derivation="Enumerate, join, and filter",
            substantive_final_condition="The status condition removes one entry",
        )


def test_no_generate_question_dossier_rejects_partial_semantics() -> None:
    """Future failure: a blocker must not be emitted beside a misleading partial question."""
    with pytest.raises(ValidationError, match="cannot contain question semantics"):
        GroundedQuestionDossier(
            status="no_generate",
            question="Which entry qualifies?",
            failure_reason="The complete roster is unavailable",
            failure_class="reasoning_no_generate",
        )


def test_ready_dossier_requires_one_coherent_response_mode_contract() -> None:
    """Future failure: QG must not emit an ambiguous or half-structured public answer contract."""
    common = {
        "status": "ready",
        "subject": "Public roster",
        "route_summary": "Join the roster to status records",
        "question": "Which entry qualifies?",
        "answers": [{"answer_id": "A1", "value": "Alpha"}],
        "requirements": [{"description": "Check every roster entry"}],
        "source_facts": [{"statement": "Alpha qualifies", "evidence_ids": ["E1"]}],
        "derivation": "Enumerate, join, and filter",
        "why_not_one_page": "The status record is separate from the roster",
        "substantive_final_condition": "The status condition removes one entry",
    }

    with pytest.raises(ValidationError, match="requires response_mode"):
        GroundedQuestionDossier(**common)
    with pytest.raises(ValidationError, match="plain_text dossier cannot contain structured"):
        GroundedQuestionDossier(
            **common,
            response_mode="plain_text",
            output_schema_json='{"type":"object"}',
        )
    with pytest.raises(ValidationError, match="structured dossier requires"):
        GroundedQuestionDossier(**common, response_mode="structured")

    structured = GroundedQuestionDossier(
        **common,
        response_mode="structured",
        output_schema_json='{"type":"object"}',
        structured_answer_json='{"answer":"Alpha"}',
    )
    assert structured.response_mode == "structured"


def test_reference_proof_enforces_public_citation_position_limit() -> None:
    """Future failure: accepted references must never be silently truncated by the 200-position judge boundary."""
    common = {
        "status": "finalized",
        "answer_text": "Alpha is the published result [[1]].",
        "answers": (ReferenceAnswerSelection(answer_id="A1"),),
        "proof_steps": (
            ProofStep(step_id="S1", statement="Alpha is published.", kind="supported", evidence_ids=("E1",)),
        ),
    }

    accepted = ReferenceProof(**common, citation_evidence_ids=tuple("E1" for _ in range(200)))

    assert len(accepted.citation_evidence_ids) == 200
    with pytest.raises(ValidationError, match="at most 200"):
        ReferenceProof(**common, citation_evidence_ids=tuple("E1" for _ in range(201)))
