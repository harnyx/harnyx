from uuid import UUID

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain_tweak_generation import (
    DomainTweakBatchGenerationResult,
    DomainTweakFinalizedTask,
    GroundedQuestionDossier,
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
    SlotAttemptEvent,
)


def _finalized(*, structured: bool = False) -> DomainTweakFinalizedTask:
    output_schema = None
    if structured:
        output_schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }
    return DomainTweakFinalizedTask(
        task=MinerTask(
            task_id=UUID(int=1),
            query=Query(text="question", output_schema=output_schema),
            reference_answer=ReferenceAnswer(text='{"value":1}' if structured else "answer"),
        )
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
            required_response_mode_counts={"plain_text": 0, "structured": 0},
            finalized_response_mode_counts={"plain_text": 0, "structured": 0},
            failed_slot_attempt_counts_by_required_response_mode={"plain_text": 10_000, "structured": 0},
        )

    assert "discarded_candidates" not in DomainTweakBatchGenerationResult.model_fields
    assert "rejected_attempts" not in DomainTweakBatchGenerationResult.model_fields
    assert "completed" not in DomainTweakBatchGenerationResult.model_fields


def test_attempt_and_batch_contracts_require_complete_response_mode_accounting() -> None:
    """Future failure: sampled requirements must not be inferred from accepted tasks after failures."""
    assert "required_response_mode" in SlotAttemptEvent.model_fields
    assert "actual_response_mode" in SlotAttemptEvent.model_fields
    assert {
        "required_response_mode_counts",
        "finalized_response_mode_counts",
        "failed_slot_attempt_counts_by_required_response_mode",
    } <= set(DomainTweakBatchGenerationResult.model_fields)


def test_batch_result_derives_finalized_mode_counts_from_task_output_schemas() -> None:
    """Future failure: reported actual counts must not be copied from sampled slot requirements."""
    with pytest.raises(ValidationError, match="must match finalized task output schemas"):
        DomainTweakBatchGenerationResult(
            target_count=1,
            finalized_tasks=(_finalized(),),
            required_response_mode_counts={"plain_text": 0, "structured": 1},
            finalized_response_mode_counts={"plain_text": 0, "structured": 1},
            failed_slot_attempt_counts_by_required_response_mode={"plain_text": 0, "structured": 0},
        )


def test_batch_result_rejects_finalized_mode_that_does_not_match_sampled_requirement() -> None:
    """Future failure: a wrong-mode task must not make a batch successful even when totals match."""
    with pytest.raises(ValidationError, match="must equal required response mode counts"):
        DomainTweakBatchGenerationResult(
            target_count=1,
            finalized_tasks=(_finalized(),),
            required_response_mode_counts={"plain_text": 0, "structured": 1},
            finalized_response_mode_counts={"plain_text": 1, "structured": 0},
            failed_slot_attempt_counts_by_required_response_mode={"plain_text": 0, "structured": 1},
        )


def test_batch_result_rejects_failed_mode_counts_that_do_not_cover_failed_attempts() -> None:
    """Future failure: successful result accounting must not lose failed slot-attempt attribution."""
    with pytest.raises(ValidationError, match="slot_attempt_count minus target_count"):
        DomainTweakBatchGenerationResult(
            target_count=1,
            finalized_tasks=(_finalized(),),
            slot_attempt_count=3,
            failure_counts={"contract_invalid": 2},
            required_response_mode_counts={"plain_text": 1, "structured": 0},
            finalized_response_mode_counts={"plain_text": 1, "structured": 0},
            failed_slot_attempt_counts_by_required_response_mode={"plain_text": 0, "structured": 0},
        )


def test_batch_result_rejects_failure_class_counts_that_omit_failed_attempts() -> None:
    """Future failure: failure-class and required-mode accounting must describe the same failed attempts."""
    with pytest.raises(ValidationError, match="must equal failure_counts total"):
        DomainTweakBatchGenerationResult(
            target_count=1,
            finalized_tasks=(_finalized(),),
            slot_attempt_count=2,
            required_response_mode_counts={"plain_text": 1, "structured": 0},
            finalized_response_mode_counts={"plain_text": 1, "structured": 0},
            failed_slot_attempt_counts_by_required_response_mode={"plain_text": 1, "structured": 0},
        )


def test_finalized_attempt_rejects_wrong_actual_response_mode() -> None:
    """Future failure: an attempt event must not describe a wrong-mode candidate as finalized."""
    with pytest.raises(ValidationError, match="must match required_response_mode"):
        SlotAttemptEvent(
            attempt_id="attempt-1",
            round_index=1,
            output_slot=0,
            required_response_mode="structured",
            actual_response_mode="plain_text",
            outcome="finalized",
            terminal_stage="audit",
            elapsed_ms=1,
        )


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


@pytest.mark.parametrize(
    ("answer_text", "structured_answer_json"),
    [
        ("Alpha is the published result.", None),
    ],
)
def test_finalized_reference_proof_requires_a_public_citation_position(
    answer_text: str | None,
    structured_answer_json: str | None,
) -> None:
    """Future failure: finalized public answers must not be accepted with only private proof evidence."""
    with pytest.raises(ValidationError, match="at least one public citation position"):
        ReferenceProof(
            status="finalized",
            answer_text=answer_text,
            citation_evidence_ids=(),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(
                    step_id="S1",
                    statement="Alpha is published.",
                    kind="supported",
                    evidence_ids=("E1",),
                ),
            ),
            structured_answer_json=structured_answer_json,
        )

    giveup = ReferenceProof(
        status="giveup",
        answer_text=None,
        citation_evidence_ids=(),
        structured_answer_json=None,
        giveup_reason="No public source establishes the required answer.",
    )
    assert giveup.status == "giveup"
