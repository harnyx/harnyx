from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from caster_commons.domain.claim import MinerTaskClaim, ReferenceAnswer, Rubric
from caster_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from caster_commons.domain.verdict import BINARY_VERDICT_OPTIONS
from caster_validator.application.dto.evaluation import (
    EvaluationOutcome,
    MinerTaskResult,
    TokenUsageSummary,
)
from caster_validator.application.services.evaluation_scoring import EvaluationScore
from caster_validator.domain.evaluation import MinerAnswer, MinerCriterionEvaluation
from caster_validator.infrastructure.state.evaluation_record import InMemoryEvaluationRecordStore


def test_in_memory_store_records_miner_task_results() -> None:
    store = InMemoryEvaluationRecordStore()

    claim = MinerTaskClaim(
        claim_id=uuid4(),
        text="Claim text",
        rubric=Rubric(
            title="Accuracy",
            description="Assess accuracy.",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
    )

    evaluation = MinerCriterionEvaluation(
        criterion_evaluation_id=uuid4(),
        session_id=uuid4(),
        uid=7,
        artifact_id=uuid4(),
        claim_id=claim.claim_id,
        rubric=claim.rubric,
        miner_answer=MinerAnswer(verdict=1, justification="ok", citations=()),
        completed_at=datetime.now(UTC),
    )
    outcome = EvaluationOutcome(
        criterion_evaluation=evaluation,
        score=EvaluationScore(
            verdict_score=1.0,
            support_score=1.0,
            justification_pass=True,
            failed_citation_ids=(),
            grader_rationale="pass",
        ),
        tool_receipts=(),
        usage=TokenUsageSummary.from_totals(
            {
                "search": {
                    "web": LlmUsageTotals(
                        prompt_tokens=2,
                        completion_tokens=3,
                        total_tokens=5,
                        call_count=1,
                    ),
                },
            },
        ),
    )

    issued_at = datetime.now(UTC)
    session = Session(
        session_id=evaluation.session_id,
        uid=evaluation.uid,
        claim_id=claim.claim_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=0.1,
        usage=SessionUsage(
            llm_usage_totals={
                "search": {
                    "web": LlmUsageTotals(
                        prompt_tokens=2,
                        completion_tokens=3,
                        total_tokens=5,
                        call_count=1,
                    ),
                },
            },
            total_cost_usd=0.0,
        ),
    )

    result = MinerTaskResult(
        batch_id=uuid4(),
        validator_uid=4,
        outcome=outcome,
        session=session,
    )

    store.record(result)

    records = store.records()
    assert records == (result,)
