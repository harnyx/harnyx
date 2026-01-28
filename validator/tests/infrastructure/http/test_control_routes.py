from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from caster_commons.domain.claim import MinerTaskClaim, ReferenceAnswer, Rubric
from caster_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from caster_commons.domain.tool_usage import (
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from caster_commons.domain.verdict import BINARY_VERDICT_OPTIONS
from caster_validator.application.dto.evaluation import (
    EvaluationOutcome,
    MinerTaskResult,
    TokenUsageSummary,
)
from caster_validator.application.services.evaluation_scoring import EvaluationScore
from caster_validator.domain.evaluation import MinerAnswer, MinerCriterionEvaluation
from caster_validator.infrastructure.http.routes import ValidatorControlDeps, add_control_routes
from caster_validator.infrastructure.state.run_progress import RunProgressSnapshot


def _create_test_app(provider: DemoControlDependencyProvider) -> FastAPI:
    app = FastAPI()
    add_control_routes(app, provider)
    return app


class StubAcceptBatch:
    def execute(self, _: object) -> None:
        return None


class StubStatusProvider:
    def snapshot(self) -> dict[str, object]:
        return {"status": "ok"}


class FakeProgressTracker:
    def __init__(self, *, snapshot: RunProgressSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, _: UUID) -> RunProgressSnapshot:
        return self._snapshot


class DemoControlDependencyProvider:
    def __init__(self, *, snapshot: RunProgressSnapshot) -> None:
        self._deps = ValidatorControlDeps(
            accept_batch=StubAcceptBatch(),
            status_provider=StubStatusProvider(),
            auth=_allow_all_auth,
            progress_tracker=FakeProgressTracker(snapshot=snapshot),
        )

    def __call__(self) -> ValidatorControlDeps:
        return self._deps


def _allow_all_auth(_: Request, __: bytes) -> str:
    return "caller"


def _make_task_result(*, batch_id: UUID) -> MinerTaskResult:
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

    total_tool_usage = ToolUsageSummary(
        search_tool=SearchToolUsageSummary(call_count=2, cost=0.005),
        search_tool_cost=0.005,
        llm=LlmUsageSummary(
            call_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.02,
            providers={
                "openai": {
                    "openai/gpt-oss-20b": LlmModelUsageCost(
                        usage=LlmUsageTotals(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                            call_count=1,
                        ),
                        cost=0.02,
                    )
                }
            },
        ),
        llm_cost=0.02,
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
        usage=TokenUsageSummary.empty(),
        total_tool_usage=total_tool_usage,
    )

    issued_at = datetime.now(UTC)
    session = Session(
        session_id=evaluation.session_id,
        uid=evaluation.uid,
        claim_id=claim.claim_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        usage=SessionUsage(total_cost_usd=0.025),
    )

    return MinerTaskResult(
        batch_id=batch_id,
        validator_uid=4,
        outcome=outcome,
        session=session,
    )


def test_progress_endpoint_includes_total_tool_usage() -> None:
    batch_id = uuid4()
    result = _make_task_result(batch_id=batch_id)
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 1,
        "completed": 1,
        "remaining": 0,
        "miner_task_results": (result,),
    }

    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/progress")

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"] == str(batch_id)
    assert body["total"] == 1
    assert body["completed"] == 1
    assert body["remaining"] == 0

    tool_usage = body["miner_task_results"][0]["total_tool_usage"]
    assert tool_usage["search_tool_cost"] == pytest.approx(0.005)
    assert tool_usage["llm_cost"] == pytest.approx(0.02)

    llm = tool_usage["llm"]
    assert llm["providers"]["openai"]["openai/gpt-oss-20b"]["cost"] == pytest.approx(0.02)

