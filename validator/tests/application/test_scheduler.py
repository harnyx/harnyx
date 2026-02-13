from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.claim import FeedSearchContext, MinerTaskClaim, ReferenceAnswer, Rubric
from caster_commons.domain.session import LlmUsageTotals
from caster_commons.domain.verdict import VerdictOption, VerdictOptions
from caster_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_commons.sandbox.manager import SandboxDeployment, SandboxManager
from caster_validator.application.dto.evaluation import (
    EvaluationOutcome,
    EvaluationRequest,
    MinerTaskResult,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from caster_validator.application.invoke_entrypoint import SandboxInvocationError
from caster_validator.application.ports.subtensor import ValidatorNodeInfo
from caster_validator.application.providers.claims import StaticClaimsProvider
from caster_validator.application.scheduler import EvaluationScheduler, SchedulerConfig
from caster_validator.application.services.evaluation_runner import EvaluationRunner
from caster_validator.application.services.evaluation_scoring import EvaluationScore
from caster_validator.domain.evaluation import MinerAnswer, MinerCriterionEvaluation
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")

BINARY_VERDICT_OPTIONS = VerdictOptions(
    options=(
        VerdictOption(value=-1, description="Fail"),
        VerdictOption(value=1, description="Pass"),
    )
)


class DummySandboxManager(SandboxManager):
    def __init__(self) -> None:
        self.starts: list[object | None] = []
        self.stops: list[SandboxDeployment] = []

    def start(self, options: object | None = None) -> SandboxDeployment:
        self.starts.append(options)
        return SandboxDeployment(client=object())

    def stop(self, deployment: SandboxDeployment) -> None:
        self.stops.append(deployment)


class DummyEvaluationRecordStore:
    def __init__(self) -> None:
        self.miner_task_results: list[MinerTaskResult] = []

    def record(self, result: MinerTaskResult) -> None:
        self.miner_task_results.append(result)


def _sample_claim(text: str) -> MinerTaskClaim:
    return MinerTaskClaim(
        claim_id=uuid4(),
        text=text,
        rubric=Rubric(
            title="Accuracy",
            description="Check facts.",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
    )


async def test_scheduler_runs_all_claims_for_each_uid() -> None:
    claims = (_sample_claim("one"), _sample_claim("two"))
    claims_provider = StaticClaimsProvider(claims)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())

    recorded_requests: list[EvaluationRequest] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request: EvaluationRequest):
                recorded_requests.append(request)
                evaluation = MinerCriterionEvaluation(
                    criterion_evaluation_id=request.criterion_evaluation_id,
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    claim_id=request.claim.claim_id,
                    rubric=request.claim.rubric,
                    miner_answer=MinerAnswer(verdict=1, justification="ok", citations=()),
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return EvaluationOutcome(
                    criterion_evaluation=evaluation,
                    score=EvaluationScore(
                        verdict_score=0.5,
                        support_score=0.5,
                        justification_pass=True,
                        failed_citation_ids=(),
                        grader_rationale="ok",
                    ),
                    tool_receipts=(),
                    usage=TokenUsageSummary.from_totals(
                        {
                            "search": {
                                "web": LlmUsageTotals(
                                    prompt_tokens=10,
                                    completion_tokens=12,
                                    total_tokens=22,
                                    call_count=1,
                                ),
                            },
                        },
                    ),
                )

        return StubOrchestrator()

    config = SchedulerConfig(
        entrypoint="evaluate_criterion",
        token_secret_bytes=8,
        session_ttl=timedelta(minutes=5),
    )

    scheduler = EvaluationScheduler(
        claims_provider=claims_provider,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda candidate: {"uid": candidate.uid, "artifact_id": candidate.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=config,
    )

    batch_id = uuid4()
    candidates = (
        ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0),
        ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0),
    )
    result = await scheduler.run(batch_id=batch_id, requested_candidates=candidates)

    assert len(sandbox_manager.starts) == 2
    assert len(sandbox_manager.stops) == 2
    assert len(recorded_requests) == len(claims) * 2
    assert len(result.evaluations) == len(recorded_requests)
    assert result.claims == claims
    assert result.candidate_uids == (3, 5)
    assert len(evaluation_records.miner_task_results) == len(result.evaluations)
    assert all(record.batch_id == batch_id for record in evaluation_records.miner_task_results)


async def test_scheduler_accepts_candidate_filter() -> None:
    claims_provider = StaticClaimsProvider((_sample_claim("only"),))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())

    def orchestrator_factory(_client: object):
        class Stub:
            async def evaluate(self, request: EvaluationRequest):
                evaluation = MinerCriterionEvaluation(
                    criterion_evaluation_id=request.criterion_evaluation_id,
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    claim_id=request.claim.claim_id,
                    rubric=request.claim.rubric,
                    miner_answer=MinerAnswer(verdict=1, justification="ok", citations=()),
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return EvaluationOutcome(
                    criterion_evaluation=evaluation,
                    score=EvaluationScore(
                        verdict_score=0.5,
                        support_score=0.5,
                        justification_pass=True,
                        failed_citation_ids=(),
                        grader_rationale="ok",
                    ),
                    tool_receipts=(),
                    usage=TokenUsageSummary.from_totals(
                        {
                            "search": {
                                "web": LlmUsageTotals(
                                    prompt_tokens=8,
                                    completion_tokens=9,
                                    total_tokens=17,
                                    call_count=1,
                                ),
                            },
                        },
                    ),
                )

        return Stub()

    config = SchedulerConfig(
        entrypoint="evaluate_criterion",
        token_secret_bytes=8,
        session_ttl=timedelta(minutes=5),
    )

    scheduler = EvaluationScheduler(
        claims_provider=claims_provider,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda candidate: {"uid": candidate.uid, "artifact_id": candidate.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=config,
    )

    requested_candidate = ScriptArtifactSpec(uid=11, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    result = await scheduler.run(batch_id=uuid4(), requested_candidates=(requested_candidate,))

    assert result.candidate_uids == (11,)
    assert len(sandbox_manager.starts) == 1
    assert len(evaluation_records.miner_task_results) == len(result.evaluations)


async def test_scheduler_skips_miner_when_sandbox_invocation_errors() -> None:
    claim = _sample_claim("unstable")
    claims_provider = StaticClaimsProvider((claim,))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())

    def orchestrator_factory(_client: object):
        class FailingOrchestrator:
            async def evaluate(self, request: EvaluationRequest):
                raise SandboxInvocationError("upstream tool failure")

        return FailingOrchestrator()

    config = SchedulerConfig(
        entrypoint="evaluate_criterion",
        token_secret_bytes=8,
        session_ttl=timedelta(minutes=5),
    )

    scheduler = EvaluationScheduler(
        claims_provider=claims_provider,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda candidate: {"uid": candidate.uid, "artifact_id": candidate.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=config,
    )

    candidates = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_candidates=candidates)

    assert result.evaluations == ()
    assert len(evaluation_records.miner_task_results) == 1
    assert len(sandbox_manager.starts) == 1
    assert len(sandbox_manager.stops) == 1


async def test_evaluation_runner_issues_session_with_claim_budget() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    evaluation_records = DummyEvaluationRecordStore()
    config = SchedulerConfig(
        entrypoint="evaluate_criterion",
        token_secret_bytes=8,
        session_ttl=timedelta(minutes=5),
    )
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        config=config,
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        progress=None,
    )

    claim = MinerTaskClaim(
        claim_id=uuid4(),
        text="budgeted",
        rubric=Rubric(
            title="Accuracy",
            description="Check facts.",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
        budget_usd=0.123,
    )

    issued = runner._issue_session(uid=3, claim=claim)

    assert issued.session.claim_id == claim.claim_id
    assert issued.session.budget_usd == pytest.approx(0.123)


async def test_evaluation_runner_request_includes_feed_search_context_when_present() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    evaluation_records = DummyEvaluationRecordStore()
    config = SchedulerConfig(
        entrypoint="evaluate_criterion",
        token_secret_bytes=8,
        session_ttl=timedelta(minutes=5),
    )
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        config=config,
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        progress=None,
    )

    context = FeedSearchContext(feed_id=uuid4(), enqueue_seq=6)
    claim = MinerTaskClaim(
        claim_id=uuid4(),
        text="context claim",
        rubric=Rubric(
            title="Accuracy",
            description="Check facts.",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
        context=context,
    )

    request = runner._build_request(
        session_id=uuid4(),
        token=uuid4().hex,
        uid=3,
        artifact_id=uuid4(),
        claim=claim,
    )

    assert request.payload["context"] == {
        "feed_id": str(context.feed_id),
        "enqueue_seq": context.enqueue_seq,
    }
