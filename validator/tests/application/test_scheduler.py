from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.claim import EvaluationClaim, ReferenceAnswer, Rubric
from caster_commons.domain.session import LlmUsageTotals
from caster_commons.domain.verdict import BINARY_VERDICT_OPTIONS
from caster_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_commons.sandbox.manager import SandboxDeployment, SandboxManager
from caster_validator.application.dto.evaluation import (
    EvaluationCloseout,
    EvaluationOutcome,
    EvaluationRequest,
    TokenUsageSummary,
)
from caster_validator.application.invoke_entrypoint import SandboxInvocationError
from caster_validator.application.ports.subtensor import ValidatorNodeInfo
from caster_validator.application.providers.claims import StaticClaimsProvider
from caster_validator.application.scheduler import EvaluationScheduler, SchedulerConfig
from caster_validator.application.services.evaluation_scoring import EvaluationScore
from caster_validator.domain.evaluation import MinerAnswer, MinerEvaluation
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")


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
        self.closeouts: list[EvaluationCloseout] = []

    def record(self, closeout: EvaluationCloseout) -> None:
        self.closeouts.append(closeout)


def _sample_claim(text: str) -> EvaluationClaim:
    return EvaluationClaim(
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
    subtensor.metagraph = subtensor.metagraph.__class__(uids=(3, 5), hotkeys=("a", "b"))
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())

    recorded_requests: list[EvaluationRequest] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request: EvaluationRequest):
                recorded_requests.append(request)
                evaluation = MinerEvaluation(
                    evaluation_id=request.evaluation_id,
                    session_id=request.session_id,
                    uid=request.uid,
                    claim_id=request.claim.claim_id,
                    rubric=request.claim.rubric,
                    miner_answer=MinerAnswer(verdict=1, justification="ok", citations=()),
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return EvaluationOutcome(
                    evaluation=evaluation,
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
        budget_usd=0.1,
    )

    scheduler = EvaluationScheduler(
        claims_provider=claims_provider,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda uid: {"uid": uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=config,
    )

    run_id = uuid4()
    result = await scheduler.run(run_id=run_id)

    assert len(sandbox_manager.starts) == 2
    assert len(sandbox_manager.stops) == 2
    assert len(recorded_requests) == len(claims) * 2
    assert len(result.evaluations) == len(recorded_requests)
    assert result.claims == claims
    assert result.uids == (3, 5)
    assert len(evaluation_records.closeouts) == len(result.evaluations)
    assert all(closeout.run_id == run_id for closeout in evaluation_records.closeouts)


async def test_scheduler_accepts_requested_uid_filter() -> None:
    claims_provider = StaticClaimsProvider((_sample_claim("only"),))
    subtensor = FakeSubtensorClient()
    subtensor.metagraph = subtensor.metagraph.__class__(uids=(9, 11), hotkeys=("x", "y"))
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())

    def orchestrator_factory(_client: object):
        class Stub:
            async def evaluate(self, request: EvaluationRequest):
                evaluation = MinerEvaluation(
                    evaluation_id=request.evaluation_id,
                    session_id=request.session_id,
                    uid=request.uid,
                    claim_id=request.claim.claim_id,
                    rubric=request.claim.rubric,
                    miner_answer=MinerAnswer(verdict=1, justification="ok", citations=()),
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return EvaluationOutcome(
                    evaluation=evaluation,
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
        budget_usd=0.1,
    )

    scheduler = EvaluationScheduler(
        claims_provider=claims_provider,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda uid: {"uid": uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=config,
    )

    requested_uid = 11
    result = await scheduler.run(run_id=uuid4(), requested_uids=(requested_uid,))

    assert result.uids == (requested_uid,)
    assert len(sandbox_manager.starts) == 1
    assert len(evaluation_records.closeouts) == len(result.evaluations)


async def test_scheduler_skips_miner_when_sandbox_invocation_errors() -> None:
    claim = _sample_claim("unstable")
    claims_provider = StaticClaimsProvider((claim,))
    subtensor = FakeSubtensorClient()
    subtensor.metagraph = subtensor.metagraph.__class__(uids=(7,), hotkeys=("miner",))
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
        budget_usd=0.1,
    )

    scheduler = EvaluationScheduler(
        claims_provider=claims_provider,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda uid: {"uid": uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=config,
    )

    result = await scheduler.run(run_id=uuid4())

    assert result.evaluations == ()
    assert len(evaluation_records.closeouts) == 0
    assert len(sandbox_manager.starts) == 1
    assert len(sandbox_manager.stops) == 1
