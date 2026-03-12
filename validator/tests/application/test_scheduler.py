from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from caster_commons.domain.tool_usage import ToolUsageSummary
from caster_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_commons.sandbox.manager import SandboxDeployment, SandboxManager
from caster_validator.application.dto.evaluation import (
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
)
from caster_validator.application.invoke_entrypoint import SandboxInvocationError
from caster_validator.application.ports.subtensor import ValidatorNodeInfo
from caster_validator.application.scheduler import EvaluationScheduler, SchedulerConfig
from caster_validator.domain.evaluation import MinerTaskRun
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
        self.records_by_batch: list[MinerTaskRunSubmission] = []

    def record(self, result: MinerTaskRunSubmission) -> None:
        self.records_by_batch.append(result)


class DummyReceiptLog(ReceiptLogPort):
    def __init__(self) -> None:
        self._records: dict[str, object] = {}

    def record(self, receipt: object) -> None:
        self._records[str(len(self._records))] = receipt

    def lookup(self, receipt_id: str) -> object | None:
        return self._records.get(receipt_id)

    def values(self):
        return tuple(self._records.values())

    def for_session(self, session_id):
        return ()

    def clear_session(self, session_id) -> None:
        return None


class DummyProgressRecorder:
    def __init__(self, recorded: frozenset[tuple[UUID, UUID]] = frozenset()) -> None:
        self._recorded = set(recorded)

    def register(self, _batch) -> None:
        return None

    def record(self, result: MinerTaskRunSubmission) -> None:
        self._recorded.add((result.run.artifact_id, result.run.task_id))

    def recorded_pairs(self, _batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        return frozenset(self._recorded)


def _task(text: str, *, budget_usd: float = 0.05) -> MinerTask:
    return MinerTask(
        task_id=uuid4(),
        query=Query(text=text),
        reference_answer=ReferenceAnswer(text=f"reference {text}"),
        budget_usd=budget_usd,
    )


async def test_scheduler_runs_all_tasks_for_each_artifact() -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    recorded_requests: list[tuple[int, MinerTask]] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.task))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (
        ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0),
        ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0),
    )
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(sandbox_manager.starts) == 2
    assert len(sandbox_manager.stops) == 2
    assert len(recorded_requests) == len(tasks) * 2
    assert len(result.runs) == len(recorded_requests)
    assert result.tasks == tasks
    assert len(evaluation_records.records_by_batch) == len(result.runs)


async def test_scheduler_records_failure_when_sandbox_invocation_errors() -> None:
    task = _task("unstable")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    clock_values = iter(
        (
            datetime(2025, 10, 27, 0, 0, 0, tzinfo=UTC),
            datetime(2025, 10, 27, 0, 0, 2, tzinfo=UTC),
        ),
    )

    def orchestrator_factory(_client: object):
        class FailingOrchestrator:
            async def evaluate(self, request):
                raise SandboxInvocationError("upstream tool failure")

        return FailingOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: next(clock_values),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(result.runs) == 1
    assert result.runs[0].score == 0.0
    assert result.runs[0].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="upstream tool failure",
    )
    assert result.runs[0].run.details.elapsed_ms == pytest.approx(2000.0)


async def test_scheduler_records_failure_when_scoring_errors() -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    class FailingThenSuccessfulOrchestrator:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("embedding client unavailable")
            details = EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=1.0,
                    similarity_score=0.5,
                    total_score=0.75,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            )
            run = MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text=f"answer {request.task.query.text}"),
                details=details,
                completed_at=datetime(2025, 10, 27, tzinfo=UTC),
            )
            return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

    orchestrator = FailingThenSuccessfulOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(result.runs) == 2
    assert len(evaluation_records.records_by_batch) == 2
    assert result.runs[0].score == 0.0
    assert result.runs[0].run.details.error == EvaluationError(
        code="task_run_failed",
        message="embedding client unavailable",
    )
    assert result.runs[1].score == pytest.approx(0.75)
    assert result.runs[1].run.response == Response(text="answer second")


async def test_evaluation_runner_issues_session_with_task_budget() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    evaluation_records = DummyEvaluationRecordStore()
    receipt_log = DummyReceiptLog()
    scheduler = EvaluationScheduler(
        tasks=(),
        subtensor_client=subtensor,
        sandbox_manager=DummySandboxManager(),
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda artifact: artifact,
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    task = _task("budgeted", budget_usd=0.123)
    issued = scheduler._runner._issue_session(uid=3, task=task)

    assert issued.session.task_id == task.task_id
    assert issued.session.budget_usd == pytest.approx(0.123)


async def test_scheduler_runs_only_remaining_pairs() -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    progress = DummyProgressRecorder(recorded=frozenset({(artifact.artifact_id, tasks[0].task_id)}))
    recorded_requests: list[tuple[int, MinerTask]] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.task))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda current_artifact: {"uid": current_artifact.uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=progress,
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=(artifact,))

    assert len(sandbox_manager.starts) == 1
    assert [(uid, task.task_id) for uid, task in recorded_requests] == [(artifact.uid, tasks[1].task_id)]
    assert len(result.runs) == 1
    assert result.runs[0].run.task_id == tasks[1].task_id


async def test_scheduler_skips_artifact_when_all_pairs_are_already_recorded() -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    first_artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    second_artifact = ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    progress = DummyProgressRecorder(
        recorded=frozenset((first_artifact.artifact_id, task.task_id) for task in tasks)
    )
    recorded_requests: list[tuple[int, UUID]] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.artifact_id))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda current_artifact: {"uid": current_artifact.uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=progress,
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=(first_artifact, second_artifact))

    assert len(sandbox_manager.starts) == 1
    assert all(uid == second_artifact.uid for uid, _artifact_id in recorded_requests)
    assert len(result.runs) == len(tasks)
