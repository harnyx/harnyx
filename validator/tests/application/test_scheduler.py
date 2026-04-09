from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event
from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest

import harnyx_validator.application.scheduler as scheduler_module
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolExecutionFacts
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
)
from harnyx_validator.application.invoke_entrypoint import SandboxInvocationError
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo
from harnyx_validator.application.scheduler import EvaluationScheduler, SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import (
    ArtifactEvaluationOutcome,
    ArtifactFailure,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError
from validator.tests.fixtures.fakes import FakeReceiptLog
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def blocking_executor() -> ThreadPoolExecutor:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-validator-batch-blocking")
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


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

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        return None

    def record_provider_call(self, *, session_id: UUID, provider: str, model: str) -> None:
        return None

    def record_provider_failure(self, *, session_id: UUID, provider: str, model: str) -> None:
        return None

    def consume_provider_failures(self, session_id: UUID) -> tuple[dict[str, object], ...]:
        return ()

    def clear_task_session(self, session_id: UUID) -> None:
        return None


def _task(text: str, *, budget_usd: float = 0.05) -> MinerTask:
    return MinerTask(
        task_id=uuid4(),
        query=Query(text=text),
        reference_answer=ReferenceAnswer(text=f"reference {text}"),
        budget_usd=budget_usd,
    )


def _sandbox_invocation_error(
    message: str,
    *,
    status_code: int = 0,
    detail_exception: str = "RuntimeError",
    detail_error: str | None = None,
) -> SandboxInvocationError:
    return SandboxInvocationError(
        message,
        status_code=status_code,
        detail_code=None,
        detail_exception=detail_exception,
        detail_error=detail_error or message,
    )


def _llm_receipt(*, session_id: UUID, uid: int, total_tokens: int, elapsed_ms: float) -> ToolCall:
    return ToolCall(
        receipt_id=uuid4().hex,
        session_id=session_id,
        uid=uid,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 27, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="req",
            response_hash="res",
            response_payload={"usage": {"total_tokens": total_tokens}},
            execution=ToolExecutionFacts(elapsed_ms=elapsed_ms),
        ),
    )


def _submission_for_task(
    *,
    batch_id: UUID,
    validator_uid: int,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    error: EvaluationError | None = None,
) -> MinerTaskRunSubmission:
    issued_at = datetime(2025, 10, 27, tzinfo=UTC)
    session_id = uuid4()
    session = Session(
        session_id=session_id,
        uid=artifact.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=task.budget_usd,
        usage=SessionUsage(),
        status=SessionStatus.ERROR if error is not None else SessionStatus.COMPLETED,
    )
    if error is None:
        run = MinerTaskRun(
            session_id=session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=Response(text=f"answer {task.query.text}"),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=1.0,
                    similarity_score=1.0,
                    total_score=1.0,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            ),
            completed_at=issued_at,
        )
        return MinerTaskRunSubmission(
            batch_id=batch_id,
            validator_uid=validator_uid,
            run=run,
            score=1.0,
            usage=TokenUsageSummary.empty(),
            session=session,
        )

    run = MinerTaskRun(
        session_id=session_id,
        uid=artifact.uid,
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        details=EvaluationDetails(
            error=error,
            total_tool_usage=ToolUsageSummary.zero(),
        ),
        completed_at=issued_at,
    )
    return MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=validator_uid,
        run=run,
        score=0.0,
        usage=TokenUsageSummary.empty(),
        session=session,
    )


async def test_scheduler_runs_all_tasks_for_each_artifact(
    blocking_executor: ThreadPoolExecutor,
) -> None:
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
        blocking_executor=blocking_executor,
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


async def test_scheduler_avoids_asyncio_to_thread_for_blocking_work(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
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
                    response=Response(text="answer one"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    async def _unexpected_to_thread(*args, **kwargs):
        raise AssertionError("scheduler should not use asyncio.to_thread")

    monkeypatch.setattr(scheduler_module.asyncio, "to_thread", _unexpected_to_thread)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(result.runs) == 1
    assert len(sandbox_manager.starts) == 1
    assert len(sandbox_manager.stops) == 1


async def test_scheduler_cancellation_does_not_wait_for_blocking_lane_shutdown(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("shutdown")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    stop_started = Event()
    release_stop = Event()
    stop_finished = Event()

    class BlockingStopSandboxManager(DummySandboxManager):
        def stop(self, deployment: SandboxDeployment) -> None:
            self.stops.append(deployment)
            stop_started.set()
            try:
                release_stop.wait(timeout=5.0)
            finally:
                stop_finished.set()

    sandbox_manager = BlockingStopSandboxManager()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
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
                    response=Response(text="answer shutdown"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    run_task = asyncio.create_task(
        scheduler.run(
            batch_id=uuid4(),
            requested_artifacts=(ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),),
        )
    )

    try:
        assert await asyncio.to_thread(stop_started.wait, 1.0) is True
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=0.2)
        assert release_stop.is_set() is False
    finally:
        release_stop.set()
        assert await asyncio.to_thread(stop_finished.wait, 1.0) is True


async def test_scheduler_records_zero_score_when_sandbox_invocation_errors(
    blocking_executor: ThreadPoolExecutor,
) -> None:
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
                raise _sandbox_invocation_error("upstream tool failure")

        return FailingOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
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
    assert evaluation_records.records_by_batch == list(result.runs)


async def test_scheduler_retries_only_transient_sandbox_invocation_errors(
    blocking_executor: ThreadPoolExecutor,
) -> None:
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
                raise _sandbox_invocation_error("upstream tool failure")
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
        blocking_executor=blocking_executor,
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
    assert orchestrator.calls == 3
    assert result.runs[0].score == pytest.approx(0.75)
    assert result.runs[0].run.response == Response(text="answer first")
    assert result.runs[1].score == pytest.approx(0.75)
    assert result.runs[1].run.response == Response(text="answer second")


async def test_scheduler_fails_batch_for_generic_post_invoke_error(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    class GenericFailureThenSuccessOrchestrator:
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

    orchestrator = GenericFailureThenSuccessOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    with pytest.raises(ValidatorBatchFailedError, match="embedding client unavailable") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert exc_info.value.error_code == "unexpected_validator_failure"
    assert orchestrator.calls == 1
    assert evaluation_records.records_by_batch == []


async def test_scheduler_records_retry_exhausted_internal_timeout_as_task_failure(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = FakeReceiptLog()

    class AlwaysTimeoutOrchestrator:
        def __init__(self) -> None:
            self.calls = 0
            self.session_ids: list[UUID] = []

        async def evaluate(self, request):
            self.calls += 1
            self.session_ids.append(request.session_id)
            raise httpx.ReadTimeout(
                "embedding timed out",
                request=httpx.Request("POST", "https://validator.invalid/scoring"),
            )

    orchestrator = AlwaysTimeoutOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    with pytest.raises(ValidatorBatchFailedError, match="embedding timed out") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert exc_info.value.error_code == "validator_internal_timeout"
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 1
    assert evaluation_records.records_by_batch == []


async def test_scheduler_uses_successful_baseline_across_execution_for_timeout_inconclusive(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("baseline"), _task("timeout"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = FakeReceiptLog()

    class BaselineThenTimeoutOrchestrator:
        def __init__(self, receipt_log: FakeReceiptLog) -> None:
            self._receipt_log = receipt_log
            self.timeout_calls = 0

        async def evaluate(self, request):
            if request.task.query.text == "baseline":
                receipt = _llm_receipt(
                    session_id=request.session_id,
                    uid=request.uid,
                    total_tokens=100,
                    elapsed_ms=1000.0,
                )
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
                    response=Response(text="baseline answer"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(
                    run=run,
                    tool_receipts=(receipt,),
                    usage=TokenUsageSummary.empty(),
                )

            self.timeout_calls += 1
            self._receipt_log.record(
                _llm_receipt(
                    session_id=request.session_id,
                    uid=request.uid,
                    total_tokens=100,
                    elapsed_ms=2500.0,
                )
            )
            raise _sandbox_invocation_error(
                "sandbox entrypoint request timed out",
                status_code=504,
                detail_exception="TimeoutException",
                detail_error="sandbox entrypoint request timed out",
            )

    orchestrator = BaselineThenTimeoutOrchestrator(receipt_log)
    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )
    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)

    with pytest.raises(ValidatorBatchFailedError, match="terminal timeout") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert exc_info.value.error_code == "timeout_inconclusive"
    assert orchestrator.timeout_calls == 3
    assert evaluation_records.records_by_batch[0].score == pytest.approx(0.75)
    assert evaluation_records.records_by_batch[-1].run.details.error == EvaluationError(
        code="timeout_inconclusive",
        message="terminal timeout",
    )


async def test_retry_round_preserves_earlier_completed_runs_when_later_round_aborts(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("later failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
    )
    batch_id = uuid4()
    earlier_task = _task("earlier success")
    later_task = _task("later unresolved")
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    first_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=earlier_task,
    )
    calls = 0

    class _RetryWaveRunner:
        async def evaluate_artifact_with_state(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ArtifactEvaluationOutcome(
                    submissions=(first_submission,),
                    unresolved_tasks=(later_task,),
                    timeout_observations_by_pair={},
                    slowest_successful_tps=None,
                )
            return ArtifactEvaluationOutcome(
                submissions=(first_submission,),
                unresolved_tasks=(later_task,),
                timeout_observations_by_pair={},
                slowest_successful_tps=None,
                artifact_failure=ArtifactFailure(
                    error_code="sandbox_invocation_failed",
                    message="later round failed",
                    failure_detail=ValidatorBatchFailureDetail(
                        error_code="sandbox_invocation_failed",
                        error_message="later round failed",
                        occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                        artifact_id=artifact.artifact_id,
                        uid=artifact.uid,
                        exception_type="SandboxInvocationError",
                    ),
                    artifact_breaker_tripped=True,
                ),
            )

    scheduler._runner = _RetryWaveRunner()  # type: ignore[assignment]

    result = await scheduler._evaluate_artifact_with_timeout_state(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(earlier_task, later_task),
        orchestrator=cast(scheduler_module.TaskRunOrchestrator, object()),
        successful_baseline_tps=None,
        timeout_retry_state_by_pair={},
    )

    assert result.artifact_failure is not None
    assert result.submissions == (first_submission,)
    assert result.unresolved_tasks == (later_task,)


async def test_retry_round_passes_earlier_runs_back_to_runner_for_breaker_start(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("later timeout")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
    )
    batch_id = uuid4()
    earlier_task = _task("earlier breaker failure")
    later_task = _task("later unresolved")
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    earlier_failure = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=earlier_task,
        error=EvaluationError(code="sandbox_invocation_failed", message="earlier round failed"),
    )
    seen_earlier_submissions: list[tuple[MinerTaskRunSubmission, ...]] = []
    calls = 0

    class _RetryWaveRunner:
        async def evaluate_artifact_with_state(self, **kwargs):
            nonlocal calls
            calls += 1
            seen_earlier_submissions.append(kwargs["earlier_submissions"])
            if calls == 1:
                return ArtifactEvaluationOutcome(
                    submissions=(earlier_failure,),
                    unresolved_tasks=(later_task,),
                    timeout_observations_by_pair={},
                    slowest_successful_tps=None,
                )
            return ArtifactEvaluationOutcome(
                submissions=(earlier_failure,),
                unresolved_tasks=(),
                timeout_observations_by_pair={},
                slowest_successful_tps=None,
            )

    scheduler._runner = _RetryWaveRunner()  # type: ignore[assignment]

    result = await scheduler._evaluate_artifact_with_timeout_state(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(earlier_task, later_task),
        orchestrator=cast(scheduler_module.TaskRunOrchestrator, object()),
        successful_baseline_tps=None,
        timeout_retry_state_by_pair={},
    )

    assert seen_earlier_submissions[0] == ()
    assert seen_earlier_submissions[1] == (earlier_failure,)
    assert result.submissions == (earlier_failure,)


async def test_scheduler_keeps_successful_baseline_when_artifact_returns_failure_outcome(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    successful_task = _task("baseline carry success")
    failed_task = _task("baseline carry failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    scheduler = EvaluationScheduler(
        tasks=(successful_task, failed_task),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
    )
    batch_id = uuid4()
    first_artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    second_artifact = ScriptArtifactSpec(uid=8, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    successful_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=first_artifact,
        task=successful_task,
    )
    failed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=first_artifact,
        task=failed_task,
        error=EvaluationError(code="sandbox_invocation_failed", message="shared sandbox failure"),
    )
    later_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=second_artifact,
        task=successful_task,
    )

    class _FailureOutcomeRunner:
        def __init__(self) -> None:
            self.seen_baselines: list[float | None] = []
            self.record_failure_calls: list[tuple[UUID, tuple[MinerTask, ...]]] = []

        async def evaluate_artifact_with_state(
            self,
            *,
            artifact: ScriptArtifactSpec,
            successful_baseline_tps: float | None,
            **_kwargs,
        ) -> ArtifactEvaluationOutcome:
            self.seen_baselines.append(successful_baseline_tps)
            if artifact.artifact_id == first_artifact.artifact_id:
                return ArtifactEvaluationOutcome(
                    submissions=(successful_submission,),
                    unresolved_tasks=(failed_task,),
                    timeout_observations_by_pair={},
                    slowest_successful_tps=40.0,
                    artifact_failure=ArtifactFailure(
                        error_code="sandbox_invocation_failed",
                        message="shared sandbox failure",
                        failure_detail=ValidatorBatchFailureDetail(
                            error_code="sandbox_invocation_failed",
                            error_message="shared sandbox failure",
                            occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                            artifact_id=artifact.artifact_id,
                            uid=artifact.uid,
                            exception_type="SandboxInvocationError",
                        ),
                        artifact_breaker_tripped=True,
                    ),
                )
            return ArtifactEvaluationOutcome(
                submissions=(later_submission,),
                unresolved_tasks=(),
                timeout_observations_by_pair={},
                slowest_successful_tps=40.0,
            )

        async def record_failure_for_artifact(
            self,
            *,
            artifact: ScriptArtifactSpec,
            tasks,
            error_code: str,
            error_message: str,
            **_kwargs,
        ) -> list[MinerTaskRunSubmission]:
            self.record_failure_calls.append((artifact.artifact_id, tuple(tasks)))
            assert artifact.artifact_id == first_artifact.artifact_id
            assert error_code == "sandbox_invocation_failed"
            assert error_message == "shared sandbox failure"
            assert tuple(tasks) == (failed_task,)
            return [failed_submission]

    runner = _FailureOutcomeRunner()
    scheduler._runner = runner  # type: ignore[assignment]

    result = await scheduler.run(
        batch_id=batch_id,
        requested_artifacts=(first_artifact, second_artifact),
    )

    assert runner.seen_baselines == [None, 40.0]
    assert runner.record_failure_calls == [(first_artifact.artifact_id, (failed_task,))]
    assert result.runs == (successful_submission, failed_submission, later_submission)


async def test_scheduler_artifact_failure_outcome_keeps_single_owner_for_completed_runs(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    earlier_task = _task("single owner earlier")
    later_task = _task("single owner later")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    scheduler = EvaluationScheduler(
        tasks=(earlier_task, later_task),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
    )
    batch_id = uuid4()
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    earlier_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=earlier_task,
    )
    seen_earlier_submissions: list[tuple[MinerTaskRunSubmission, ...]] = []
    calls = 0

    class _SingleOwnerRunner:
        async def evaluate_artifact_with_state(self, **kwargs) -> ArtifactEvaluationOutcome:
            nonlocal calls
            calls += 1
            seen_earlier_submissions.append(kwargs["earlier_submissions"])
            if calls == 1:
                return ArtifactEvaluationOutcome(
                    submissions=(earlier_submission,),
                    unresolved_tasks=(later_task,),
                    timeout_observations_by_pair={},
                    slowest_successful_tps=40.0,
                )
            return ArtifactEvaluationOutcome(
                submissions=(earlier_submission,),
                unresolved_tasks=(later_task,),
                timeout_observations_by_pair={},
                slowest_successful_tps=40.0,
                artifact_failure=ArtifactFailure(
                    error_code="sandbox_invocation_failed",
                    message="shared sandbox failure",
                    failure_detail=ValidatorBatchFailureDetail(
                        error_code="sandbox_invocation_failed",
                        error_message="shared sandbox failure",
                        occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                        artifact_id=artifact.artifact_id,
                        uid=artifact.uid,
                        exception_type="SandboxInvocationError",
                    ),
                    artifact_breaker_tripped=True,
                ),
            )

    scheduler._runner = _SingleOwnerRunner()  # type: ignore[assignment]

    result = await scheduler._evaluate_artifact_with_timeout_state(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(earlier_task, later_task),
        orchestrator=cast(scheduler_module.TaskRunOrchestrator, object()),
        successful_baseline_tps=None,
        timeout_retry_state_by_pair={},
    )

    assert seen_earlier_submissions == [(), (earlier_submission,)]
    assert result.submissions == (earlier_submission,)
    assert result.artifact_failure is not None


async def test_scheduler_retries_sandbox_start_once_before_running_tasks(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("startup")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class FlakySandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            if len(self.starts) == 1:
                raise RuntimeError("sandbox cold start failed")
            return SandboxDeployment(client=object())

    sandbox_manager = FlakySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
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
                    response=Response(text="answer startup"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(sandbox_manager.starts) == 2
    assert len(result.runs) == 1
    assert result.runs[0].score == pytest.approx(0.75)


async def test_scheduler_records_zero_score_for_terminal_setup_failures(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("startup failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class AlwaysFailingSandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            raise RuntimeError("sandbox cold start failed")

    sandbox_manager = AlwaysFailingSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(sandbox_manager.starts) == 2
    assert len(result.runs) == 1
    assert result.runs[0].run.details.error == EvaluationError(
        code="sandbox_start_failed",
        message="sandbox cold start failed",
    )
    assert evaluation_records.records_by_batch == list(result.runs)


async def test_scheduler_fails_batch_after_three_sandbox_start_breakers(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("startup failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class AlwaysFailingSandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            raise RuntimeError("sandbox cold start failed")

    sandbox_manager = AlwaysFailingSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0)
        for uid in (3, 5, 7)
    )
    with pytest.raises(
        ValidatorBatchFailedError,
        match="validator artifact breaker tripped across 3 artifacts",
    ) as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert exc_info.value.error_code == "artifact_breaker_tripped"
    assert exc_info.value.failure_detail.error_code == "artifact_breaker_tripped"
    assert len(sandbox_manager.starts) == 6
    assert [submission.run.details.error for submission in evaluation_records.records_by_batch] == [
        EvaluationError(code="sandbox_start_failed", message="sandbox cold start failed"),
        EvaluationError(code="sandbox_start_failed", message="sandbox cold start failed"),
        EvaluationError(code="sandbox_start_failed", message="sandbox cold start failed"),
    ]


async def test_scheduler_fails_batch_after_three_artifact_fetch_breakers(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("fetch failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda _artifact: (_ for _ in ()).throw(
            ArtifactPreparationError(
                error_code="artifact_fetch_failed",
                message="platform artifact fetch exhausted retries",
                exception_type="PlatformClientError",
            )
        ),
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0)
        for uid in (3, 5, 7)
    )

    with pytest.raises(
        ValidatorBatchFailedError,
        match="validator artifact breaker tripped across 3 artifacts",
    ) as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert exc_info.value.error_code == "artifact_breaker_tripped"
    assert len(sandbox_manager.starts) == 0
    assert [submission.run.details.error for submission in evaluation_records.records_by_batch] == [
        EvaluationError(code="artifact_fetch_failed", message="platform artifact fetch exhausted retries"),
        EvaluationError(code="artifact_fetch_failed", message="platform artifact fetch exhausted retries"),
        EvaluationError(code="artifact_fetch_failed", message="platform artifact fetch exhausted retries"),
    ]


async def test_scheduler_keeps_three_hash_mismatch_failures_artifact_scoped(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("hash mismatch")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda _artifact: (_ for _ in ()).throw(
            ArtifactPreparationError(
                error_code="artifact_hash_mismatch",
                message="platform agent sha256 mismatch",
            )
        ),
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0)
        for uid in (3, 5, 7)
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(result.runs) == 3
    assert len(sandbox_manager.starts) == 0
    assert [submission.run.details.error for submission in result.runs] == [
        EvaluationError(code="artifact_hash_mismatch", message="platform agent sha256 mismatch"),
        EvaluationError(code="artifact_hash_mismatch", message="platform agent sha256 mismatch"),
        EvaluationError(code="artifact_hash_mismatch", message="platform agent sha256 mismatch"),
    ]


async def test_scheduler_fails_batch_after_three_artifact_breakers(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("artifact breaker")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    batch_id = uuid4()
    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0)
        for uid in (3, 5, 7)
    )

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    class ArtifactFailingRunner:
        def __init__(self) -> None:
            self.failed_artifact_ids: list[UUID] = []
            self.recorded_artifact_ids: list[UUID] = []

        async def evaluate_artifact_with_state(self, *, artifact: ScriptArtifactSpec, tasks, **_kwargs):
            self.failed_artifact_ids.append(artifact.artifact_id)
            return ArtifactEvaluationOutcome(
                submissions=(),
                unresolved_tasks=tuple(tasks),
                timeout_observations_by_pair={},
                slowest_successful_tps=None,
                artifact_failure=ArtifactFailure(
                    error_code="sandbox_invocation_failed",
                    message="shared sandbox failure",
                    failure_detail=ValidatorBatchFailureDetail(
                        error_code="sandbox_invocation_failed",
                        error_message="shared sandbox failure",
                        occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                        artifact_id=artifact.artifact_id,
                        uid=artifact.uid,
                        exception_type="SandboxInvocationError",
                    ),
                    artifact_breaker_tripped=True,
                ),
            )

        async def record_failure_for_artifact(
            self,
            *,
            artifact: ScriptArtifactSpec,
            tasks,
            error_code: str,
            error_message: str,
            **_kwargs,
        ) -> list[MinerTaskRunSubmission]:
            self.recorded_artifact_ids.append(artifact.artifact_id)
            assert tuple(tasks) == (task,)
            assert error_code == "sandbox_invocation_failed"
            assert error_message == "shared sandbox failure"
            return []

    failing_runner = ArtifactFailingRunner()
    scheduler._runner = failing_runner

    with pytest.raises(
        ValidatorBatchFailedError,
        match="validator artifact breaker tripped across 3 artifacts",
    ) as exc_info:
        await scheduler.run(batch_id=batch_id, requested_artifacts=artifacts)

    assert exc_info.value.error_code == "artifact_breaker_tripped"
    assert exc_info.value.failure_detail.error_code == "artifact_breaker_tripped"
    assert len(sandbox_manager.starts) == 3
    assert len(sandbox_manager.stops) == 3
    assert failing_runner.failed_artifact_ids == [artifact.artifact_id for artifact in artifacts]
    assert failing_runner.recorded_artifact_ids == [artifact.artifact_id for artifact in artifacts]


async def test_evaluation_runner_issues_session_with_task_budget(
    blocking_executor: ThreadPoolExecutor,
) -> None:
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
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda artifact: artifact,
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    task = _task("budgeted", budget_usd=0.123)
    issued = scheduler._runner._issue_session(
        batch_id=uuid4(),
        uid=3,
        task=task,
    )

    assert issued.session.task_id == task.task_id
    assert issued.session.budget_usd == pytest.approx(0.123)


async def test_scheduler_runs_only_remaining_pairs(
    blocking_executor: ThreadPoolExecutor,
) -> None:
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
        blocking_executor=blocking_executor,
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


async def test_scheduler_skips_artifact_when_all_pairs_are_already_recorded(
    blocking_executor: ThreadPoolExecutor,
) -> None:
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
        blocking_executor=blocking_executor,
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
