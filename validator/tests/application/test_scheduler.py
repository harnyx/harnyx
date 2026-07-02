from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    MinerTask,
    MinerTaskErrorCode,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskRunSubmission,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
    SandboxFailureDiagnostics,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.application.ports.progress import RunProgressPage, RunProgressSummary
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo
from harnyx_validator.application.scheduler import (
    EvaluationScheduler,
    SchedulerConfig,
    _sandbox_failure_diagnostics_from_options,
)
from harnyx_validator.application.services.evaluation_runner import (
    ArtifactExecutionFailedError,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")
_ASSIGNMENT_TOKEN = "assignment-token"  # noqa: S105 - fixed test-only assignment token
_ASSIGNMENT_TOKEN_1 = "assignment-token-1"  # noqa: S105 - fixed test-only assignment token
_ASSIGNMENT_TOKEN_2 = "assignment-token-2"  # noqa: S105 - fixed test-only assignment token


def test_sandbox_failure_diagnostics_reads_docker_command_error_files(tmp_path) -> None:
    options = SandboxOptions(
        image="harnyx/sandbox:test",
        pull_policy="always",
        container_name="harnyx-sandbox-7-artifact-batch",
        env={
            "SECRET_TOKEN": "super-secret",
            "SANDBOX_HOST": "127.0.0.1",
        },
        failure_diagnostics_dir=str(tmp_path),
    )
    (tmp_path / "sandbox-options.json").write_text(
        '{"image":"harnyx/sandbox:test","pull_policy":"always","container_name":"harnyx-sandbox-7-artifact-batch"}',
        encoding="utf-8",
    )
    (tmp_path / "docker-inspect.json").write_text(
        '[{"Id":"container-id","State":{"Status":"exited","ExitCode":255,"OOMKilled":false,"Error":""}}]',
        encoding="utf-8",
    )
    (tmp_path / "docker-pull-result.json").write_text('{"returncode":0}', encoding="utf-8")
    (tmp_path / "docker-run-result.json").write_text('{"returncode":0}', encoding="utf-8")
    (tmp_path / "error.txt").write_text("sandbox start failed", encoding="utf-8")
    (tmp_path / "docker-logs.txt").write_text("", encoding="utf-8")
    (tmp_path / "docker-inspect.json.error.txt").write_text(
        "command=docker inspect stderr=No such container token=super-secret",
        encoding="utf-8",
    )
    (tmp_path / "docker-logs.txt.error.txt").write_text(
        "command=docker logs stderr=daemon unavailable token=super-secret",
        encoding="utf-8",
    )

    diagnostics = _sandbox_failure_diagnostics_from_options(options)

    assert diagnostics is not None
    assert diagnostics.docker_inspect_error_tail == (
        "command=docker inspect stderr=No such container token=<redacted>"
    )
    assert diagnostics.docker_logs_error_tail == (
        "command=docker logs stderr=daemon unavailable token=<redacted>"
    )


class _AssignedWork:
    def __init__(self, queued: tuple[MinerTaskWorkAssignment, ...] = ()) -> None:
        self.dispatch_ready = False
        self.queue: asyncio.Queue[MinerTaskWorkAssignment] = asyncio.Queue()
        for assignment in queued:
            self.queue.put_nowait(assignment)
        self.initial_claims: list[MinerTaskWorkAssignment] = []
        self.started: list[tuple[MinerTaskWorkAssignment, UUID]] = []

    async def take_for_startup(self) -> MinerTaskWorkAssignment:
        return await self.queue.get()

    def take_nowait_for_startup(self) -> MinerTaskWorkAssignment:
        return self.queue.get_nowait()

    def drain_for_setup_failure(self) -> tuple[MinerTaskWorkAssignment, ...]:
        drained: list[MinerTaskWorkAssignment] = []
        while True:
            try:
                drained.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return tuple(drained)

    def mark_dispatch_ready(self) -> None:
        self.dispatch_ready = True

    def claim_initial_for_dispatch(self, assignment: MinerTaskWorkAssignment) -> _ClaimedAssignedTaskFake:
        self.initial_claims.append(assignment)
        return _ClaimedAssignedTaskFake(self, assignment)

    async def claim_for_dispatch(self) -> _ClaimedAssignedTaskFake:
        return _ClaimedAssignedTaskFake(self, await self.queue.get())

    def claim_nowait_for_dispatch(self) -> _ClaimedAssignedTaskFake:
        return _ClaimedAssignedTaskFake(self, self.queue.get_nowait())

    def _mark_started(self, assignment: MinerTaskWorkAssignment, validator_session_id: UUID) -> None:
        self.started.append((assignment, validator_session_id))

    def _fail_before_start(self, _assignment: MinerTaskWorkAssignment, _result: PlatformOwnedTaskResult) -> None:
        return None


class _ClaimedAssignedTaskFake:
    def __init__(self, owner: _AssignedWork, assignment: MinerTaskWorkAssignment) -> None:
        self._owner = owner
        self._assignment = assignment

    @property
    def assignment(self) -> MinerTaskWorkAssignment:
        return self._assignment

    def mark_started(self, validator_session_id: UUID, **_: object) -> None:
        self._owner._mark_started(self._assignment, validator_session_id)

    def fail_before_start(self, result: PlatformOwnedTaskResult) -> None:
        self._owner._fail_before_start(self._assignment, result)


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

    def start_pending_receipt(self, *, started_call) -> None:
        return None

    def complete_pending_receipt(self, receipt, settle_usage):
        return settle_usage()

    def abandon_pending_receipt(self, receipt_id: str) -> None:
        return None

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
        self._submissions_by_pair: dict[tuple[UUID, UUID], MinerTaskRunSubmission] = {}
        self._sequence_by_pair: dict[tuple[UUID, UUID], int] = {}
        self._pair_by_sequence: dict[int, tuple[UUID, UUID]] = {}
        self._latest_attempt_number_by_pair: dict[tuple[UUID, UUID], int] = {}
        self._next_sequence = 1

    def register(self, _batch) -> None:
        return None

    def record(self, result: MinerTaskRunSubmission) -> None:
        pair = (result.run.artifact_id, result.run.task_id)
        self._recorded.add(pair)
        if pair not in self._sequence_by_pair:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._sequence_by_pair[pair] = sequence
            self._pair_by_sequence[sequence] = pair
        self._submissions_by_pair[pair] = result
        self._latest_attempt_number_by_pair[pair] = max(
            self._latest_attempt_number_by_pair.get(pair, 0),
            1,
        )

    def record_terminated_attempt(self, attempt: MinerTaskAttemptAuditRecord) -> None:
        pair = (attempt.artifact_id, attempt.task_id)
        self._latest_attempt_number_by_pair[pair] = max(
            self._latest_attempt_number_by_pair.get(pair, 0),
            attempt.attempt_number,
        )

    def next_attempt_number(self, _batch_id: UUID, artifact_id: UUID, task_id: UUID) -> int:
        return self._latest_attempt_number_by_pair.get((artifact_id, task_id), 0) + 1

    def recorded_pairs(self, _batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        return frozenset(self._recorded)

    def summary(self, batch_id: UUID) -> RunProgressSummary:
        completed = len(self._recorded)
        return {
            "batch_id": batch_id,
            "total": completed,
            "completed": completed,
            "remaining": 0,
            "latest_sequence": self._next_sequence - 1,
            "provider_evidence": (),
        }

    def completed_run_page(
        self,
        _batch_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> RunProgressPage:
        latest_sequence = self._next_sequence - 1
        sequences = tuple(range(after_sequence + 1, min(latest_sequence, after_sequence + limit) + 1))
        items = []
        for sequence in sequences:
            pair = self._pair_by_sequence.get(sequence)
            if pair is None:
                continue
            items.append(
                {
                    "sequence": sequence,
                    "kind": "completed_run",
                    "submission": self._submissions_by_pair[pair],
                    "attempt": None,
                }
            )
        next_after_sequence = items[-1]["sequence"] if items else after_sequence
        return {
            "batch_id": _batch_id,
            "after_sequence": after_sequence,
            "limit": limit,
            "latest_sequence": latest_sequence,
            "next_after_sequence": next_after_sequence,
            "has_more": next_after_sequence < latest_sequence,
            "items": tuple(items),
        }

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        return None

    def record_provider_call(self, *, session_id: UUID, provider: str, model: str) -> None:
        return None

    def record_provider_failure(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        return None

    def consume_provider_failures(self, session_id: UUID) -> tuple[dict[str, object], ...]:
        return ()

    def clear_task_session(self, session_id: UUID) -> None:
        return None


class FailingTerminatedAttemptProgressRecorder(DummyProgressRecorder):
    def record_terminated_attempt(self, attempt: MinerTaskAttemptAuditRecord) -> None:
        raise RuntimeError("terminated attempt progress write failed")


def _task(text: str, *, budget_usd: float = 0.05) -> MinerTask:
    return MinerTask(
        task_id=uuid4(),
        query=Query(text=text),
        reference_answer=ReferenceAnswer(text=f"reference {text}"),
        budget_usd=budget_usd,
    )



async def test_scheduler_returns_platform_result_for_assigned_task_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("assigned")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    progress = DummyProgressRecorder()
    now = datetime(2025, 10, 27, tzinfo=UTC)

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
        clock=lambda: now,
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=progress,
    )

    artifact = ScriptArtifactSpec(
        uid=3,
        artifact_id=uuid4(),
        content_hash="a",
        size_bytes=0,
        miner_hotkey_ss58="miner-hotkey",
    )
    batch_id = uuid4()

    async def fail_setup(**kwargs):
        _ = kwargs
        raise ArtifactExecutionFailedError(
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            message="artifact setup failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="sandbox_start_failed",
                error_message="artifact setup failed",
                occurred_at=now,
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
            ),
            completed_submissions=(),
            remaining_tasks=(task,),
        )

    monkeypatch.setattr(scheduler, "_start_artifact_with_retry", fail_setup)

    result = await scheduler.run_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    assert result.batch_id == batch_id
    assert result.artifact_id == artifact.artifact_id
    assert result.task_id == task.task_id
    assert result.result is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == "sandbox_start_failed"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
    assert result.terminal_attempt.validator_session_id != UUID(int=0)
    assert progress.next_attempt_number(batch_id, artifact.artifact_id, task.task_id) == 2


async def test_scheduler_runs_multiple_platform_assignments_in_one_artifact_sandbox(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    now = datetime(2025, 10, 27, tzinfo=UTC)

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                return TaskRunOutcome(
                    run=MinerTaskRun(
                        session_id=request.session_id,
                        uid=request.uid,
                        artifact_id=request.artifact_id,
                        task_id=request.task.task_id,
                        response=Response(text=f"answer {request.task.query.text}"),
                        details=EvaluationDetails(
                            score_breakdown=ScoreBreakdown(
                                comparison_score=1.0,
                                total_score=1.0,
                                scoring_version="v1",
                            ),
                            total_tool_usage=ToolUsageSummary.zero(),
                        ),
                        completed_at=now,
                    ),
                    usage=TokenUsageSummary.empty(),
                )

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
        clock=lambda: now,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()
    assignments = tuple(
        MinerTaskWorkAssignment(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            attempt_number=index,
            max_attempts=3,
            assignment_token=f"{_ASSIGNMENT_TOKEN}-{index}",
        )
        for index, task in enumerate(tasks, start=1)
    )
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    close_requested = asyncio.Event()
    close_requested.set()

    await scheduler.run_assigned_artifact_queue(
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=assignments,
        assigned_work=_AssignedWork(),
        close_requested=close_requested,
        result_queue=result_queue,
    )

    assert len(sandbox_manager.starts) == 1
    assert len(sandbox_manager.stops) == 1
    results = (result_queue.get_nowait(), result_queue.get_nowait())
    assert {result.task_id for result in results} == {task.task_id for task in tasks}
    assert {result.attempt_number for result in results} == {1, 2}
    assert all(
        result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
        for result in results
    )


async def test_scheduler_returns_pair_results_for_assigned_artifact_script_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    progress = DummyProgressRecorder()
    now = datetime(2025, 10, 27, tzinfo=UTC)
    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: now,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=progress,
    )
    artifact = ScriptArtifactSpec(
        uid=3,
        artifact_id=uuid4(),
        content_hash="a",
        size_bytes=0,
        miner_hotkey_ss58="miner-hotkey",
    )
    batch_id = uuid4()
    first_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=tasks[0],
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN_1,
    )
    second_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=tasks[1],
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN_2,
    )
    assigned_work = _AssignedWork((second_assignment,))
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()

    async def fail_setup(**kwargs):
        _ = kwargs
        raise ArtifactExecutionFailedError(
            error_code=MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED,
            message="script validation failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="script_validation_failed",
                error_message="script validation failed",
                occurred_at=now,
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
            ),
            completed_submissions=(),
            remaining_tasks=tasks,
        )

    monkeypatch.setattr(scheduler, "_start_artifact_with_retry", fail_setup)

    await scheduler.run_assigned_artifact_queue(
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(first_assignment,),
        assigned_work=assigned_work,
        close_requested=asyncio.Event(),
        result_queue=result_queue,
    )

    results = (result_queue.get_nowait(), result_queue.get_nowait())
    assert {result.task_id for result in results} == {task.task_id for task in tasks}
    assert all(result.result is not None for result in results)
    assert all(
        result.result is not None
        and result.result.run.details.error is not None
        and result.result.run.details.error.code == MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED
        for result in results
    )
    assert all(
        result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
        for result in results
    )
    assert all(
        result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
        for result in results
    )


async def test_scheduler_marks_assigned_work_dispatch_ready_before_runner(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("assigned")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=DummySandboxManager(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=DummyProgressRecorder(),
    )
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    assignment = MinerTaskWorkAssignment(
        batch_id=uuid4(),
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    assigned_work = _AssignedWork()
    observed_dispatch_ready: bool | None = None

    async def fake_assigned_queue_runner(**kwargs: object) -> None:
        nonlocal observed_dispatch_ready
        observed_dispatch_ready = kwargs["assigned_work"].dispatch_ready

    monkeypatch.setattr(scheduler._runner, "evaluate_assigned_task_queue", fake_assigned_queue_runner)

    await scheduler.run_assigned_artifact_queue(
        batch_id=assignment.batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=assigned_work,
        close_requested=asyncio.Event(),
        result_queue=asyncio.Queue(),
    )

    assert observed_dispatch_ready is True


async def test_scheduler_stops_assigned_artifact_sandbox_when_queue_execution_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("assigned")
    sandbox_manager = DummySandboxManager()
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=DummyProgressRecorder(),
    )
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    assignment = MinerTaskWorkAssignment(
        batch_id=uuid4(),
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    runner_started = asyncio.Event()

    async def blocked_assigned_queue_runner(**_kwargs: object) -> None:
        runner_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(scheduler._runner, "evaluate_assigned_task_queue", blocked_assigned_queue_runner)
    execution = asyncio.create_task(
        scheduler.run_assigned_artifact_queue(
            batch_id=assignment.batch_id,
            artifact=artifact,
            initial_assignments=(assignment,),
            assigned_work=_AssignedWork(),
            close_requested=asyncio.Event(),
            result_queue=asyncio.Queue(),
        )
    )

    await asyncio.wait_for(runner_started.wait(), timeout=1.0)
    execution.cancel()
    result = await asyncio.gather(execution, return_exceptions=True)

    assert isinstance(result[0], asyncio.CancelledError)
    assert len(sandbox_manager.stops) == 1


async def test_scheduler_returns_single_delivery_failure_for_assigned_artifact_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    now = datetime(2025, 10, 27, tzinfo=UTC)
    progress = DummyProgressRecorder()
    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=FakeSubtensorClient(),
        sandbox_manager=DummySandboxManager(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: now,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=progress,
    )
    artifact = ScriptArtifactSpec(
        uid=3,
        artifact_id=uuid4(),
        content_hash="a",
        size_bytes=0,
        miner_hotkey_ss58="miner-hotkey",
    )
    batch_id = uuid4()
    first_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=tasks[0],
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN_1,
    )
    second_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=tasks[1],
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN_2,
    )
    assigned_work = _AssignedWork((second_assignment,))
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    failure_detail = ValidatorBatchFailureDetail(
        error_code="sandbox_start_failed",
        error_message="artifact setup failed",
        occurred_at=now,
        artifact_id=artifact.artifact_id,
        uid=artifact.uid,
        sandbox_diagnostics=SandboxFailureDiagnostics(
            image="harnyx/sandbox:test",
            status="exited",
            exit_code=255,
        ),
    )

    async def fail_setup(**kwargs):
        _ = kwargs
        raise ArtifactExecutionFailedError(
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            message="artifact setup failed",
            failure_detail=failure_detail,
            completed_submissions=(),
            remaining_tasks=tasks,
        )

    monkeypatch.setattr(scheduler, "_start_artifact_with_retry", fail_setup)

    await scheduler.run_assigned_artifact_queue(
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(first_assignment,),
        assigned_work=assigned_work,
        close_requested=asyncio.Event(),
        result_queue=result_queue,
    )

    result = result_queue.get_nowait()
    assert result_queue.empty()
    assert assigned_work.queue.empty()
    assert result.task_id in {task.task_id for task in tasks}
    assert result.result is None
    assert result.terminal_attempt.error_code == "sandbox_start_failed"
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
    assert result.terminal_attempt.delivery_failure_detail == failure_detail
    assert progress.next_attempt_number(batch_id, artifact.artifact_id, result.task_id) == 2


async def test_scheduler_assigned_setup_failure_survives_progress_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("assigned")
    now = datetime(2025, 10, 27, tzinfo=UTC)
    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=FakeSubtensorClient(),
        sandbox_manager=DummySandboxManager(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: now,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=FailingTerminatedAttemptProgressRecorder(),
    )
    artifact = ScriptArtifactSpec(
        uid=3,
        artifact_id=uuid4(),
        content_hash="a",
        size_bytes=0,
        miner_hotkey_ss58="miner-hotkey",
    )

    async def fail_setup(**kwargs):
        _ = kwargs
        raise ArtifactExecutionFailedError(
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            message="artifact setup failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="sandbox_start_failed",
                error_message="artifact setup failed",
                occurred_at=now,
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
            ),
            completed_submissions=(),
            remaining_tasks=(task,),
        )

    monkeypatch.setattr(scheduler, "_start_artifact_with_retry", fail_setup)

    result = await scheduler.run_assigned_task(
        batch_id=uuid4(),
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    assert result.result is None
    assert result.terminal_attempt.error_code == "sandbox_start_failed"
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE


async def test_scheduler_assigned_results_survive_teardown_and_activity_failures(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("teardown")
    now = datetime(2025, 10, 27, tzinfo=UTC)

    class FailingStopSandboxManager(DummySandboxManager):
        def stop(self, deployment: SandboxDeployment) -> None:
            super().stop(deployment)
            raise RuntimeError("sandbox stop failed")

    class FailingActivity:
        def mark_batch_started(self, _batch_id: UUID) -> None:
            raise RuntimeError("batch start failed")

        def mark_artifact_started(self, _batch_id: UUID, _artifact_id: UUID) -> None:
            raise RuntimeError("artifact start failed")

        def mark_artifact_finished(self, _batch_id: UUID, _artifact_id: UUID) -> None:
            raise RuntimeError("artifact finish failed")

        def mark_batch_finished(self, _batch_id: UUID) -> None:
            raise RuntimeError("batch finish failed")

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                return TaskRunOutcome(
                    run=MinerTaskRun(
                        session_id=request.session_id,
                        uid=request.uid,
                        artifact_id=request.artifact_id,
                        task_id=request.task.task_id,
                        response=Response(text="answer"),
                        details=EvaluationDetails(
                            score_breakdown=ScoreBreakdown(
                                comparison_score=1.0,
                                total_score=1.0,
                                scoring_version="v1",
                            ),
                            total_tool_usage=ToolUsageSummary.zero(),
                        ),
                        completed_at=now,
                    ),
                    usage=TokenUsageSummary.empty(),
                )

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=FakeSubtensorClient(),
        sandbox_manager=FailingStopSandboxManager(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: now,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=DummyProgressRecorder(),
        activity=FailingActivity(),
    )
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    assignment = MinerTaskWorkAssignment(
        batch_id=uuid4(),
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    close_requested = asyncio.Event()
    close_requested.set()

    await scheduler.run_assigned_artifact_queue(
        batch_id=assignment.batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=_AssignedWork(),
        close_requested=close_requested,
        result_queue=result_queue,
    )

    result = result_queue.get_nowait()
    assert result.result is not None
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT



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
        progress=DummyProgressRecorder(),
    )

    task = _task("budgeted", budget_usd=0.123)
    issued = scheduler._runner._issue_session(
        batch_id=uuid4(),
        uid=3,
        task=task,
    )
    assert issued.session.task_id == task.task_id
    assert issued.session.budget_usd == pytest.approx(0.123)
