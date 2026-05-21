from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import EvaluationDetails, EvaluationError, MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain.session import Session
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchRunResult,
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.application.services.evaluation_batch import (
    EvaluationBatchConfig,
    MinerTaskBatchService,
)
from harnyx_validator.application.services.evaluation_batch_prep import BatchExecutionPlanner, RunContext
from harnyx_validator.application.services.evaluation_runner import ValidatorBatchFailedError
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
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


class DummyEvaluationRecordStore:
    def record(self, _result: MinerTaskRunSubmission) -> None:
        return None


def _progress(tmp_path: Path) -> FileBackedRunProgress:
    return FileBackedRunProgress(storage_root=tmp_path / "run-progress")


def _task(text: str) -> MinerTask:
    return MinerTask(
        task_id=uuid4(),
        query=Query(text=text),
        reference_answer=ReferenceAnswer(text=f"reference {text}"),
    )


def _batch() -> MinerTaskBatchSpec:
    task = _task("example")
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="abc", size_bytes=1)
    return MinerTaskBatchSpec(
        batch_id=uuid4(),
        cutoff_at="2025-01-01T00:00:00Z",
        created_at="2025-01-01T00:00:00Z",
        tasks=(task,),
        artifacts=(artifact,),
    )


def _run_context(batch: MinerTaskBatchSpec, tmp_path: Path) -> RunContext:
    return RunContext(
        batch_id=batch.batch_id,
        tasks=batch.tasks,
        config=EvaluationBatchConfig(state_dir=str(tmp_path)),
        base_options=SandboxOptions(image="sandbox:test", container_name="sandbox-base"),
        base_env={},
        base_volumes=(),
        state_dir=tmp_path,
    )


def _failure_submission(batch: MinerTaskBatchSpec) -> MinerTaskRunSubmission:
    task = batch.tasks[0]
    artifact = batch.artifacts[0]
    issued_at = datetime(2025, 10, 27, 0, 0, 0, tzinfo=UTC)
    completed_at = datetime(2025, 10, 27, 0, 0, 2, tzinfo=UTC)
    return MinerTaskRunSubmission(
        batch_id=batch.batch_id,
        validator_uid=41,
        run=MinerTaskRun(
            session_id=uuid4(),
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=None,
            details=EvaluationDetails(
                error=EvaluationError(code="batch_execution_failed", message="worker boom"),
                elapsed_ms=2000.0,
            ),
            completed_at=completed_at,
        ),
        score=0.0,
        usage=TokenUsageSummary.empty(),
        session=Session(
            session_id=uuid4(),
            uid=artifact.uid,
            task_id=task.task_id,
            issued_at=issued_at,
            expires_at=issued_at + timedelta(minutes=5),
            budget_usd=task.budget_usd,
        ),
    )


def test_batch_execution_planner_passes_task_parallelism_to_scheduler(
    tmp_path: Path,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    config = EvaluationBatchConfig(
        state_dir=str(tmp_path),
        artifact_parallelism=4,
        artifact_task_parallelism=7,
    )
    planner = BatchExecutionPlanner(
        subtensor_client=FakeSubtensorClient(),
        sandbox_manager=object(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=FakeReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda: SandboxOptions(image="sandbox:test", container_name="sandbox-base"),
        agent_resolver=lambda *_args: {},
        progress=_progress(tmp_path),
        config=config,
    )
    batch = _batch()

    run_ctx = planner.build_run_context(batch)
    _artifacts, scheduler = planner.prepare_execution(run_ctx, batch)

    assert scheduler._config.artifact_parallelism == 4
    assert scheduler._config.artifact_task_parallelism == 7


async def test_process_async_fails_batch_after_scheduler_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    batch = _batch()
    status = StatusProvider()
    service = MinerTaskBatchService(
        platform_client=object(),
        subtensor_client=FakeSubtensorClient(),
        sandbox_manager=object(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=FakeReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda: SandboxOptions(image="sandbox:test", container_name="sandbox-base"),
        agent_resolver=lambda *_args: {},
        progress=_progress(tmp_path),
        status_provider=status,
        config=EvaluationBatchConfig(state_dir=str(tmp_path)),
    )

    monkeypatch.setattr(
        service._planner,
        "build_run_context",
        lambda current_batch: _run_context(current_batch, tmp_path),
    )

    async def _raise_execute(_run_ctx: RunContext, _batch_spec: MinerTaskBatchSpec):
        raise RuntimeError("worker boom")

    monkeypatch.setattr(service, "_execute_batch", _raise_execute)

    logged: dict[str, object] = {}

    def _capture_logs(
        batch_id,
        batch_result: MinerTaskBatchRunResult,
        elapsed_seconds: float,
    ) -> None:
        logged["batch_id"] = batch_id
        logged["batch_result"] = batch_result
        logged["elapsed_seconds"] = elapsed_seconds

    monkeypatch.setattr(service, "_log_results", _capture_logs)

    with pytest.raises(ValidatorBatchFailedError, match="worker boom") as exc_info:
        await service.process_async(batch)

    assert exc_info.value.error_code == "batch_execution_failed"
    assert exc_info.value.failure_detail.error_code == "batch_execution_failed"
    assert exc_info.value.failure_detail.error_message == "worker boom"
    assert exc_info.value.failure_detail.exception_type == "RuntimeError"
    assert exc_info.value.failure_detail.traceback is not None
    assert "RuntimeError: worker boom" in exc_info.value.failure_detail.traceback
    assert logged == {}


async def test_process_async_fails_from_build_run_context_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    batch = _batch()
    status = StatusProvider()
    progress = _progress(tmp_path)
    accept_batch = AcceptEvaluationBatch(
        inbox=InMemoryBatchInbox(),
        status=status,
        progress=progress,
    )
    accept_batch.execute(batch)
    accept_batch.mark_processing(batch.batch_id)

    service = MinerTaskBatchService(
        platform_client=object(),
        subtensor_client=FakeSubtensorClient(),
        sandbox_manager=object(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=FakeReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda: SandboxOptions(image="sandbox:test", container_name="sandbox-base"),
        agent_resolver=lambda *_args: {},
        status_provider=status,
        config=EvaluationBatchConfig(state_dir=str(tmp_path)),
        progress=progress,
    )

    monkeypatch.setattr(
        service._planner,
        "build_run_context",
        lambda _current_batch: (_ for _ in ()).throw(RuntimeError("setup boom")),
    )
    with pytest.raises(ValidatorBatchFailedError, match="setup boom") as exc_info:
        await service.process_async(batch)

    assert exc_info.value.error_code == "batch_execution_failed"
    assert exc_info.value.failure_detail.error_code == "batch_execution_failed"
    assert exc_info.value.failure_detail.error_message == "setup boom"
    assert exc_info.value.failure_detail.exception_type == "RuntimeError"
    assert exc_info.value.failure_detail.traceback is not None
    assert "RuntimeError: setup boom" in exc_info.value.failure_detail.traceback
    assert accept_batch.lifecycle_for(batch.batch_id) == "processing"
    assert progress.recorded_pairs(batch.batch_id) == frozenset()
