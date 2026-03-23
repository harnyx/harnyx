from __future__ import annotations

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
from harnyx_validator.application.services.evaluation_batch_prep import RunContext
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.infrastructure.state.run_progress import InMemoryRunProgress
from validator.tests.fixtures.fakes import FakeReceiptLog
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")


class DummyEvaluationRecordStore:
    def record(self, _result: MinerTaskRunSubmission) -> None:
        return None


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


async def test_process_async_terminalizes_missing_pairs_after_scheduler_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch()
    status = StatusProvider()
    recovered_submission = _failure_submission(batch)
    service = MinerTaskBatchService(
        platform_client=object(),
        subtensor_client=FakeSubtensorClient(),
        sandbox_manager=object(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=FakeReceiptLog(),
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda: SandboxOptions(image="sandbox:test", container_name="sandbox-base"),
        agent_resolver=lambda *_args: {},
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

    recovery_calls: list[tuple[MinerTaskBatchSpec, str, str]] = []

    async def _recover_missing_pairs(
        *,
        batch: MinerTaskBatchSpec,
        error_code: str,
        error_message: str,
    ) -> list[MinerTaskRunSubmission]:
        recovery_calls.append((batch, error_code, error_message))
        return [recovered_submission]

    monkeypatch.setattr(service._failure_recorder, "record_failure_for_missing_pairs", _recover_missing_pairs)

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

    await service.process_async(batch)

    assert recovery_calls == [(batch, "batch_execution_failed", "worker boom")]
    assert status.state.last_error is None
    assert status.state.last_completed_at is not None
    assert status.state.running is False
    assert logged["batch_id"] == batch.batch_id
    assert logged["batch_result"] == MinerTaskBatchRunResult(
        batch_id=batch.batch_id,
        tasks=batch.tasks,
        runs=(recovered_submission,),
    )
    assert logged["elapsed_seconds"] == 0.0


async def test_process_async_recovers_from_build_run_context_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch()
    status = StatusProvider()
    progress = InMemoryRunProgress()
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
    service._failure_recorder._validator_uid = 41

    await service.process_async(batch)

    accept_batch.mark_completed(batch.batch_id)

    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert progress.recorded_pairs(batch.batch_id) == frozenset(
        (artifact.artifact_id, task.task_id)
        for artifact in batch.artifacts
        for task in batch.tasks
    )
    assert status.state.running is False
    assert status.state.last_error is None
    assert status.state.last_completed_at is not None
