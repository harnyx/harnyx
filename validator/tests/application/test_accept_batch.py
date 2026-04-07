from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import Session, SessionStatus
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.application.ports.progress import ProviderFailureEvidence
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox


class ProgressSpy:
    def __init__(self) -> None:
        self.register_attempts: list[MinerTaskBatchSpec] = []
        self.registered: list[MinerTaskBatchSpec] = []
        self.restore_attempts: list[tuple[MinerTaskBatchSpec, tuple[MinerTaskRunSubmission, ...]]] = []
        self.restore_provider_evidence_attempts: list[
            tuple[MinerTaskBatchSpec, tuple[ProviderFailureEvidence, ...]]
        ] = []
        self._batches_by_id: dict[UUID, MinerTaskBatchSpec] = {}
        self._recorded_by_batch: dict[UUID, set[tuple[UUID, UUID]]] = {}

    def register(self, batch: MinerTaskBatchSpec) -> None:
        self.register_attempts.append(batch)
        existing = self._batches_by_id.get(batch.batch_id)
        if existing is not None:
            if existing != batch:
                raise RuntimeError("batch_id already exists with different contents")
            return
        self._batches_by_id[batch.batch_id] = batch
        self.registered.append(batch)
        self._recorded_by_batch.setdefault(batch.batch_id, set())

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        return frozenset(self._recorded_by_batch.get(batch_id, set()))

    def set_recorded_pairs(
        self,
        batch: MinerTaskBatchSpec,
        pairs: Iterable[tuple[UUID, UUID]],
    ) -> None:
        self._recorded_by_batch[batch.batch_id] = set(pairs)

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        submissions: Iterable[MinerTaskRunSubmission],
        provider_evidence: Iterable[ProviderFailureEvidence] = (),
    ) -> None:
        restored = tuple(submissions)
        restored_provider_evidence = tuple(provider_evidence)
        self.register(batch)
        if restored:
            self.restore_attempts.append((batch, restored))
        if restored_provider_evidence:
            self.restore_provider_evidence_attempts.append((batch, restored_provider_evidence))
        bucket = self._recorded_by_batch.setdefault(batch.batch_id, set())
        for submission in restored:
            bucket.add((submission.run.artifact_id, submission.run.task_id))


def _make_batch(*, batch_id: UUID | None = None, query_text: str = "example") -> MinerTaskBatchSpec:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text=query_text),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="abc", size_bytes=1)
    return MinerTaskBatchSpec(
        batch_id=batch_id or uuid4(),
        cutoff_at="2025-01-01T00:00:00Z",
        created_at="2025-01-01T00:00:00Z",
        tasks=(task,),
        artifacts=(artifact,),
    )


def _all_pairs(batch: MinerTaskBatchSpec) -> frozenset[tuple[UUID, UUID]]:
    return frozenset((artifact.artifact_id, task.task_id) for artifact in batch.artifacts for task in batch.tasks)


def _completed_submission(batch: MinerTaskBatchSpec) -> MinerTaskRunSubmission:
    artifact = batch.artifacts[0]
    task = batch.tasks[0]
    issued_at = datetime.now(UTC)
    completed_at = issued_at + timedelta(seconds=5)
    session = Session(
        session_id=uuid4(),
        uid=artifact.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=completed_at + timedelta(minutes=5),
        budget_usd=task.budget_usd,
        status=SessionStatus.COMPLETED,
    )
    details = EvaluationDetails(
        score_breakdown=ScoreBreakdown(
            comparison_score=1.0,
            similarity_score=1.0,
            total_score=1.0,
            scoring_version="v1",
        ),
        total_tool_usage=ToolUsageSummary.zero(),
    )
    return MinerTaskRunSubmission(
        batch_id=batch.batch_id,
        validator_uid=1,
        run=MinerTaskRun(
            session_id=session.session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=Response(text="completed"),
            details=details,
            completed_at=completed_at,
        ),
        score=1.0,
        usage=TokenUsageSummary.empty(),
        session=session,
    )


def _failed_submission(batch: MinerTaskBatchSpec, *, error_code: str) -> MinerTaskRunSubmission:
    artifact = batch.artifacts[0]
    task = batch.tasks[0]
    issued_at = datetime.now(UTC)
    completed_at = issued_at + timedelta(seconds=5)
    session = Session(
        session_id=uuid4(),
        uid=artifact.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=completed_at + timedelta(minutes=5),
        budget_usd=task.budget_usd,
        status=SessionStatus.COMPLETED,
    )
    details = EvaluationDetails(
        error=EvaluationError(code=error_code, message="terminal timeout"),
        total_tool_usage=ToolUsageSummary.zero(),
        elapsed_ms=5000.0,
    )
    return MinerTaskRunSubmission(
        batch_id=batch.batch_id,
        validator_uid=1,
        run=MinerTaskRun(
            session_id=session.session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=None,
            details=details,
            completed_at=completed_at,
        ),
        score=0.0,
        usage=TokenUsageSummary.empty(),
        session=session,
    )


def test_accept_batch_ignores_exact_duplicate_replay_while_queued() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    accept_batch.execute(batch)

    assert len(inbox) == 1
    assert status.state.queued_batches == 1
    assert progress.register_attempts == [batch, batch]
    assert progress.registered == [batch]


def test_accept_batch_ignores_exact_duplicate_replay_while_processing() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    accept_batch.execute(batch)

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert progress.register_attempts == [batch, batch]
    assert progress.registered == [batch]


def test_accept_batch_ignores_exact_duplicate_replay_while_completed() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    progress.set_recorded_pairs(batch, _all_pairs(batch))
    accept_batch.mark_completed(batch.batch_id)
    accept_batch.execute(batch)

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert progress.register_attempts == [batch, batch]
    assert progress.registered == [batch]


def test_accept_batch_marks_new_batch_completed_without_queue_when_restore_runs_cover_all_pairs() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()
    restored = _completed_submission(batch)

    accept_batch.execute(batch, restore_runs=(restored,))

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert progress.register_attempts == [batch]
    assert progress.restore_attempts == [(batch, (restored,))]


def test_accept_batch_marks_new_batch_completed_for_terminal_timeout_restore() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()
    restored = _failed_submission(batch, error_code="timeout_miner_owned")

    accept_batch.execute(batch, restore_runs=(restored,))

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert progress.register_attempts == [batch]
    assert progress.restore_attempts == [(batch, (restored,))]


def test_accept_batch_duplicate_processing_replay_becomes_completed_when_all_pairs_are_recorded() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    progress.set_recorded_pairs(batch, _all_pairs(batch))
    accept_batch.execute(batch)

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert progress.register_attempts == [batch, batch]


def test_accept_batch_duplicate_replay_can_restore_completion_without_conflicting_batch_identity() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()
    restored = _completed_submission(batch)

    accept_batch.execute(batch)
    accept_batch.execute(batch, restore_runs=(restored,))

    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert progress.register_attempts == [batch, batch]
    assert progress.restore_attempts == [(batch, (restored,))]


def test_accept_batch_duplicate_processing_replay_keeps_processing_lifecycle() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()
    restored = _completed_submission(batch)

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    accept_batch.execute(batch, restore_runs=(restored,))

    assert accept_batch.lifecycle_for(batch.batch_id) == "processing"
    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert progress.restore_attempts == [(batch, (restored,))]


def test_accept_batch_forwards_restore_provider_evidence_alongside_terminal_timeout_failure() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()
    restored = _failed_submission(batch, error_code="timeout_inconclusive")
    provider_evidence = (
        {
            "provider": "openai",
            "model": "gpt-4o",
            "total_calls": 3,
            "failed_calls": 1,
        },
    )

    accept_batch.execute(
        batch,
        restore_runs=(restored,),
        restore_provider_evidence=provider_evidence,
    )

    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert progress.restore_attempts == [(batch, (restored,))]
    assert progress.restore_provider_evidence_attempts == [(batch, provider_evidence)]


def test_begin_processing_skips_dequeued_batch_restored_to_completion() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()
    restored = _completed_submission(batch)

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.execute(batch, restore_runs=(restored,))

    assert accept_batch.begin_processing(batch.batch_id) is False
    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert len(inbox) == 0
    assert status.state.queued_batches == 0
    assert progress.restore_attempts == [(batch, (restored,))]


def test_accept_batch_mark_completed_rejects_when_pairs_are_missing() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)

    with pytest.raises(RuntimeError, match="cannot mark batch completed before all pairs are recorded"):
        accept_batch.mark_completed(batch.batch_id)
    assert accept_batch.lifecycle_for(batch.batch_id) == "processing"
    assert len(inbox) == 0
    assert status.state.queued_batches == 0


def test_accept_batch_mark_completed_marks_completed_when_all_pairs_are_recorded() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch = _make_batch()

    accept_batch.execute(batch)
    assert inbox.next() == batch
    status.state.queued_batches = len(inbox)
    accept_batch.mark_processing(batch.batch_id)
    progress.set_recorded_pairs(batch, _all_pairs(batch))

    accept_batch.mark_completed(batch.batch_id)
    assert accept_batch.lifecycle_for(batch.batch_id) == "completed"
    assert len(inbox) == 0
    assert status.state.queued_batches == 0


def test_accept_batch_rejects_conflicting_replay() -> None:
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    progress = ProgressSpy()
    accept_batch = AcceptEvaluationBatch(inbox=inbox, status=status, progress=progress)
    batch_id = uuid4()
    batch = _make_batch(batch_id=batch_id, query_text="original")
    conflicting = _make_batch(batch_id=batch_id, query_text="different")

    accept_batch.execute(batch)

    with pytest.raises(RuntimeError, match="batch_id already exists with different contents"):
        accept_batch.execute(conflicting)

    assert len(inbox) == 1
    assert status.state.queued_batches == 1
    assert progress.register_attempts == [batch]
    assert progress.registered == [batch]
