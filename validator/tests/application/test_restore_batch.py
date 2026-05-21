from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from harnyx_validator.application.ports.platform import RestoreMetadata, RestoreRunsPage
from harnyx_validator.application.restore_batch import RestoreEvaluationBatch


@dataclass(slots=True)
class _AcceptedBatches:
    batch_id: UUID
    restore_calls: list[tuple[object, tuple[object, ...], tuple[object, ...]]] = field(default_factory=list)
    queued: list[UUID] = field(default_factory=list)

    def batch_for(self, batch_id: UUID) -> object:
        assert batch_id == self.batch_id
        return object()

    def restore_completed_runs(
        self,
        batch: object,
        items: tuple[object, ...],
        provider_evidence: tuple[object, ...],
    ) -> None:
        self.restore_calls.append((batch, items, provider_evidence))

    def queue_after_restore(self, batch_id: UUID) -> None:
        self.queued.append(batch_id)


@dataclass(slots=True)
class _Platform:
    metadata: RestoreMetadata
    pages: list[RestoreRunsPage]
    cursors: list[int] = field(default_factory=list)

    def get_restore_metadata(self, batch_id: UUID) -> RestoreMetadata:
        assert batch_id == self.metadata.batch_id
        return self.metadata

    def get_restore_runs_page(
        self,
        *,
        batch: object,
        snapshot_received_at: datetime,
        cursor: int,
        limit: int,
    ) -> RestoreRunsPage:
        _ = batch
        assert snapshot_received_at == self.metadata.snapshot_received_at
        assert limit == self.metadata.page_limit
        self.cursors.append(cursor)
        return self.pages.pop(0)


@dataclass(slots=True)
class _BatchActivity:
    stages: list[tuple[UUID, str]] = field(default_factory=list)

    def mark_artifact_stage(self, batch_id: UUID, stage: str) -> None:
        self.stages.append((batch_id, stage))


def test_restore_uses_null_next_cursor_not_advisory_total() -> None:
    batch_id = uuid4()
    snapshot = datetime.now(UTC)
    platform = _Platform(
        metadata=RestoreMetadata(
            batch_id=batch_id,
            snapshot_received_at=snapshot,
            total_restore_runs=99,
            page_limit=500,
        ),
        pages=[
            RestoreRunsPage(
                batch_id=batch_id,
                snapshot_received_at=snapshot,
                cursor=0,
                limit=500,
                next_cursor=None,
                items=(),
            )
        ],
    )
    accepted_batches = _AcceptedBatches(batch_id=batch_id)
    batch_activity = _BatchActivity()

    RestoreEvaluationBatch(
        accepted_batches=accepted_batches,
        platform=platform,
        batch_activity=batch_activity,
    ).restore(batch_id)

    assert platform.cursors == [0]
    assert accepted_batches.restore_calls == []
    assert accepted_batches.queued == [batch_id]
    assert batch_activity.stages == [(batch_id, "queued")]
