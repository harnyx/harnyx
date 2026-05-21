"""Use case for restoring persisted validator batch progress before evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec
from harnyx_validator.application.ports.platform import PlatformPort, RestoreMetadata, RestoreRunsPage
from harnyx_validator.application.services.evaluation_runner import ValidatorBatchFailureDetail
from harnyx_validator.application.status import BatchActivityTracker

logger = logging.getLogger("harnyx_validator.restore_batch")


class RestoreOutcome(StrEnum):
    SUCCEEDED = "succeeded"


@dataclass(frozen=True, slots=True)
class RestoreStartDecision:
    batch_id: UUID
    should_start_restore: bool


@dataclass(slots=True)
class RestoreEvaluationBatch:
    accepted_batches: AcceptEvaluationBatch
    platform: PlatformPort
    batch_activity: BatchActivityTracker

    def accept(self, batch: MinerTaskBatchSpec) -> RestoreStartDecision:
        should_start_restore = self.accepted_batches.register_for_restore(batch)
        if should_start_restore:
            self.batch_activity.mark_artifact_stage(batch.batch_id, "restore")
        return RestoreStartDecision(
            batch_id=batch.batch_id,
            should_start_restore=should_start_restore,
        )

    def restore(self, batch_id: UUID) -> RestoreOutcome:
        batch = self.accepted_batches.batch_for(batch_id)
        metadata = self.platform.get_restore_metadata(batch_id)
        self._validate_metadata(batch_id=batch_id, metadata=metadata)
        cursor = 0
        provider_evidence_applied = False
        while True:
            page = self.platform.get_restore_runs_page(
                batch=batch,
                snapshot_received_at=metadata.snapshot_received_at,
                cursor=cursor,
                limit=metadata.page_limit,
            )
            self._validate_page(metadata=metadata, page=page, cursor=cursor)
            provider_evidence = (
                metadata.provider_model_evidence
                if not provider_evidence_applied
                else ()
            )
            if page.items or provider_evidence:
                self.accepted_batches.restore_completed_runs(batch, page.items, provider_evidence)
            if provider_evidence:
                provider_evidence_applied = True
            logger.info(
                "validator restore page applied",
                extra={
                    "data": {
                        "batch_id": str(batch_id),
                        "cursor": page.cursor,
                        "next_cursor": page.next_cursor,
                        "item_count": len(page.items),
                    }
                },
            )
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        if not provider_evidence_applied and metadata.provider_model_evidence:
            self.accepted_batches.restore_completed_runs(batch, (), metadata.provider_model_evidence)
        self.accepted_batches.queue_after_restore(batch_id)
        self.batch_activity.mark_artifact_stage(batch_id, "queued")
        logger.info("validator restore completed", extra={"data": {"batch_id": str(batch_id)}})
        return RestoreOutcome.SUCCEEDED

    def mark_restore_failed(self, batch_id: UUID, *, error: Exception) -> None:
        self.accepted_batches.mark_failed(
            batch_id,
            error_code="restore_failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="restore_failed",
                error_message=str(error) or type(error).__name__,
                occurred_at=datetime.now(UTC),
                exception_type=type(error).__name__,
            ),
            terminal_at=datetime.now(UTC),
        )

    @staticmethod
    def _validate_metadata(*, batch_id: UUID, metadata: RestoreMetadata) -> None:
        if metadata.batch_id != batch_id:
            raise RuntimeError("restore metadata batch_id mismatch")
        if metadata.total_restore_runs < 0:
            raise RuntimeError("restore metadata total_restore_runs must be non-negative")
        if metadata.page_limit < 1:
            raise RuntimeError("restore metadata page_limit must be positive")

    @staticmethod
    def _validate_page(*, metadata: RestoreMetadata, page: RestoreRunsPage, cursor: int) -> None:
        if page.batch_id != metadata.batch_id:
            raise RuntimeError("restore page batch_id mismatch")
        if page.snapshot_received_at != metadata.snapshot_received_at:
            raise RuntimeError("restore page snapshot mismatch")
        if page.cursor != cursor:
            raise RuntimeError("restore page cursor mismatch")
        if page.limit < 1:
            raise RuntimeError("restore page limit must be positive")
        if page.next_cursor is not None and page.next_cursor < cursor:
            raise RuntimeError("restore page next_cursor moved backwards")
        if page.items and page.next_cursor is not None and page.next_cursor <= cursor:
            raise RuntimeError("restore page did not advance cursor")
        if page.next_cursor is not None and page.next_cursor != cursor + len(page.items):
            raise RuntimeError("restore page next_cursor does not match item count")
        if page.next_cursor is not None and not page.items:
            raise RuntimeError("restore page was empty before final page")


__all__ = ["RestoreEvaluationBatch", "RestoreOutcome", "RestoreStartDecision"]
