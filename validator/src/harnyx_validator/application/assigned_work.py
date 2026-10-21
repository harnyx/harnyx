"""Internal protocol for platform-owned assigned artifact work."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

from harnyx_commons.domain.miner_task import MinerTaskErrorCode
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
    ValidatorBatchFailureDetail,
)

TASK_START_WAIT_SECONDS = 300.0
TASK_START_WAIT_ATTEMPTS = 2
TASK_START_TIMEOUT_SECONDS = TASK_START_WAIT_SECONDS * TASK_START_WAIT_ATTEMPTS


def task_start_timeout_result(assignment: MinerTaskWorkAssignment, occurred_at: datetime) -> PlatformOwnedTaskResult:
    """Close a validator delivery when its bounded pre-execution waiting is exhausted."""
    error_code = str(MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT)
    return PlatformOwnedTaskResult(
        batch_id=assignment.batch_id,
        artifact_id=assignment.artifact.artifact_id,
        task_id=assignment.task.task_id,
        attempt_number=assignment.attempt_number,
        terminal_attempt=MinerTaskAttemptAuditRecord(
            validator_session_id=uuid4(),
            batch_id=assignment.batch_id,
            artifact_id=assignment.artifact.artifact_id,
            task_id=assignment.task.task_id,
            attempt_number=assignment.attempt_number,
            uid=assignment.artifact.uid,
            miner_hotkey_ss58=assignment.artifact.miner_hotkey_ss58 or "unknown-miner-hotkey",
            started_at=occurred_at,
            finished_at=occurred_at,
            status=MinerTaskAttemptStatus.FAILED,
            error_code=error_code,
            error_summary_code=error_code,
            retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
            terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
            max_attempts=assignment.max_attempts,
            delivery_failure_detail=ValidatorBatchFailureDetail(
                error_code=error_code,
                error_message="Validator could not start the task within its two five-minute waiting periods",
                occurred_at=occurred_at,
                artifact_id=assignment.artifact.artifact_id,
                task_id=assignment.task.task_id,
                uid=assignment.artifact.uid,
                exception_type="TimeoutError",
            ),
        ),
    )


class ClaimedAssignedTask(Protocol):
    """Assignment claimed for validator session start."""

    def remaining_dispatch_seconds(self) -> float:
        """Remaining original dispatch lease, including admission wait."""
        ...

    def is_live(self) -> bool:
        """Whether this unstarted assignment can still start."""
        ...

    def release_to_queue_if_live(self) -> None:
        """Release an unstarted claim without renewing its dispatch lease."""
        ...

    @property
    def assignment(self) -> MinerTaskWorkAssignment:
        """The platform-owned assignment carried by this claim."""

    def mark_started(self, validator_session_id: UUID, *, started_at: datetime) -> None:
        """Attach the issued validator session and make the claim reportable."""

    def fail_before_start(self, result: PlatformOwnedTaskResult) -> bool:
        """Publish and return True only if this preparation still owns the task."""


class AssignedArtifactWork(Protocol):
    """Worker-owned handle for one platform-owned artifact assignment group."""

    async def take_for_startup(self) -> MinerTaskWorkAssignment:
        """Read queued work while the artifact sandbox is still starting."""

    def take_nowait_for_startup(self) -> MinerTaskWorkAssignment:
        """Read queued startup work without claiming task dispatch."""

    def drain_for_setup_failure(self) -> tuple[MinerTaskWorkAssignment, ...]:
        """Drain not-yet-dispatched assignments for setup-failure result creation."""

    def mark_dispatch_ready(self) -> None:
        """Mark the artifact ready to dispatch assigned tasks."""

    def claim_initial_for_dispatch(self, assignment: MinerTaskWorkAssignment) -> ClaimedAssignedTask | None:
        """Claim a startup-drained assignment before validator session start."""

    async def claim_for_dispatch(self) -> ClaimedAssignedTask:
        """Wait for and claim a queued assignment before validator session start."""

    def claim_nowait_for_dispatch(self) -> ClaimedAssignedTask:
        """Claim a queued assignment before validator session start without waiting."""


class PhaseRecorder(Protocol):
    """Records the currently awaited assigned-attempt phase."""

    def mark(self, phase: str) -> str:
        """Record that a named awaited phase is about to run."""


__all__ = ["AssignedArtifactWork", "ClaimedAssignedTask", "PhaseRecorder"]
