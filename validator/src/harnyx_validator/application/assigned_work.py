"""Internal protocol for platform-owned assigned artifact work."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskWorkAssignment, PlatformOwnedTaskResult


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

    def fail_before_start(self, result: PlatformOwnedTaskResult) -> None:
        """Publish a terminal result and release the claim before sandbox execution starts."""


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
