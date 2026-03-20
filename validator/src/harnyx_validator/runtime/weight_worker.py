"""Background worker for tempo-aware weight submission."""

from __future__ import annotations

from datetime import UTC, datetime

from harnyx_commons.runtime.base_worker import BaseWorker
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.application.submit_weights import (
    DEFAULT_MIN_BLOCKS,
    WeightSubmissionService,
)
from harnyx_validator.infrastructure.observability.sentry import capture_exception

# Default polling interval in seconds
DEFAULT_POLL_INTERVAL = 30.0


class WeightWorker(BaseWorker):
    """Background worker that submits weights on a tempo-aware schedule.

    This worker polls periodically and submits weights to Subtensor when
    the submission window is open (based on min_blocks since last update).
    """

    worker_name = "validator-weight-worker"
    logger_name = "harnyx_validator.weight_worker"
    default_poll_interval = DEFAULT_POLL_INTERVAL

    def __init__(
        self,
        *,
        submission_service: WeightSubmissionService,
        status_provider: StatusProvider | None = None,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        super().__init__(poll_interval=poll_interval_seconds)
        self._submission_service = submission_service
        self._status = status_provider

    def _tick(self) -> None:
        """Attempt to submit weights if the window is open."""
        try:
            result = self._submission_service.try_submit()
        except Exception as exc:
            capture_exception(exc)
            raise
        if result is not None:
            self._logger.info(
                "weights submitted",
                extra={
                    "tx_hash": result.tx_hash,
                    "weight_count": len(result.weights),
                    "champion_uid": result.champion_uid,
                },
            )
            if self._status is not None:
                self._status.state.last_weight_submission_at = datetime.now(UTC)
                self._status.state.last_weight_error = None

    def _on_error(self) -> None:
        """Update status on submission failure."""
        if self._status is not None:
            # The exception message is logged by base class; we just track state
            self._status.state.last_weight_error = "weight submission failed (see logs)"


def create_weight_worker(
    *,
    submission_service: WeightSubmissionService,
    status_provider: StatusProvider | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL,
) -> WeightWorker:
    """Factory function to create a WeightWorker with injected dependencies."""
    return WeightWorker(
        submission_service=submission_service,
        status_provider=status_provider,
        poll_interval_seconds=poll_interval_seconds,
    )


__all__ = [
    "DEFAULT_MIN_BLOCKS",
    "DEFAULT_POLL_INTERVAL",
    "WeightWorker",
    "create_weight_worker",
]
