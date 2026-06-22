"""Background worker for periodic weight submission."""

from __future__ import annotations

from datetime import UTC, datetime

from harnyx_commons.runtime.base_worker import BaseWorker
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.application.submit_weights import WeightSubmissionService
from harnyx_validator.infrastructure.observability.sentry import capture_exception
from harnyx_validator.infrastructure.transient_network import (
    TransientNetworkCause,
    classify_transient_network_failure,
)

# Default polling interval in seconds.
DEFAULT_POLL_INTERVAL = 300.0
_TRANSIENT_NETWORK_CAPTURE_ATTEMPTS = 3
_TRANSIENT_NETWORK_STATUS = "weight submission retrying after transient network failure"


class WeightWorker(BaseWorker):
    """Background worker that periodically attempts weight submission.

    The subtensor adapter classifies chain-level cadence refusals so the worker
    can retry on the next scheduled tick.
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
        self._transient_network_attempts = 0
        self._transient_network_captured = False

    def _tick(self) -> None:
        """Attempt to submit weights once."""
        try:
            result = self._submission_service.try_submit()
        except Exception as exc:
            cause = classify_transient_network_failure(exc)
            if cause is not None:
                self._record_transient_network_failure(cause, exc)
                raise
            self._reset_transient_network_failure()
            capture_exception(exc)
            raise
        self._reset_transient_network_failure()
        if self._status is not None:
            self._status.state.last_weight_error = None
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

    def _record_transient_network_failure(
        self,
        cause: TransientNetworkCause,
        exc: Exception,
    ) -> None:
        self._transient_network_attempts += 1
        self._logger.warning(
            "weight submission transient network failure",
            extra={
                "attempts": self._transient_network_attempts,
                "cause_kind": cause.kind,
            },
            exc_info=exc,
        )
        if self._status is not None:
            self._status.state.last_weight_error = _TRANSIENT_NETWORK_STATUS
        if (
            self._transient_network_attempts == _TRANSIENT_NETWORK_CAPTURE_ATTEMPTS
            and not self._transient_network_captured
        ):
            capture_exception(
                RuntimeError("weight worker transient network outage"),
                tags={
                    "failure_kind": "retryable_network",
                    "worker": self.worker_name,
                },
                context_name="retryable_network",
                context={
                    "attempts": self._transient_network_attempts,
                    "threshold": _TRANSIENT_NETWORK_CAPTURE_ATTEMPTS,
                    "cause_type": cause.exception_type,
                    "cause_kind": cause.kind,
                    "errno": cause.errno,
                },
                fingerprint=["validator-weight-worker", "retryable-network"],
            )
            self._transient_network_captured = True

    def _reset_transient_network_failure(self) -> None:
        self._transient_network_attempts = 0
        self._transient_network_captured = False

    def _on_error(self) -> None:
        """Update status on submission failure."""
        if self._status is not None and self._transient_network_attempts == 0:
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
    "DEFAULT_POLL_INTERVAL",
    "WeightWorker",
    "create_weight_worker",
]
