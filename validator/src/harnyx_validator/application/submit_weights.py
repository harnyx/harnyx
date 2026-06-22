"""Champion-aware submission orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from harnyx_validator.application.ports.platform import PlatformPort
from harnyx_validator.application.ports.subtensor import SubtensorClientPort, WeightSubmissionTooEarlyError

weights_logger = logging.getLogger("harnyx_validator.weights.ranking")


@dataclass(frozen=True)
class WeightSubmissionResult:
    """Champion-aware weight submission outcome."""

    champion_uid: int | None
    weights: dict[int, float]
    tx_hash: str


class WeightSubmissionService:
    """Submits platform-provided weights to Subtensor."""

    def __init__(
        self,
        *,
        subtensor: SubtensorClientPort,
        netuid: int,
        clock: Callable[[], datetime],
        platform: PlatformPort,
    ) -> None:
        self._subtensor = subtensor
        self._netuid = netuid
        self._clock = clock
        self._platform = platform

    def try_submit(self) -> WeightSubmissionResult | None:
        """Try to submit weights.

        Returns the submission result if weights were submitted, or None if the
        chain says the validator must wait before submitting again.
        """
        try:
            return self.submit()
        except WeightSubmissionTooEarlyError as exc:
            # A chain-level too-early refusal is harmless; the next scheduled attempt will retry.
            weights_logger.debug(
                "weight submission skipped because chain reported the attempt is too early",
                exc_info=exc,
            )
            return None

    def submit(self) -> WeightSubmissionResult:
        """Submit weights using the platform-provided champion scores."""
        selection = self._platform.get_champion_weights()
        weights = selection.weights
        champion_uid = selection.champion_uid
        if not weights:
            raise RuntimeError("platform returned empty weights")
        weights_logger.debug("submitting weights to subtensor", extra={"weights": weights})
        tx_hash = self._subtensor.submit_weights(weights)
        submitted_at = self._clock()
        weights_logger.info(
            "submitted champion weights from platform",
            extra={
                "event": "champion_weights_submitted",
                "champion_uid": champion_uid,
                "weights": weights,
                "tx_hash": tx_hash,
                "submitted_at": submitted_at.isoformat(),
            },
        )
        return WeightSubmissionResult(champion_uid=champion_uid, weights=weights, tx_hash=tx_hash)


__all__ = ["WeightSubmissionResult", "WeightSubmissionService"]
