"""Champion-aware submission orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from caster_validator.application.ports.platform import PlatformPort
from caster_validator.application.ports.subtensor import SubtensorClientPort
from caster_validator.application.scheduling.gate import is_submission_window_open

weights_logger = logging.getLogger("caster_validator.weights.ranking")

# Default minimum blocks between weight submissions
DEFAULT_MIN_BLOCKS = 100


@dataclass(frozen=True)
class WeightSubmissionResult:
    """Champion-aware weight submission outcome."""

    final_top: tuple[int | None, int | None, int | None]
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
        min_blocks: int = DEFAULT_MIN_BLOCKS,
    ) -> None:
        self._subtensor = subtensor
        self._netuid = netuid
        self._clock = clock
        self._platform = platform
        self._min_blocks = min_blocks

    def try_submit(self) -> WeightSubmissionResult | None:
        """Submit weights if the submission window is open.

        Returns the submission result if weights were submitted, or None if
        the window is not yet open.
        """
        info = self._subtensor.validator_info()
        if not is_submission_window_open(
            self._subtensor,
            info.uid,
            min_blocks=self._min_blocks,
        ):
            weights_logger.debug(
                "weight submission window closed",
                extra={"uid": info.uid, "min_blocks": self._min_blocks},
            )
            return None
        return self.submit()

    def submit(
        self,
        *,
        roster: object | None = None,
    ) -> WeightSubmissionResult:
        """Submit weights unconditionally (caller must ensure window is open)."""
        selection = self._platform.get_champion_weights()
        weights = selection.weights
        final_top = selection.final_top
        if not weights:
            raise RuntimeError("platform returned empty weights")
        weights_logger.debug("submitting weights to subtensor", extra={"weights": weights})
        tx_hash = self._subtensor.submit_weights(weights)
        submitted_at = self._clock()
        weights_logger.info(
            "submitted champion weights from platform",
            extra={
                "event": "champion_weights_submitted",
                "final_top": [uid for uid in final_top if uid is not None],
                "weights": weights,
                "tx_hash": tx_hash,
                "submitted_at": submitted_at.isoformat(),
            },
        )
        return WeightSubmissionResult(final_top=final_top, weights=weights, tx_hash=tx_hash)


__all__ = ["WeightSubmissionResult", "WeightSubmissionService", "DEFAULT_MIN_BLOCKS"]
