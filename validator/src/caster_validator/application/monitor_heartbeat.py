"""Use case for monitoring and restarting validator components."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class HeartbeatProbe:
    """Represents the heartbeat state of a monitored component."""

    component: str
    last_seen: datetime
    timeout: timedelta
    restart: Callable[[str], None]


class HeartbeatMonitor:
    """Evaluates heartbeat probes and triggers restarts when stale."""

    def __init__(self, *, clock: Callable[[], datetime]) -> None:
        self._clock = clock

    def evaluate(self, probes: Iterable[HeartbeatProbe]) -> list[str]:
        """Check all probes and restart components whose heartbeat expired."""
        now = self._clock()
        restarted: list[str] = []
        for probe in probes:
            if now - probe.last_seen > probe.timeout:
                probe.restart(probe.component)
                restarted.append(probe.component)
        return restarted


__all__ = ["HeartbeatMonitor", "HeartbeatProbe"]
