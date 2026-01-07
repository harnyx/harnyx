from __future__ import annotations

from datetime import UTC, datetime, timedelta

from caster_validator.application.monitor_heartbeat import HeartbeatMonitor, HeartbeatProbe


def test_heartbeat_monitor_triggers_restart_when_stale() -> None:
    restarted: list[str] = []

    def restart(component: str) -> None:
        restarted.append(component)

    monitor = HeartbeatMonitor(clock=lambda: datetime(2025, 10, 17, 12, tzinfo=UTC))
    probes = [
        HeartbeatProbe(
            component="scheduler",
            last_seen=datetime(2025, 10, 17, 11, 45, tzinfo=UTC),
            timeout=timedelta(minutes=10),
            restart=restart,
        ),
        HeartbeatProbe(
            component="rpc",
            last_seen=datetime(2025, 10, 17, 11, 59, tzinfo=UTC),
            timeout=timedelta(minutes=10),
            restart=restart,
        ),
    ]

    restarted_components = monitor.evaluate(probes)

    assert restarted_components == ["scheduler"]
    assert restarted == ["scheduler"]


def test_heartbeat_monitor_no_restart_when_recent() -> None:
    restarted: list[str] = []

    def restart(component: str) -> None:
        restarted.append(component)

    monitor = HeartbeatMonitor(clock=lambda: datetime(2025, 10, 17, 12, tzinfo=UTC))
    probes = [
        HeartbeatProbe(
            component="scheduler",
            last_seen=datetime(2025, 10, 17, 11, 55, tzinfo=UTC),
            timeout=timedelta(minutes=10),
            restart=restart,
        ),
    ]

    restarted_components = monitor.evaluate(probes)

    assert restarted_components == []
    assert restarted == []
