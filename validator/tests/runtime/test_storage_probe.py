"""Probe failure cannot change execution or repeatedly allocate storage."""
from unittest.mock import Mock

import pytest

from harnyx_validator.runtime.storage_probe import PROBE_LABELS, PROBE_POOL, StorageProbeWorker


def worker(factory, report):
    return StorageProbeWorker(factory, report, iops=32, bytes_per_second=4 * 1024**2)


@pytest.mark.parametrize("failure", [None, "identity", "recover", "cleanup"])
def test_probe_reports_failure_and_always_attempts_owned_cleanup(failure, monkeypatch):
    secret = "synthetic-probe-secret"  # noqa: S105
    monkeypatch.setenv("TEST_STORAGE_SECRET", secret)
    manager = Mock(stage="mount")
    factory = Mock(return_value=manager)
    if failure == "identity":
        factory.side_effect = RuntimeError(secret)
    elif failure == "recover":
        manager.recover.side_effect = RuntimeError(secret)
    elif failure == "cleanup":
        manager.finish_probe.side_effect = RuntimeError(secret)
    probe = worker(factory, Mock())
    result = probe._probe()
    assert result.status == ("succeeded" if failure is None else "failed")
    assert secret not in result.model_dump_json()
    if failure == "identity":
        manager.finish_probe.assert_not_called()
        assert result.cleanup == "not_needed"
    else:
        manager.recover.assert_called_once_with(
            sandbox_labels=PROBE_LABELS, sandbox_name_prefix=f"{PROBE_POOL}-consumer-",
        )
        manager.finish_probe.assert_called_once()
        assert result.cleanup == ("unconfirmed" if failure == "cleanup" else "succeeded")
    if failure:
        assert "RuntimeError" in result.details + result.cleanup_details


def test_delivery_retries_do_not_repeat_probe_or_affect_registration():
    manager = Mock(stage="mount")
    send = Mock(side_effect=RuntimeError("old platform rejects diagnostics"))
    probe = worker(Mock(return_value=manager), send)
    probe._stop = Mock()
    probe._stop.is_set.return_value = False
    probe._stop.wait.return_value = False
    probe._run()
    manager.recover.assert_called_once()
    manager.finish_probe.assert_called_once()
    assert send.call_count == 3
    assert len({call.args[0].probe_id for call in send.call_args_list}) == 1


def test_shutdown_before_start_does_not_allocate():
    factory = Mock()
    send = Mock()
    probe = worker(factory, send)
    probe.stop()
    probe._run()
    factory.assert_not_called()
    send.assert_not_called()
