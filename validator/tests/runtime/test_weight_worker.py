from __future__ import annotations

import errno
import logging
import socket

import pytest

import harnyx_validator.runtime.weight_worker as worker_mod
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.application.submit_weights import WeightSubmissionResult
from harnyx_validator.runtime.weight_worker import WeightWorker


class _FailingSubmissionService:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("boom")

    def try_submit(self) -> None:
        raise self.exc


class _SequenceSubmissionService:
    def __init__(self, outcomes: list[Exception | WeightSubmissionResult | None]) -> None:
        self._outcomes = outcomes

    def try_submit(self) -> WeightSubmissionResult | None:
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_weight_worker_success_log_includes_submitted_weights(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = WeightSubmissionResult(
        champion_uid=5,
        weights={5: 0.6, 1: 0.4},
        tx_hash="0xabc",
    )
    worker = WeightWorker(submission_service=_SequenceSubmissionService([result]))

    with caplog.at_level(logging.INFO, logger="harnyx_validator.weight_worker"):
        worker._tick()

    record = next(record for record in caplog.records if record.message == "weights submitted")
    assert record.data == {
        "event": "validator_weight_worker_submitted",
        "tx_hash": "0xabc",
        "weight_count": 2,
        "champion_uid": 5,
        "weights": {5: 0.6, 1: 0.4},
    }


def test_weight_worker_captures_sanitized_transient_network_outage_once_at_threshold(monkeypatch) -> None:
    captured: list[tuple[BaseException, dict[str, object]]] = []

    def capture(exc: BaseException, **kwargs: object) -> None:
        captured.append((exc, kwargs))

    monkeypatch.setattr(worker_mod, "capture_exception", capture)
    worker = WeightWorker(
        submission_service=_FailingSubmissionService(
            socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
        )
    )

    for _ in range(4):
        with pytest.raises(socket.gaierror):
            worker._tick()

    assert len(captured) == 1
    exc, kwargs = captured[0]
    assert type(exc) is RuntimeError
    assert str(exc) == "weight worker transient network outage"
    assert kwargs["tags"] == {
        "failure_kind": "retryable_network",
        "worker": "validator-weight-worker",
    }
    assert kwargs["context_name"] == "retryable_network"
    assert kwargs["fingerprint"] == ["validator-weight-worker", "retryable-network"]
    assert kwargs["context"] == {
        "attempts": 3,
        "threshold": 3,
        "cause_type": "gaierror",
        "cause_kind": "temporary_dns",
        "errno": socket.EAI_AGAIN,
    }


def test_weight_worker_groups_websocket_handshake_timeout_at_threshold(monkeypatch) -> None:
    captured: list[tuple[BaseException, dict[str, object]]] = []

    def capture(exc: BaseException, **kwargs: object) -> None:
        captured.append((exc, kwargs))

    monkeypatch.setattr(worker_mod, "capture_exception", capture)
    worker = WeightWorker(
        submission_service=_FailingSubmissionService(TimeoutError("timed out while waiting for handshake response"))
    )

    for _ in range(4):
        with pytest.raises(TimeoutError):
            worker._tick()

    assert len(captured) == 1
    exc, kwargs = captured[0]
    assert type(exc) is RuntimeError
    assert str(exc) == "weight worker transient network outage"
    assert kwargs["tags"] == {
        "failure_kind": "retryable_network",
        "worker": "validator-weight-worker",
    }
    assert kwargs["context_name"] == "retryable_network"
    assert kwargs["fingerprint"] == ["validator-weight-worker", "retryable-network"]
    assert kwargs["context"] == {
        "attempts": 3,
        "threshold": 3,
        "cause_type": "TimeoutError",
        "cause_kind": "websocket_handshake_timeout",
        "errno": None,
    }


def test_weight_worker_counts_consecutive_transient_failures_across_causes(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(worker_mod, "capture_exception", lambda exc, **_: captured.append(exc))
    service = _SequenceSubmissionService(
        [
            socket.gaierror(socket.EAI_AGAIN, "temporary dns"),
            ConnectionError(socket.EAI_AGAIN, "temporary dns"),
            ConnectionResetError(errno.ECONNRESET, "connection reset"),
        ]
    )
    worker = WeightWorker(submission_service=service)

    for expected in (socket.gaierror, ConnectionError, ConnectionResetError):
        with pytest.raises(expected):
            worker._tick()

    assert [str(exc) for exc in captured] == ["weight worker transient network outage"]


def test_weight_worker_resets_transient_network_failure_after_success(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(worker_mod, "capture_exception", captured.append)
    status = StatusProvider()
    service = _SequenceSubmissionService(
        [
            socket.gaierror(socket.EAI_AGAIN, "temporary dns"),
            None,
            socket.gaierror(socket.EAI_AGAIN, "temporary dns"),
            socket.gaierror(socket.EAI_AGAIN, "temporary dns"),
        ]
    )
    worker = WeightWorker(submission_service=service, status_provider=status)

    with pytest.raises(socket.gaierror):
        worker._tick()
    assert status.state.last_weight_error == "weight submission retrying after transient network failure"

    worker._tick()
    assert status.state.last_weight_error is None

    for _ in range(2):
        with pytest.raises(socket.gaierror):
            worker._tick()

    assert captured == []


def test_weight_worker_resets_transient_network_failure_before_non_transient_failure(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(worker_mod, "capture_exception", captured.append)
    status = StatusProvider()
    service = _SequenceSubmissionService(
        [
            socket.gaierror(socket.EAI_AGAIN, "temporary dns"),
            RuntimeError("deterministic failure"),
        ]
    )
    worker = WeightWorker(submission_service=service, status_provider=status)

    with pytest.raises(socket.gaierror):
        worker._tick()
    with pytest.raises(RuntimeError, match="deterministic failure"):
        worker._tick()
    worker._on_error()

    assert [str(exc) for exc in captured] == ["deterministic failure"]
    assert status.state.last_weight_error == "weight submission failed (see logs)"
