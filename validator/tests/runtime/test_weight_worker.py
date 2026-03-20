from __future__ import annotations

import pytest

import harnyx_validator.runtime.weight_worker as worker_mod
from harnyx_validator.runtime.weight_worker import WeightWorker


class _FailingSubmissionService:
    def try_submit(self) -> None:
        raise RuntimeError("boom")


def test_weight_worker_captures_exception_before_reraising(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(worker_mod, "capture_exception", captured.append)

    worker = WeightWorker(submission_service=_FailingSubmissionService())

    with pytest.raises(RuntimeError, match="boom"):
        worker._tick()

    assert [str(exc) for exc in captured] == ["boom"]
