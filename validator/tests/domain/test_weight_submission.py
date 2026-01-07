from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from caster_validator.domain.weight import WeightSubmission


def test_weight_submission_accepts_normalized_vector() -> None:
    submission = WeightSubmission(
        run_id=uuid4(),
        submitted_at=datetime(2025, 10, 16, tzinfo=UTC),
        weights={1: 0.6, 2: 0.4},
        tx_hash="0xabc",
    )

    assert pytest.approx(submission.total_weight, rel=1e-6) == 1.0


def test_weight_submission_rejects_unbalanced_weights() -> None:
    with pytest.raises(ValueError):
        WeightSubmission(
            run_id=uuid4(),
            submitted_at=datetime(2025, 10, 16, tzinfo=UTC),
            weights={1: 0.5, 2: 0.3},
            tx_hash="0xabc",
        )
