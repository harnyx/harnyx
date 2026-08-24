from __future__ import annotations

from uuid import uuid4

import pytest

from harnyx_commons.domain.claim import GeneratedClaim, Rubric
from harnyx_commons.domain.verdict import VerdictOption, VerdictOptions


def test_generated_claim_rejects_unknown_verdict() -> None:
    rubric = Rubric(
        title="Accuracy",
        description="Check facts.",
        verdict_options=VerdictOptions(
            options=(
                VerdictOption(value=-1, description="Fail"),
                VerdictOption(value=1, description="Pass"),
            )
        ),
    )

    with pytest.raises(ValueError):
        GeneratedClaim(
            claim_id=uuid4(),
            text="example claim",
            rubric=rubric,
            verdict=2,
            justification="unsupported",
        )
