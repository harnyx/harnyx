from __future__ import annotations

from uuid import uuid4

import pytest

from harnyx_commons.domain.claim import FeedSearchContext, GeneratedClaim, Rubric, Span
from harnyx_commons.domain.verdict import VerdictOption, VerdictOptions

_BINARY_VERDICT_OPTIONS = VerdictOptions(
    options=(
        VerdictOption(value=-1, description="Fail"),
        VerdictOption(value=1, description="Pass"),
    )
)


def _rubric() -> Rubric:
    return Rubric(
        title="Accuracy",
        description="Check facts.",
        verdict_options=_BINARY_VERDICT_OPTIONS,
    )


def test_feed_search_context_accepts_positive_enqueue_seq() -> None:
    context = FeedSearchContext(feed_id=uuid4(), enqueue_seq=4)

    assert context.enqueue_seq == 4


def test_feed_search_context_rejects_negative_enqueue_seq() -> None:
    with pytest.raises(ValueError):
        FeedSearchContext(feed_id=uuid4(), enqueue_seq=-1)


def test_generated_claim_validates_verdict_against_rubric() -> None:
    claim = GeneratedClaim(
        claim_id=uuid4(),
        text="example claim",
        rubric=_rubric(),
        verdict=1,
        justification="supported",
    )

    assert claim.verdict == 1


def test_generated_claim_rejects_unknown_verdict() -> None:
    with pytest.raises(ValueError):
        GeneratedClaim(
            claim_id=uuid4(),
            text="example claim",
            rubric=_rubric(),
            verdict=2,
            justification="unsupported",
        )

def test_span_rejects_end_before_start() -> None:
    with pytest.raises(ValueError):
        Span(excerpt="bad", start=3, end=2)
