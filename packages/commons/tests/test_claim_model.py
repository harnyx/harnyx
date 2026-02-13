from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from caster_commons.domain.claim import FeedSearchContext, MinerTaskClaim, ReferenceAnswer, Rubric
from caster_commons.domain.verdict import VerdictOption, VerdictOptions

_CLAIM_ADAPTER = TypeAdapter(MinerTaskClaim)
_BINARY_VERDICT_OPTIONS = VerdictOptions(
    options=(
        VerdictOption(value=-1, description="Fail"),
        VerdictOption(value=1, description="Pass"),
    )
)


def _claim_kwargs() -> dict[str, object]:
    return {
        "claim_id": uuid4(),
        "text": "example claim",
        "rubric": Rubric(
            title="Accuracy",
            description="Check facts.",
            verdict_options=_BINARY_VERDICT_OPTIONS,
        ),
        "reference_answer": ReferenceAnswer(verdict=1, justification="reference", citations=()),
        "budget_usd": 0.1,
    }


def test_claim_accepts_context_object() -> None:
    context = FeedSearchContext(feed_id=uuid4(), enqueue_seq=4)

    claim = MinerTaskClaim(**_claim_kwargs(), context=context)

    assert claim.context == context


def test_claim_rejects_context_with_wrong_type() -> None:
    with pytest.raises(TypeError, match="claim context must be FeedSearchContext"):
        MinerTaskClaim(**_claim_kwargs(), context={"feed_id": str(uuid4()), "enqueue_seq": 1})


def test_claim_adapter_parses_context_from_dict() -> None:
    raw = {
        "claim_id": str(uuid4()),
        "text": "example claim",
        "rubric": {
            "title": "Accuracy",
            "description": "Check facts.",
            "verdict_options": {"options": [{"value": -1, "description": "Fail"}, {"value": 1, "description": "Pass"}]},
        },
        "reference_answer": {"verdict": 1, "justification": "reference", "citations": []},
        "budget_usd": 0.1,
        "context": {"feed_id": str(uuid4()), "enqueue_seq": 3},
    }

    claim = _CLAIM_ADAPTER.validate_python(raw)

    assert claim.context is not None
    assert claim.context.enqueue_seq == 3


def test_claim_adapter_rejects_negative_context_enqueue_seq() -> None:
    raw = {
        "claim_id": str(uuid4()),
        "text": "example claim",
        "rubric": {
            "title": "Accuracy",
            "description": "Check facts.",
            "verdict_options": {"options": [{"value": -1, "description": "Fail"}, {"value": 1, "description": "Pass"}]},
        },
        "reference_answer": {"verdict": 1, "justification": "reference", "citations": []},
        "budget_usd": 0.1,
        "context": {"feed_id": str(uuid4()), "enqueue_seq": -1},
    }

    with pytest.raises(ValidationError):
        _CLAIM_ADAPTER.validate_python(raw)
