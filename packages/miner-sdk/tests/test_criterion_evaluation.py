from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from caster_miner_sdk.criterion_evaluation import CriterionEvaluationRequest


def _payload() -> dict[str, object]:
    return {
        "claim_text": "Claim text",
        "rubric_title": "Accuracy",
        "rubric_description": "Judge factual accuracy.",
        "verdict_options": [
            {"value": -1, "description": "Fail"},
            {"value": 1, "description": "Pass"},
        ],
    }


def test_request_accepts_no_context() -> None:
    request = CriterionEvaluationRequest.model_validate(_payload())

    assert request.context is None


def test_request_accepts_context_with_required_fields() -> None:
    payload = _payload()
    payload["context"] = {"feed_id": str(uuid4()), "enqueue_seq": 9}

    request = CriterionEvaluationRequest.model_validate(payload)

    assert request.context is not None
    assert request.context.enqueue_seq == 9


def test_request_rejects_partial_context() -> None:
    payload = _payload()
    payload["context"] = {"feed_id": str(uuid4())}

    with pytest.raises(ValidationError):
        CriterionEvaluationRequest.model_validate(payload)


def test_request_rejects_negative_enqueue_seq() -> None:
    payload = _payload()
    payload["context"] = {"feed_id": str(uuid4()), "enqueue_seq": -1}

    with pytest.raises(ValidationError):
        CriterionEvaluationRequest.model_validate(payload)
