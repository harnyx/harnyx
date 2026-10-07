from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.tools.decision_models import (
    DecisionQueryRequest,
    DecisionQueryResponse,
    validate_decision_answers,
)


def _request() -> dict:
    return {
        "provider": "openrouter",
        "model": "cloudflare/clef-flash",
        "state": {"text": " unchanged "},
        "questions": {
            "b": {"type": "boolean", "instructions": "Relevant?"},
            "c": {"type": "choice", "instructions": {"text": "Pick"}, "criteria": {"a": None, "b": "B"}},
            "s": {"type": "score", "instructions": "Rate", "criteria": ["Low", "High"]},
        },
    }


def _response() -> dict:
    return {
        "model": "cloudflare/clef-flash-versioned",
        "answers": {
            "b": {"type": "boolean", "probability": 0.8},
            "c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.66, "b": 0.33}},
            "s": {"type": "score", "score": 0.8},
        },
    }


def test_structured_context_rounded_and_absent_distributions_are_preserved() -> None:
    request = DecisionQueryRequest.model_validate(_request())
    response = DecisionQueryResponse.model_validate(_response())
    validate_decision_answers(request, response)
    assert request.state == {"text": " unchanged "}
    assert response.model == "cloudflare/clef-flash-versioned"
    assert response.model_dump()["answers"]["c"]["probabilities"] == {"a": 0.66, "b": 0.33}


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "ai_gateway"},
        {"model": "unapproved"},
        {"questions": {}},
        {"timeout": -1.0},
        {"state": {"nested": [float("nan")]}},
        {"temperature": 0.2},
        {"questions": {"b": {"type": "noul", "instructions": "?"}}},
        {"questions": {"b": {"type": "boolean", "instructions": "?", "criteria": {"true": "yes"}}}},
        {"questions": {"s": {"type": "score", "instructions": "?", "criteria": []}}},
    ],
)
def test_invalid_requests_fail_before_paid_dispatch(change: dict) -> None:
    with pytest.raises(ValidationError):
        DecisionQueryRequest.model_validate({**_request(), **change})


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "boolean", "probability": float("inf")},
        {"type": "boolean", "probability": 1.1},
        {"type": "choice", "choice": "unrequested"},
        {"type": "score", "score": 2.0},
        {"type": "choice", "choice": "a", "probabilities": {"a": 0.8}},
    ],
)
def test_invalid_answers_do_not_escape_validation(answer: dict) -> None:
    raw = copy.deepcopy(_response())
    key = {"boolean": "b", "choice": "c", "score": "s"}[answer["type"]]
    raw["answers"][key] = answer
    with pytest.raises((ValueError, ValidationError)):
        validate_decision_answers(
            DecisionQueryRequest.model_validate(_request()), DecisionQueryResponse.model_validate(raw)
        )


def test_missing_answer_and_changed_type_are_rejected() -> None:
    request = DecisionQueryRequest.model_validate(_request())
    for answers in ({}, {**_response()["answers"], "b": {"type": "score", "score": 0.0}}):
        with pytest.raises(ValueError):
            validate_decision_answers(request, DecisionQueryResponse.model_validate({"model": "m", "answers": answers}))
