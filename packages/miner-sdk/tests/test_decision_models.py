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


@pytest.mark.parametrize("model", ["new-vendor/new-decision", "vendor/decision:nitro", "vendor/decision-20261021"])
def test_native_openrouter_models_do_not_need_a_release_to_be_selected(model: str) -> None:
    assert DecisionQueryRequest.model_validate({**_request(), "model": model}).model == model


@pytest.mark.parametrize(
    "model", ["openrouter/free", "vendor/model:free", "vendor/model:free:nitro", "vendor/model?x=y"]
)
def test_free_routes_and_non_native_ids_are_rejected(model: str) -> None:
    with pytest.raises(ValidationError):
        DecisionQueryRequest.model_validate({**_request(), "model": model})


@pytest.mark.parametrize("criteria,legend,valid", [
    (["Low", "High"], {"1": "High", "0": "Low"}, True),
    (["Low", "High"], {"0": "High", "1": "Low"}, False),
    (["Low", "High"], {"0": "Low", "1": "Changed"}, False),
    ([{"text": "Low"}, ["High"]], {"0": {"text": "Low"}, "1": ["High"]}, True),
    ([{"text": "Low"}, ["High"]], {"0": {"text": "Changed"}, "1": ["High"]}, False),
    ([{"paid": True}, {"paid": False}], {"0": {"paid": 1}, "1": {"paid": 0}}, False),
    ([[True], [False]], {"0": [1], "1": [0]}, False),
    ([{"count": 1, "paid": False}, [2]], {"0": {"paid": False, "count": 1.0}, "1": [2.0]}, True),
    (["Low", "High"], None, True),
])
def test_native_score_legends_preserve_requested_criteria(criteria, legend, valid) -> None:
    request = DecisionQueryRequest.model_validate({
        "provider": "cloudflare", "model": "@cf/cloudflare/clef-flash", "state": "Document",
        "questions": {"s": {"type": "score", "instructions": "Rate", "criteria": criteria}},
    })
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"s": {"type": "score", "score": 0.5, "legend": legend}},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="preserve requested score criteria"):
            validate_decision_answers(request, response)
