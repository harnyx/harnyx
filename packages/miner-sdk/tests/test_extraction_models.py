"""Reject unsupported model/question combinations before paid provider dispatch."""

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.tools.decision_models import (
    DecisionQueryRequest,
    DecisionQueryResponse,
    validate_decision_answers,
)


def _request(question: dict, *, model: str = "@cf/cloudflare/clairic") -> dict:
    return {"provider": "cloudflare", "model": model, "state": "Total 100", "questions": {"amount": question}}


@pytest.mark.parametrize("payload", [
    _request({"type": "number"}, model="@cf/cloudflare/clef"),
    _request({"type": "boolean", "instructions": "Paid?"}),
    _request({"type": "list"}),
    _request({"type": "estimate", "min": 20, "max": 10}),
    _request({"type": "forecast", "min": 0, "max": 10}),
    _request({"type": "number", "integer": True, "decimals": 8}),
    _request({"type": "duration", "unit": "fortnights"}),
    _request({"type": "duration", "unit": ""}),
])
def test_invalid_extraction_requests_fail_before_invocation(payload: dict) -> None:
    with pytest.raises(ValidationError):
        DecisionQueryRequest.model_validate(payload)


@pytest.mark.parametrize("kind", ["estimate", "forecast"])
@pytest.mark.parametrize("bounds,step,valid", [
    ((0, 10), "month", False), ((0, 10), "year", False), ((0, 10), 2, True),
    (("2026-01-01", "2027-01-01"), "month", True),
    (("2026-01-01", "2027-01-01"), "year", True),
    (("2026-01-01", "2027-01-01"), 2, True),
])
def test_calendar_prediction_steps_require_date_bounds(kind, bounds, step, valid) -> None:
    payload = _request({
        "type": kind, "min": bounds[0], "max": bounds[1], "step": step,
        **({"horizon": 2, "interval": "day"} if kind == "forecast" else {}),
    })
    if valid:
        DecisionQueryRequest.model_validate(payload)
    else:
        with pytest.raises(ValidationError, match="calendar prediction steps require date bounds"):
            DecisionQueryRequest.model_validate(payload)


@pytest.mark.parametrize("answer", [
    {"type": "number", "value": True},
    {"type": "number", "value": float("nan")},
    {"type": "number", "value": 100, "span": [0, 100]},
    {"type": "text", "value": "100"},
])
def test_invalid_answer_values_and_source_spans_are_rejected(answer: dict) -> None:
    request = DecisionQueryRequest.model_validate(_request({"type": "number"}))
    with pytest.raises(ValueError):
        response = DecisionQueryResponse.model_validate({"model": request.model, "answers": {"amount": answer}})
        validate_decision_answers(request, response)


@pytest.mark.parametrize("steps", [None, [], [{}, {}], [{"value": -1}, {"value": 4}], [{"value": True}, {"value": 4}]])
def test_forecast_requires_one_valid_bounded_estimate_per_step_even_with_null_value(steps: list | None) -> None:
    request = DecisionQueryRequest.model_validate(_request({
        "type": "forecast", "min": 0, "max": 10, "horizon": 2, "interval": "day",
    }))
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {"type": "forecast", "value": None, "steps": steps}},
    })
    with pytest.raises(ValueError):
        validate_decision_answers(request, response)


@pytest.mark.parametrize("values,steps,valid", [
    (None, None, False), (None, [{"value": 2}, {"value": 4}], True),
    ([2, 4], None, True), ([None, 4], None, False),
    ([None, 4], [{"value": 2}, {"value": 4}], False),
])
def test_forecasts_require_usable_values_despite_optional_summaries(values, steps, valid) -> None:
    request = DecisionQueryRequest.model_validate(_request({
        "type": "forecast", "min": 0, "max": 10, "horizon": 2, "interval": "day",
    }))
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {
            "type": "forecast", "value": values, "mean": [2.5, 4.5], "steps": steps,
        }},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError):
            validate_decision_answers(request, response)


@pytest.mark.parametrize("items", ["text", "number", "date"])
@pytest.mark.parametrize("value,valid", [(None, False), ([], True)])
def test_missing_list_matches_require_an_empty_list(items, value, valid) -> None:
    request = DecisionQueryRequest.model_validate(_request({"type": "list", "items": items}))
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {"type": "list", "value": value}},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="list answer must contain a list"):
            validate_decision_answers(request, response)


@pytest.mark.parametrize("answer,valid", [
    ({"type": "estimate", "value": None}, False),
    ({"type": "estimate", "value": None, "guess": 5}, False),
    ({"type": "estimate", "value": 5}, True),
    ({"type": "estimate", "value": None, "mean": 5}, True),
])
def test_standalone_estimates_require_a_prediction_without_using_the_guess(answer, valid) -> None:
    request = DecisionQueryRequest.model_validate(_request({"type": "estimate", "min": 0, "max": 10}))
    response = DecisionQueryResponse.model_validate({"model": request.model, "answers": {"amount": answer}})
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="estimate answer must contain a prediction"):
            validate_decision_answers(request, response)


@pytest.mark.parametrize("bounds,arrays,step,valid", [
    ((0, 10), {"value": [1]}, {"value": 9}, False),
    ((0, 10), {"value": [1], "median": [2]}, {"value": 1, "median": 9}, False),
    ((0, 10), {"value": [1], "p10": [9]}, {"value": 1, "p90": 2}, False),
    ((0, 10), {"value": [1], "p10": [2]}, {"value": 1, "p90": 9}, True),
    ((0, 10), {"value": [1], "mean": [2.5]}, {"value": 1.0, "mean": 2.50}, True),
    ((0, 10), {"value": [1], "median": [2]}, {"value": 1}, True),
    (("20260101", "20261231"), {"value": ["20260601"], "median": ["2026-07-01"]},
     {"value": "2026-06-01", "median": "20260701"}, True),
    (("20260101", "20261231"), {"value": ["20260601"]}, {"value": "2026-06-02"}, False),
    (("20260101", "20261231"), {"value": ["20260601"], "p10": ["2026-10-01"]},
     {"value": "20260601", "p90": "2026-02-01"}, False),
])
def test_forecast_representations_describe_one_coherent_estimate(bounds, arrays, step, valid) -> None:
    request = DecisionQueryRequest.model_validate(_request({
        "type": "forecast", "min": bounds[0], "max": bounds[1], "horizon": 1, "interval": "day",
    }))
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {"type": "forecast", **arrays, "steps": [step]}},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError):
            validate_decision_answers(request, response)


@pytest.mark.parametrize("component", ["value", "mean", "median", "p10", "p90"])
@pytest.mark.parametrize("difference,valid", [(0, True), (1, False)])
def test_forecast_reconciliation_preserves_adjacent_large_integers(component, difference, valid) -> None:
    point = 9007199254740993
    request = DecisionQueryRequest.model_validate(_request({
        "type": "forecast", "min": 0, "max": 10**16, "horizon": 1, "interval": "day",
    }))
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {
            "type": "forecast", "value": [point], component: [point],
            "steps": [{"value": point, component: point - difference}],
        }},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="forecast arrays and steps must agree"):
            validate_decision_answers(request, response)


def test_prediction_bounds_and_grid_preserve_large_integers() -> None:
    lower = 9007199254740993
    request = DecisionQueryRequest.model_validate(_request({
        "type": "estimate", "min": lower, "max": lower + 4, "step": lower + 2,
    }))
    assert request.questions["amount"].min == lower
    assert request.questions["amount"].max == lower + 4
    assert request.questions["amount"].step == lower + 2
    for point, valid in ((lower, True), (lower - 1, False), (lower + 5, False)):
        response = DecisionQueryResponse.model_validate({
            "model": request.model, "answers": {"amount": {"type": "estimate", "value": point}},
        })
        if valid:
            validate_decision_answers(request, response)
        else:
            with pytest.raises(ValueError, match="outside requested bounds"):
                validate_decision_answers(request, response)


def test_documented_duration_and_forecast_answers_round_trip() -> None:
    """Native unit metadata and parallel forecast arrays must survive SDK parsing."""
    request = DecisionQueryRequest.model_validate({
        "provider": "cloudflare", "model": "@cf/cloudflare/clairic", "state": "Delivered 11 days ago",
        "questions": {
            "elapsed": {"type": "duration", "unit": "days"},
            "travelers": {"type": "forecast", "min": 0, "max": 3000000, "horizon": 1, "interval": "day"},
        },
    })
    answers = {
        "elapsed": {"type": "duration", "value": 11, "unit": "days", "guess": 11, "confidence": 0.999999},
        "travelers": {"type": "forecast", "value": [2416117], "median": [2419283],
                      "p10": [2270774], "p90": [2556769]},
    }
    response = DecisionQueryResponse.model_validate({"model": request.model, "answers": answers})
    validate_decision_answers(request, response)
    assert response.model_dump(mode="json", exclude_none=True)["answers"] == answers


@pytest.mark.parametrize("field,values", [
    ("value", [1]), ("value", [1, 11]), ("value", [True, 4]),
    ("median", [1]), ("p10", [-1, 4]), ("p90", [4, 11]),
])
def test_native_forecast_arrays_validate_horizon_types_and_bounds(field: str, values: list) -> None:
    request = DecisionQueryRequest.model_validate(_request({
        "type": "forecast", "min": 0, "max": 10, "horizon": 2, "interval": "day",
    }))
    with pytest.raises(ValueError):
        response = DecisionQueryResponse.model_validate({
            "model": request.model, "answers": {"amount": {"type": "forecast", "value": [2, 4], field: values}},
        })
        validate_decision_answers(request, response)


def test_empty_extraction_state_fails_before_paid_dispatch() -> None:
    with pytest.raises(ValidationError, match="requires non-empty state"):
        DecisionQueryRequest.model_validate({**_request({"type": "number"}), "state": ""})


@pytest.mark.parametrize("field", ["mean", "median", "p10", "p90"])
@pytest.mark.parametrize("bounds,summary,valid", [
    ((0, 10), 100, False), ((0, 10), -1, False), ((0, 10), "5", False),
    ((0, 10), [5], False), ((0, 10), 5, True),
    (("2026-01-01", "2026-12-31"), "2027-01-01", False),
    (("2026-01-01", "2026-12-31"), "not-a-date", False),
    (("2026-01-01", "2026-12-31"), 5, False),
    (("2026-01-01", "2026-12-31"), "2026-06-01", True),
    (("20260101", "2026-12-31"), "2026-06-01", True),
    (("2026-01-01", "20261231"), "20260601", True),
    (("20260101", "20261231"), "2027-01-01", False),
])
def test_estimate_uncertainty_uses_value_type_and_bounds_even_when_value_is_null(
    field: str, bounds: tuple, summary, valid: bool,
) -> None:
    request = DecisionQueryRequest.model_validate(_request({"type": "estimate", "min": bounds[0], "max": bounds[1]}))
    payload = {"model": request.model, "answers": {"amount": {"type": "estimate", "value": None, field: summary}}}
    if valid:
        validate_decision_answers(request, DecisionQueryResponse.model_validate(payload))
    else:
        with pytest.raises(ValueError):
            validate_decision_answers(request, DecisionQueryResponse.model_validate(payload))


def test_numeric_extraction_preserves_freeform_unit_labels() -> None:
    """Duration vocabulary must not restrict labels on other extraction types."""
    request = DecisionQueryRequest.model_validate(_request({"type": "number", "unit": "USD"}))
    assert request.questions["amount"].unit == "USD"


@pytest.mark.parametrize("state,value,span,valid", [
    ("Invoice", "wrong", [0, 7], False),
    ("Invoice", "Invoice", [0, 7], True),
    ("😀 Café. Invoice", "Café", [2, 6], True),
    ("😀 Café. Invoice", "Café", [3, 7], False),
    ("Invoice", None, [0, 7], True),
    ("Invoice", "Invoice", None, True),
])
def test_text_source_spans_identify_exact_text_with_unicode_character_offsets(
    state: str, value: str | None, span: list[int] | None, valid: bool,
) -> None:
    request = DecisionQueryRequest.model_validate({**_request({"type": "text"}), "state": state})
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {"type": "text", "value": value, "span": span}},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="match its source span"):
            validate_decision_answers(request, response)


@pytest.mark.parametrize("requested,returned,valid", [
    ("days", "hours", False), ("days", "days", True), ("days", None, True),
    (None, "seconds", True), (None, "days", False),
])
def test_supplied_duration_units_agree_with_requested_or_default_unit(requested, returned, valid: bool) -> None:
    question = {"type": "duration", **({"unit": requested} if requested is not None else {})}
    request = DecisionQueryRequest.model_validate(_request(question))
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {"type": "duration", "value": 24, "unit": returned}},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="unit must match"):
            validate_decision_answers(request, response)


@pytest.mark.parametrize("threshold,p_null,value,valid", [
    (None, 0.9, 100, False), (0.2, 0.3, 100, False), (0.8, 0.7, 100, True),
    (0.5, 0.5, 100, True), (0.5, 0.1, None, True), (0.2, None, 100, True),
    (1.0, 1.0, 100, True), (1.0, 1.0, None, True),
])
def test_supplied_null_probability_requires_null_only_above_the_threshold(
    threshold, p_null, value, valid: bool,
) -> None:
    question = {"type": "number", **({"null_threshold": threshold} if threshold is not None else {})}
    request = DecisionQueryRequest.model_validate(_request(question))
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {"type": "number", "value": value, "p_null": p_null}},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="null above"):
            validate_decision_answers(request, response)


@pytest.mark.parametrize("kind,value,precision,valid", [
    ("number", 1.234, 2, False), ("number", 1.23, 2, True), ("number", -1.2, 2, True),
    ("number", 1.5, 0, False), ("number", 1e-7, 6, False), ("number", 1e-6, 6, True),
    ("number", 1.234, None, True), ("duration", 24.125, 2, False), ("duration", 24.12, 2, True),
    ("list", [1.2, 1.234], 2, False), ("list", [1.2, 1.23], 2, True),
])
def test_extracted_numbers_honor_only_requested_precision(kind: str, value, precision, valid: bool) -> None:
    question = {"type": kind, **({"items": "number"} if kind == "list" else {}),
                **({"decimals": precision} if precision is not None else {})}
    request = DecisionQueryRequest.model_validate(_request(question))
    response = DecisionQueryResponse.model_validate({
        "model": request.model, "answers": {"amount": {"type": kind, "value": value}},
    })
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="decimal precision"):
            validate_decision_answers(request, response)


@pytest.mark.parametrize("representation", ["estimate", "arrays", "steps"])
@pytest.mark.parametrize("bounds,lower,median,upper,valid", [
    ((0, 10), 1, 5, 9, True), ((0, 10), 2, 4, 6, True), ((0, 10), 9, 5, 1, False),
    ((0, 10), 6, 5, None, False), ((0, 10), 1, None, 9, True), ((0, 10), 9, None, 1, False),
    (("20260101", "20261231"), "2026-02-01", "20260601", "2026-10-01", True),
    (("20260101", "20261231"), "2026-10-01", "20260601", "2026-02-01", False),
])
def test_prediction_percentiles_are_ordered_in_every_representation(
    representation: str, bounds: tuple, lower, median, upper, valid: bool,
) -> None:
    """Percentiles must be coherent without restricting the mean to their interval or support grid."""
    kind = "estimate" if representation == "estimate" else "forecast"
    question = {"type": kind, "min": bounds[0], "max": bounds[1],
                **({"step": 2} if isinstance(bounds[0], int) else {}),
                **({"horizon": 1, "interval": "day"} if kind == "forecast" else {})}
    request = DecisionQueryRequest.model_validate(_request(question))
    summaries = {key: value for key, value in {
        "value": bounds[0], "mean": 7.5 if isinstance(bounds[0], int) else bounds[1],
        "p10": lower, "median": median, "p90": upper,
    }.items() if value is not None}
    answer = ({"type": kind, **summaries} if kind == "estimate" else
              {"type": kind, "value": None, "steps": [summaries]} if representation == "steps" else
              {"type": kind, **{key: [value] for key, value in summaries.items()}})
    response = DecisionQueryResponse.model_validate({"model": request.model, "answers": {"amount": answer}})
    if valid:
        validate_decision_answers(request, response)
    else:
        with pytest.raises(ValueError, match="percentiles must be ordered"):
            validate_decision_answers(request, response)
