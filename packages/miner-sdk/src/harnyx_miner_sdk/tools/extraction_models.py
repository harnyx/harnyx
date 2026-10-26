"""Typed extraction questions and answers used by ``decision_query``."""

from __future__ import annotations

from datetime import date, datetime, time
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator


class _ExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ExtractionQuestion(_ExtractionModel):
    """Cloudflare Clairic field definition, expressed as a named question.

    The field type controls the returned value. Estimates and forecasts predict
    unknown outcomes; other types extract or compute values from the state.
    """

    type: Literal["text", "number", "date", "time", "datetime", "duration", "list", "estimate", "forecast"]
    instructions: str = ""
    unit: str | None = None
    integer: bool | None = None
    decimals: int | None = Field(default=None, ge=0, le=6)
    items: Literal["text", "number", "date"] | None = None
    min: int | float | str | None = None
    max: int | float | str | None = None
    step: int | float | Literal["month", "year"] | None = None
    scale: Literal["linear", "log"] | None = None
    null_threshold: float | None = Field(default=None, gt=0, le=1)
    horizon: int | None = Field(default=None, ge=1, le=64)
    interval: Literal["step", "minute", "hour", "day", "week", "month", "quarter", "year"] | None = None

    @model_validator(mode="after")
    def _field_options(self) -> ExtractionQuestion:
        if self.type == "duration" and self.unit is not None and self.unit not in {
            "seconds", "minutes", "hours", "days", "weeks", "months", "years",
        }:
            raise ValueError("duration unit must be seconds, minutes, hours, days, weeks, months, or years")
        if self.type == "list" and self.items is None:
            raise ValueError("list questions require items")
        if self.items is not None and self.type != "list":
            raise ValueError("items is only supported for list questions")
        if self.integer is not None and self.type not in {"number", "duration"}:
            raise ValueError("integer is only supported for number and duration questions")
        if self.decimals is not None and self.type not in {"number", "duration", "list"}:
            raise ValueError("decimals requires a number, duration, or number list")
        if self.type == "list" and self.decimals is not None and self.items != "number":
            raise ValueError("decimals requires number list items")
        prediction = self.type in {"estimate", "forecast"}
        if prediction and (self.min is None or self.max is None):
            raise ValueError("estimate and forecast questions require min and max")
        if not prediction and any(v is not None for v in (self.min, self.max, self.step, self.scale)):
            raise ValueError("prediction bounds and grid require estimate or forecast")
        if isinstance(self.step, int | float) and self.step <= 0:
            raise ValueError("prediction step must be positive")
        if isinstance(self.min, int | float) and isinstance(self.max, int | float):
            if self.min >= self.max:
                raise ValueError("prediction min must be less than max")
            if isinstance(self.step, str):
                raise ValueError("calendar prediction steps require date bounds")
            if self.scale == "log" and self.min <= 0:
                raise ValueError("log prediction bounds must be positive")
        if prediction:
            if isinstance(self.min, str) and isinstance(self.max, str):
                if date.fromisoformat(self.min) >= date.fromisoformat(self.max):
                    raise ValueError("prediction min date must be before max date")
                if self.scale == "log":
                    raise ValueError("date predictions require a linear scale")
            elif not isinstance(self.min, int | float) or not isinstance(self.max, int | float):
                raise ValueError("prediction min and max must both be numbers or ISO dates")
        if self.type == "forecast" and (self.horizon is None or self.interval is None):
            raise ValueError("forecast questions require horizon and interval")
        if self.type != "forecast" and (self.horizon is not None or self.interval is not None):
            raise ValueError("horizon and interval are only supported for forecasts")
        return self


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ExtractionAnswer(_ExtractionModel):
    """Native Clairic result, including null, source spans and uncertainty.

    ``value`` is the usable result; ``guess`` may remain populated even when
    ``value`` is null. Callers must not substitute the guess for missing data.
    """

    type: Literal["text", "number", "date", "time", "datetime", "duration", "list", "estimate", "forecast"]
    value: JsonValue
    guess: JsonValue = None
    p_null: Probability | None = None
    confidence: Probability | None = None
    span: list[int] | None = Field(default=None, min_length=2, max_length=2)
    spans: list[list[int]] | None = None
    unit: str | None = None
    mean: int | float | str | list[int | float | str] | None = None
    median: int | float | str | list[int | float | str] | None = None
    p10: int | float | str | list[int | float | str] | None = None
    p90: int | float | str | list[int | float | str] | None = None
    p_below: Probability | None = None
    p_above: Probability | None = None
    quantiles: list[JsonValue] | None = None
    steps: list[dict[str, JsonValue]] | None = None


class _ForecastStep(ExtractionAnswer):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)
    type: Literal["estimate"] = "estimate"
    value: int | float | str


_FORECAST_STEP = TypeAdapter(_ForecastStep)


def validate_extraction_answer(question: ExtractionQuestion, answer: ExtractionAnswer, *, state: str) -> None:
    if question.type != answer.type:
        raise ValueError("extraction answer type must match question type")
    if question.type == "forecast":
        _validate_forecast_answer(question, answer, state=state)
    elif question.type == "estimate":
        summaries = {
            name: _validate_estimate_value(question, summary)
            for name, summary in (
                ("value", answer.value), ("mean", answer.mean), ("median", answer.median),
                ("p10", answer.p10), ("p90", answer.p90),
            ) if summary is not None
        }
        if not summaries:
            raise ValueError("estimate answer must contain a prediction")
        percentiles = [summaries[name] for name in ("p10", "median", "p90") if name in summaries]
        if any(left > right for left, right in pairwise(percentiles)):
            raise ValueError("estimate percentiles must be ordered p10 <= median <= p90")
    value = answer.value
    if question.type == "list":
        if not isinstance(value, list):
            raise ValueError("list answer must contain a list")
        for item in value:
            if question.items == "number":
                _validate_extracted_number(question, item)
            elif not isinstance(item, str):
                raise ValueError("text and date list answers must contain strings")
            elif question.items == "date":
                date.fromisoformat(item)
    if question.type == "duration" and answer.unit is not None and answer.unit != (question.unit or "seconds"):
        raise ValueError("duration answer unit must match the requested unit")
    if question.type not in {"list", "estimate", "forecast"} and answer.p_null is not None:
        threshold = question.null_threshold if question.null_threshold is not None else 0.5
        if answer.p_null > threshold and value is not None:
            raise ValueError("extraction value must be null above the requested null threshold")
    if value is not None:
        if question.type in {"text", "date", "time", "datetime"} and not isinstance(value, str):
            raise ValueError("extraction answer must contain a string")
        if question.type in {"number", "duration"}:
            _validate_extracted_number(question, value)
        if question.type in {"date", "time", "datetime"} and isinstance(value, str):
            {"date": date.fromisoformat, "time": time.fromisoformat, "datetime": datetime.fromisoformat}[
                question.type
            ](value)
    for span in ([answer.span] if answer.span is not None else []) + (answer.spans or []):
        if len(span) != 2 or not 0 <= span[0] <= span[1] <= len(state):
            raise ValueError("extraction source span is outside the input state")
    if question.type == "text" and isinstance(value, str) and answer.span is not None:
        start, end = answer.span
        if state[start:end] != value:
            raise ValueError("text extraction value must match its source span")


def _validate_forecast_answer(question: ExtractionQuestion, answer: ExtractionAnswer, *, state: str) -> None:
    assert question.horizon is not None  # Guaranteed by the forecast question validator.
    estimate_question = ExtractionQuestion.model_validate({
        **question.model_dump(exclude={"horizon", "interval"}, exclude_none=True), "type": "estimate",
    })
    if answer.value is None and answer.steps is None:
        raise ValueError("forecast answer requires values or complete steps for the requested horizon")
    arrays: dict[str, list[JsonValue]] = {}
    for name, values in (
        ("value", answer.value), ("mean", answer.mean), ("median", answer.median),
        ("p10", answer.p10), ("p90", answer.p90),
    ):
        if values is None:
            continue
        if not isinstance(values, list) or len(values) != question.horizon:
            raise ValueError("forecast answer arrays must match the requested horizon")
        if name == "value":
            for value in values:
                _validate_estimate_value(estimate_question, value)
        arrays[name] = list(values)
    if answer.steps is not None and len(answer.steps) != question.horizon:
        raise ValueError("forecast answer steps must match the requested horizon")
    for index in range(question.horizon):
        step = (_FORECAST_STEP.validate_python(answer.steps[index]) if answer.steps is not None
                else ExtractionAnswer(type="estimate", value=None))
        components: dict[str, JsonValue] = {}
        for name, values in arrays.items():
            array_value = values[index]
            step_value = getattr(step, name)
            if array_value is not None and step_value is not None and (
                _validate_estimate_value(estimate_question, array_value)
                != _validate_estimate_value(estimate_question, step_value)
            ):
                raise ValueError("forecast arrays and steps must agree on corresponding values and summaries")
            if array_value is not None:
                components[name] = array_value
        validate_extraction_answer(estimate_question, step.model_copy(update=components), state=state)


def _validate_extracted_number(question: ExtractionQuestion, value: JsonValue) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("extraction answer must contain a number")
    if question.integer and value != int(value):
        raise ValueError("integer extraction answer must be integral")
    if question.decimals is not None and value != round(value, question.decimals):
        raise ValueError("extraction answer exceeds the requested decimal precision")


def _validate_estimate_value(question: ExtractionQuestion, value: JsonValue) -> int | float:
    if isinstance(question.min, str):
        if not isinstance(value, str) or not isinstance(question.max, str):
            raise ValueError("date estimate must return an ISO date")
        parsed = date.fromisoformat(value)
        if not date.fromisoformat(question.min) <= parsed <= date.fromisoformat(question.max):
            raise ValueError("estimate value is outside requested bounds")
        return parsed.toordinal()
    else:
        if (
            isinstance(value, bool) or not isinstance(value, int | float)
            or not isinstance(question.min, int | float) or not isinstance(question.max, int | float)
        ):
            raise ValueError("numeric estimate must return a number")
        if not question.min <= value <= question.max:
            raise ValueError("estimate value is outside requested bounds")
        return value
