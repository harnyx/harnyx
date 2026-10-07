"""Canonical miner-facing decision query contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from harnyx_miner_sdk.tools.types import ToolInvocationTimeout

DecisionProviderName = Literal["openrouter", "ai_gateway"]
DecisionContext = str | dict[str, JsonValue] | list[JsonValue]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
MINER_SELECTED_DECISION_PROVIDER_MODELS: Mapping[DecisionProviderName, tuple[str, ...]] = {
    "openrouter": ("cloudflare/clef-flash", "cloudflare/clef", "jaredpalmer/kev-4b"),
    "ai_gateway": (),
}


def parse_decision_provider(provider: str) -> DecisionProviderName:
    if provider == "openrouter":
        return "openrouter"
    if provider == "ai_gateway":
        return "ai_gateway"
    raise ValueError(f"unsupported decision provider {provider!r}")


def validate_decision_provider_model(provider: DecisionProviderName, model: str) -> None:
    if model not in MINER_SELECTED_DECISION_PROVIDER_MODELS[provider]:
        raise ValueError(f"decision model {model!r} is not available through {provider!r}")


def _validate_finite_json(value: JsonValue) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("decision context must contain finite JSON values")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_json(item)


class _DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _Question(_DecisionModel):
    instructions: DecisionContext

    @field_validator("instructions")
    @classmethod
    def _finite_instructions(cls, value: DecisionContext) -> DecisionContext:
        _validate_finite_json(value)
        return value


class ChoiceQuestion(_Question):
    type: Literal["choice"]
    criteria: dict[str, DecisionContext | None] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, value: dict[str, DecisionContext | None]) -> dict[str, DecisionContext | None]:
        if any(not key for key in value):
            raise ValueError("choice option IDs must not be empty")
        for item in value.values():
            _validate_finite_json(item)
        return value


class ScoreQuestion(_Question):
    type: Literal["score"]
    criteria: list[DecisionContext] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, value: list[DecisionContext]) -> list[DecisionContext]:
        for item in value:
            _validate_finite_json(item)
        return value


class BooleanCriteria(_DecisionModel):
    true: DecisionContext
    false: DecisionContext

    @field_validator("true", "false")
    @classmethod
    def _criteria(cls, value: DecisionContext) -> DecisionContext:
        _validate_finite_json(value)
        return value


class BooleanQuestion(_Question):
    type: Literal["boolean"]
    criteria: BooleanCriteria | None = None


DecisionQuestion = Annotated[ChoiceQuestion | ScoreQuestion | BooleanQuestion, Field(discriminator="type")]


class DecisionQueryRequest(_DecisionModel):
    provider: DecisionProviderName
    model: str = Field(min_length=1)
    state: DecisionContext
    questions: dict[str, DecisionQuestion] = Field(min_length=1)
    timeout: ToolInvocationTimeout | None = None

    @field_validator("state")
    @classmethod
    def _state(cls, value: DecisionContext) -> DecisionContext:
        _validate_finite_json(value)
        return value

    @model_validator(mode="after")
    def _selection(self) -> DecisionQueryRequest:
        validate_decision_provider_model(self.provider, self.model)
        if any(not key for key in self.questions):
            raise ValueError("question IDs must not be empty")
        return self


class ChoiceAnswer(_DecisionModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability] | None = None
    confidence: Probability | None = None


class ScoreAnswer(_DecisionModel):
    type: Literal["score"]
    score: float
    probabilities: dict[str, Probability] | None = None
    confidence: Probability | None = None
    legend: dict[str, DecisionContext] | None = None


class BooleanAnswer(_DecisionModel):
    type: Literal["boolean"]
    probability: Probability


DecisionAnswer = Annotated[ChoiceAnswer | ScoreAnswer | BooleanAnswer, Field(discriminator="type")]


class DecisionUsage(_DecisionModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class DecisionRounding(_DecisionModel):
    probability_decimals: int | None = Field(default=None, ge=0)
    score_decimals: int | None = Field(default=None, ge=0)


class DecisionWarning(_DecisionModel):
    type: Literal["unsupported", "compatibility", "deprecated", "other"]
    feature: str | None = None
    details: str | None = None
    setting: str | None = None
    message: str | None = None


class DecisionQueryResponse(_DecisionModel):
    model: str
    provider: str | None = None
    id: str | None = None
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage | None = None
    rounding: DecisionRounding | None = None
    warnings: list[DecisionWarning] | None = None
    provider_metadata: dict[str, JsonValue] | None = None


def validate_decision_answers(request: DecisionQueryRequest, response: DecisionQueryResponse) -> None:
    if response.answers.keys() != request.questions.keys():
        raise ValueError("decision answer IDs must match question IDs")
    for key, question in request.questions.items():
        answer = response.answers[key]
        if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
            options = set(question.criteria)
            if answer.choice not in options:
                raise ValueError("decision choice must be a requested option")
            probabilities = answer.probabilities
        elif isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
            options = {str(index) for index in range(len(question.criteria))}
            if not 0 <= answer.score <= len(question.criteria) - 1:
                raise ValueError("decision score is outside the requested criteria range")
            if answer.legend is not None and set(answer.legend) != options:
                raise ValueError("decision legend must match requested score indices")
            probabilities = answer.probabilities
        elif isinstance(question, BooleanQuestion) and isinstance(answer, BooleanAnswer):
            continue
        else:
            raise ValueError("decision answer type must match question type")
        if probabilities is not None and set(probabilities) != options:
            raise ValueError("decision probabilities must match requested options")
