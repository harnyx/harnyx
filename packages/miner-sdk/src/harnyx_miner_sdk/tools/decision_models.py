"""Canonical miner-facing decision query contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from harnyx_miner_sdk.tools.extraction_models import ExtractionAnswer, ExtractionQuestion, validate_extraction_answer
from harnyx_miner_sdk.tools.openrouter_models import validate_paid_openrouter_model
from harnyx_miner_sdk.tools.types import ToolInvocationTimeout

DecisionProviderName = Literal["openrouter", "ai_gateway", "cloudflare", "jev"]
DecisionContext = str | dict[str, JsonValue] | list[JsonValue]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
MINER_SELECTED_DECISION_PROVIDER_MODELS: Mapping[DecisionProviderName, tuple[str, ...]] = {
    # Historical pricing baseline; OpenRouter eligibility uses its live catalog.
    "openrouter": (
        "cloudflare/clef-flash",
        "cloudflare/clef",
        "jaredpalmer/kev-4b",
        "nace-ai/drex-v2.0",
        "respan/span-02",
    ),
    "ai_gateway": (),
    "jev": ("jev-1.13.0",),
    "cloudflare": (
        "@cf/cloudflare/clef", "@cf/cloudflare/clef-flash", "@cf/cloudflare/clef-omni", "@cf/cloudflare/clairic",
    ),
}


def parse_decision_provider(provider: str) -> DecisionProviderName:
    if provider == "openrouter":
        return "openrouter"
    if provider == "ai_gateway":
        return "ai_gateway"
    if provider == "cloudflare":
        return "cloudflare"
    if provider == "jev":
        return "jev"
    raise ValueError(f"unsupported decision provider {provider!r}")


def validate_decision_provider_model(provider: DecisionProviderName, model: str) -> None:
    if provider == "openrouter":
        validate_paid_openrouter_model(model)
    elif model not in MINER_SELECTED_DECISION_PROVIDER_MODELS[provider]:
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


def _same_json_value(left: JsonValue, right: JsonValue) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_json_value(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_json_value(a, b) for a, b in zip(left, right, strict=True))
    return left == right


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


DecisionQuestion = Annotated[
    ChoiceQuestion | ScoreQuestion | BooleanQuestion | ExtractionQuestion, Field(discriminator="type")
]


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
        self.model = self.model.strip()
        validate_decision_provider_model(self.provider, self.model)
        if any(not key for key in self.questions):
            raise ValueError("question IDs must not be empty")
        extraction = self.provider == "cloudflare" and self.model == "@cf/cloudflare/clairic"
        if extraction and self.state == "":
            raise ValueError("Cloudflare extraction requires non-empty state")
        for key, question in self.questions.items():
            if isinstance(question, ExtractionQuestion) != extraction:
                raise ValueError("Clairic requires extraction questions; other models require decision questions")
            if extraction and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.\-]{0,63}", key) is None:
                raise ValueError("invalid Clairic question ID")
            if self.provider == "jev":
                if isinstance(question, ChoiceQuestion) and len(question.criteria) > 255:
                    raise ValueError("JEV choice questions support up to 255 options")
                if isinstance(question, ScoreQuestion) and not 2 <= len(question.criteria) <= 10:
                    raise ValueError("JEV score questions require 2 to 10 levels")
            if self.provider == "cloudflare" and not extraction:
                if len(self.questions) > 64 or re.fullmatch(r"[A-Za-z0-9_.\-]{1,100}", key) is None:
                    raise ValueError("Clef supports up to 64 named questions with valid IDs")
                if not question.instructions:
                    raise ValueError("Clef requires non-empty question instructions")
                if isinstance(question, ChoiceQuestion) and not 2 <= len(question.criteria) <= 255:
                    raise ValueError("Clef choice questions require 2 to 255 options")
                if isinstance(question, ScoreQuestion) and not 2 <= len(question.criteria) <= 10:
                    raise ValueError("Clef score questions require 2 to 10 levels")
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


DecisionAnswer = Annotated[
    ChoiceAnswer | ScoreAnswer | BooleanAnswer | ExtractionAnswer, Field(discriminator="type")
]


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
    truncated: bool | None = None


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
            if request.provider == "cloudflare" and answer.legend is not None:
                if any(
                    not _same_json_value(answer.legend[str(index)], criterion)
                    for index, criterion in enumerate(question.criteria)
                ):
                    raise ValueError("decision legend must preserve requested score criteria")
            probabilities = answer.probabilities
        elif isinstance(question, BooleanQuestion) and isinstance(answer, BooleanAnswer):
            continue
        elif isinstance(question, ExtractionQuestion) and isinstance(answer, ExtractionAnswer):
            state = request.state if isinstance(request.state, str) else json.dumps(request.state, ensure_ascii=False)
            _validate_finite_json(answer.model_dump(mode="json"))
            validate_extraction_answer(question, answer, state=state)
            continue
        else:
            raise ValueError("decision answer type must match question type")
        if probabilities is not None and set(probabilities) != options:
            raise ValueError("decision probabilities must match requested options")
        if request.provider == "cloudflare" and probabilities is not None:
            if not math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("decision probabilities must sum to 1")
            if isinstance(answer, ChoiceAnswer) and not math.isclose(
                probabilities[answer.choice], max(probabilities.values()), rel_tol=0.0, abs_tol=1e-6,
            ):
                raise ValueError("decision choice must have the highest probability")
            if isinstance(answer, ScoreAnswer):
                expected_score = math.fsum(int(level) * probability for level, probability in probabilities.items())
                if not math.isclose(answer.score, expected_score, rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError("decision score must equal its probability-weighted level")
