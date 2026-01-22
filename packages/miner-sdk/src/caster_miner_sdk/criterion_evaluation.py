"""Criterion evaluation request/response contracts for miners."""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, ConfigDict

from caster_miner_sdk.verdict import VerdictOption, VerdictOptions


class VerdictOptionPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        str_min_length=1,
    )

    value: int
    description: str

    def to_domain(self) -> VerdictOption:
        return VerdictOption(value=self.value, description=self.description)


class CriterionEvaluationRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        str_min_length=1,
    )

    claim_text: str
    rubric_title: str
    rubric_description: str
    verdict_options: list[VerdictOptionPayload]

    def verdict_options_domain(self) -> VerdictOptions:
        options = tuple(entry.to_domain() for entry in self.verdict_options)
        return VerdictOptions(options=options)


class CriterionEvaluationVerdict(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        str_min_length=1,
    )

    verdict: int
    justification: str

    @classmethod
    def from_llm_content(cls, content: str) -> CriterionEvaluationVerdict:
        text = content.strip()
        if text.startswith("```"):
            text = _strip_code_fence(text)
        return cls.model_validate_json(text)


class CriterionEvaluationCitationRef(TypedDict):
    receipt_id: str
    result_id: str


class CriterionEvaluationResponse(TypedDict):
    verdict: int
    justification: str
    citations: list[CriterionEvaluationCitationRef]


def _strip_code_fence(text: str) -> str:
    fence = "```"
    if not text.startswith(fence):
        return text
    stripped = text[len(fence) :]
    if stripped.startswith("json"):
        stripped = stripped[4:]
    if stripped.endswith(fence):
        stripped = stripped[: -len(fence)]
    return stripped.strip()


__all__ = [
    "CriterionEvaluationCitationRef",
    "CriterionEvaluationRequest",
    "CriterionEvaluationResponse",
    "CriterionEvaluationVerdict",
    "VerdictOptionPayload",
]
