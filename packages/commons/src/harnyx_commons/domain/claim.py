"""Rubric and reference-answer models shared across content-review services."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from harnyx_commons.domain.shared_config import COMMONS_STRICT_DATACLASS_CONFIG
from harnyx_commons.domain.verdict import VerdictOptions

NonEmptyText = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


@dataclass(frozen=True, slots=True, config=COMMONS_STRICT_DATACLASS_CONFIG)
class Rubric:
    """Evaluation rubric attached to each generated claim."""

    title: NonEmptyText
    description: NonEmptyText
    verdict_options: VerdictOptions


@dataclass(frozen=True, slots=True, config=COMMONS_STRICT_DATACLASS_CONFIG)
class Citation:
    """Reference citation metadata."""

    url: NonEmptyText
    note: NonEmptyText


@dataclass(frozen=True, slots=True, config=COMMONS_STRICT_DATACLASS_CONFIG)
class Span:
    """Indexed excerpt within the evaluated text."""

    excerpt: NonEmptyText
    start: NonNegativeInt
    end: NonNegativeInt

    @model_validator(mode="after")
    def _validate_bounds(self) -> Span:
        if self.end < self.start:
            raise ValueError("span end must be greater than or equal to start")
        return self


@dataclass(frozen=True, slots=True, config=COMMONS_STRICT_DATACLASS_CONFIG)
class FeedSearchContext:
    """Feed provenance kept for content-review flows."""

    feed_id: UUID
    enqueue_seq: NonNegativeInt


@dataclass(frozen=True, slots=True, config=COMMONS_STRICT_DATACLASS_CONFIG)
class GeneratedClaim:
    """Raw claim emitted by generators; reference answer filled later."""

    claim_id: UUID
    text: NonEmptyText
    rubric: Rubric
    verdict: int
    justification: NonEmptyText

    @model_validator(mode="after")
    def _validate_verdict(self) -> GeneratedClaim:
        self.rubric.verdict_options.validate(self.verdict)
        return self


__all__ = [
    "Citation",
    "FeedSearchContext",
    "GeneratedClaim",
    "Rubric",
    "Span",
]
