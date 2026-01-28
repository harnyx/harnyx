"""Reference claim and rubric models shared across services."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from caster_commons.domain.verdict import VerdictOptions


@dataclass(frozen=True, slots=True)
class Rubric:
    """Evaluation rubric attached to each claim."""

    title: str
    description: str
    verdict_options: VerdictOptions

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("rubric title must not be empty")
        if not self.description.strip():
            raise ValueError("rubric description must not be empty")


@dataclass(frozen=True, slots=True)
class Citation:
    """Reference citation metadata."""

    url: str
    note: str

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("citation url must not be empty")
        if not self.note.strip():
            raise ValueError("citation note must not be empty")


@dataclass(frozen=True, slots=True)
class Span:
    """Indexed excerpt within the evaluated text."""

    excerpt: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.excerpt.strip():
            raise ValueError("span excerpt must not be empty")
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end < self.start:
            raise ValueError("span end must be greater than or equal to start")


@dataclass(frozen=True, slots=True)
class ReferenceAnswer:
    """Curated reference answer used for scoring."""

    verdict: int
    justification: str
    citations: tuple[Citation, ...] = ()
    spans: tuple[Span, ...] = ()

    def __post_init__(self) -> None:
        if not self.justification.strip():
            raise ValueError("reference justification must not be empty")


@dataclass(frozen=True, slots=True)
class MinerTaskClaim:
    """Canonical miner-task claim evaluated by the subnet."""

    claim_id: UUID
    text: str
    rubric: Rubric
    reference_answer: ReferenceAnswer
    budget_usd: float = 0.05

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("claim text must not be empty")
        if self.budget_usd < 0.0:
            raise ValueError("claim budget_usd must be non-negative")
        self.rubric.verdict_options.validate(self.reference_answer.verdict)


@dataclass(frozen=True, slots=True)
class GeneratedClaim:
    """Raw claim emitted by generators; reference answer filled later."""

    claim_id: UUID
    text: str
    rubric: Rubric
    verdict: int
    justification: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("generated claim text must not be empty")
        self.rubric.verdict_options.validate(self.verdict)
        if not self.justification.strip():
            raise ValueError("generated justification must not be empty")


__all__ = [
    "Rubric",
    "Citation",
    "ReferenceAnswer",
    "Span",
    "MinerTaskClaim",
    "GeneratedClaim",
]
