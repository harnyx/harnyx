"""Miner evaluation payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from caster_commons.domain.claim import Rubric


@dataclass(frozen=True, slots=True)
class MinerCitation:
    """Citation emitted by a miner."""

    url: str | None
    note: str | None
    receipt_id: str
    result_id: str

    def __post_init__(self) -> None:
        if self.url is not None and not self.url.strip():
            raise ValueError("citation url must not be empty when supplied")
        if self.note is not None and not self.note.strip():
            raise ValueError("citation note must not be empty when supplied")
        if not self.receipt_id.strip():
            raise ValueError("citation receipt_id must not be empty")
        if not self.result_id.strip():
            raise ValueError("citation result_id must not be empty")


@dataclass(frozen=True, slots=True)
class MinerAnswer:
    """Structured response returned by a miner."""

    verdict: int
    justification: str
    citations: tuple[MinerCitation, ...] = ()

    def __post_init__(self) -> None:
        if not self.justification.strip():
            raise ValueError("justification must not be empty")


@dataclass(frozen=True, slots=True)
class MinerEvaluation:
    """Recorded miner evaluation for a single claim run."""

    evaluation_id: UUID
    session_id: UUID
    uid: int
    claim_id: UUID
    rubric: Rubric
    miner_answer: MinerAnswer
    completed_at: datetime

    def __post_init__(self) -> None:
        if self.uid <= 0:
            raise ValueError("uid must be positive")


__all__ = ["MinerAnswer", "MinerCitation", "MinerEvaluation"]
