"""Shared miner-task similarity voting contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from harnyx_commons.llm.provider_types import LlmProviderName

SimilarityVerdict = Literal["not_duplicate", "duplicate"]
SimilarityVoteStatus = Literal["responded", "disqualified"]


@dataclass(frozen=True, slots=True)
class SimilarityJudgeRequest:
    batch_id: UUID
    candidate_artifact_id: UUID
    incumbent_artifact_id: UUID
    candidate_miner_uid: int
    incumbent_miner_uid: int
    incumbent_script: str
    candidate_diff: str


@dataclass(frozen=True, slots=True)
class SimilarityJudgeResult:
    verdict: SimilarityVerdict
    reasoning: str | None
    reasoning_tokens: int | None
    model: str
    provider: LlmProviderName


@dataclass(frozen=True, slots=True)
class SimilarityVoteInput:
    status: SimilarityVoteStatus
    verdict: SimilarityVerdict | None


@dataclass(frozen=True, slots=True)
class SimilarityVoteTally:
    responding_validator_count: int
    not_duplicate_votes: int
    duplicate_votes: int
    disqualified_count: int
    passes: bool | None


def tally_similarity_votes(votes: tuple[SimilarityVoteInput, ...]) -> SimilarityVoteTally:
    responding_validator_count = 0
    not_duplicate_votes = 0
    duplicate_votes = 0
    disqualified_count = 0
    for vote in votes:
        if vote.status == "disqualified":
            if vote.verdict is not None:
                raise ValueError("disqualified similarity votes must not include a verdict")
            disqualified_count += 1
            continue
        if vote.verdict is None:
            raise ValueError("responded similarity votes must include a verdict")
        responding_validator_count += 1
        if vote.verdict == "not_duplicate":
            not_duplicate_votes += 1
        elif vote.verdict == "duplicate":
            duplicate_votes += 1
        else:
            raise ValueError(f"unsupported similarity verdict: {vote.verdict}")

    return SimilarityVoteTally(
        responding_validator_count=responding_validator_count,
        not_duplicate_votes=not_duplicate_votes,
        duplicate_votes=duplicate_votes,
        disqualified_count=disqualified_count,
        passes=None
        if responding_validator_count == 0
        else not_duplicate_votes * 2 >= responding_validator_count,
    )


__all__ = [
    "SimilarityJudgeRequest",
    "SimilarityJudgeResult",
    "SimilarityVerdict",
    "SimilarityVoteInput",
    "SimilarityVoteStatus",
    "SimilarityVoteTally",
    "tally_similarity_votes",
]
