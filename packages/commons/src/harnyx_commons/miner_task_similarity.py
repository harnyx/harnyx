"""Shared miner-task similarity classification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.llm.provider_types import LlmRouteTarget

SimilarityClassification = Literal["duplicate", "near_duplicate", "novel"]
StoredSimilarityClassification = Literal[
    "not_duplicate",
    "duplicate",
    "near_duplicate",
    "novel",
]
EligibleSimilarityClassification = Literal["near_duplicate", "novel"]
SimilarityVoteStatus = Literal["responded", "disqualified"]


@dataclass(frozen=True, slots=True)
class SimilarityJudgeRequest:
    batch_id: UUID
    candidate_artifact_id: UUID
    reference_artifact_id: UUID
    candidate_miner_uid: int
    reference_miner_uid: int
    reference_script: str
    candidate_diff: str


@dataclass(frozen=True, slots=True)
class SimilarityJudgeResult:
    classification: SimilarityClassification
    reasoning: str | None
    reasoning_tokens: int | None
    model: str
    provider: LlmRouteTarget
    judge_usage: JudgeUsageSummary | None = None


@dataclass(frozen=True, slots=True)
class SimilarityVoteInput:
    status: SimilarityVoteStatus
    classification: StoredSimilarityClassification | None


@dataclass(frozen=True, slots=True)
class SimilarityVoteTally:
    responding_validator_count: int
    eligible_votes: int
    not_duplicate_votes: int
    duplicate_votes: int
    near_duplicate_votes: int
    novel_votes: int
    disqualified_count: int
    passes: bool | None
    eligible_classification: EligibleSimilarityClassification | None


def tally_similarity_votes(votes: tuple[SimilarityVoteInput, ...]) -> SimilarityVoteTally:
    responding_validator_count = 0
    eligible_votes = 0
    not_duplicate_votes = 0
    duplicate_votes = 0
    near_duplicate_votes = 0
    novel_votes = 0
    disqualified_count = 0
    classifications: list[StoredSimilarityClassification] = []
    for vote in votes:
        if vote.status == "disqualified":
            if vote.classification is not None:
                raise ValueError("disqualified similarity votes must not include a classification")
            disqualified_count += 1
            continue
        if vote.classification is None:
            raise ValueError("responded similarity votes must include a classification")
        responding_validator_count += 1
        classifications.append(vote.classification)
        if vote.classification == "not_duplicate":
            not_duplicate_votes += 1
            eligible_votes += 1
        elif vote.classification == "duplicate":
            duplicate_votes += 1
        elif vote.classification == "near_duplicate":
            near_duplicate_votes += 1
            eligible_votes += 1
        elif vote.classification == "novel":
            novel_votes += 1
            eligible_votes += 1
        else:
            raise ValueError(f"unsupported similarity classification: {vote.classification}")

    passes = None if responding_validator_count == 0 else eligible_votes * 2 >= responding_validator_count

    return SimilarityVoteTally(
        responding_validator_count=responding_validator_count,
        eligible_votes=eligible_votes,
        not_duplicate_votes=not_duplicate_votes,
        duplicate_votes=duplicate_votes,
        near_duplicate_votes=near_duplicate_votes,
        novel_votes=novel_votes,
        disqualified_count=disqualified_count,
        passes=passes,
        eligible_classification=_aggregate_eligible_classification(classifications) if passes else None,
    )


def _aggregate_eligible_classification(
    classifications: list[StoredSimilarityClassification],
) -> EligibleSimilarityClassification | None:
    if not classifications or "not_duplicate" in classifications:
        return None
    ordered = sorted(
        "near_duplicate" if classification == "duplicate" else classification
        for classification in classifications
    )
    lower_median = ordered[(len(ordered) - 1) // 2]
    if lower_median == "near_duplicate":
        return "near_duplicate"
    if lower_median == "novel":
        return "novel"
    raise ValueError(f"unsupported eligible classification: {lower_median}")


__all__ = [
    "EligibleSimilarityClassification",
    "SimilarityClassification",
    "SimilarityJudgeRequest",
    "SimilarityJudgeResult",
    "StoredSimilarityClassification",
    "SimilarityVoteInput",
    "SimilarityVoteStatus",
    "SimilarityVoteTally",
    "tally_similarity_votes",
]
