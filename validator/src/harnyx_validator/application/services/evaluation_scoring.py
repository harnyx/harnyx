"""Scoring helpers for generic miner task runs."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from harnyx_commons.domain.miner_task import MinerTask, Response, ScoreBreakdown
from harnyx_commons.llm.json_utils import pydantic_postprocessor
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_types import LlmProviderName
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

_COMPARISON_WEIGHT = 0.5
_SIMILARITY_WEIGHT = 0.5
_PAIRWISE_SYSTEM_PROMPT = (
    "You are a strict evaluator comparing two answers to the same query. "
    "Choose the answer that better answers the query with stronger factual correctness, "
    "coverage, and directness. Ignore writing style unless it affects correctness. "
    "Do not explain your choice. Return JSON only."
)


class TextEmbeddingPort(Protocol):
    async def embed(self, text: str) -> tuple[float, ...]:
        ...


class _PairwisePreference(BaseModel):
    preferred_position: Literal["first", "second"]


@dataclass(frozen=True, slots=True)
class EvaluationScoringConfig:
    provider: LlmProviderName
    model: str
    temperature: float | None = None
    max_output_tokens: int | None = 256
    reasoning_effort: str | None = None
    timeout_seconds: float = 30.0
    scoring_version: str = "v1"


class EvaluationScoringService:
    """Scores miner task responses against their reference answers."""

    def __init__(
        self,
        llm_provider: LlmProviderPort,
        embedding_client: TextEmbeddingPort,
        config: EvaluationScoringConfig,
    ) -> None:
        self._llm = llm_provider
        self._embeddings = embedding_client
        self._config = config

    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        similarity_score = await self._score_similarity(
            miner_response=response.text,
            reference_response=task.reference_answer.text,
        )
        comparison_score = await self._score_pairwise(
            query_text=task.query.text,
            miner_response=response.text,
            reference_response=task.reference_answer.text,
        )
        total_score = self._combine_scores(
            comparison_score=comparison_score,
            similarity_score=similarity_score,
        )
        return ScoreBreakdown(
            comparison_score=comparison_score,
            similarity_score=similarity_score,
            total_score=total_score,
            scoring_version=self._config.scoring_version,
        )

    def _combine_scores(
        self,
        *,
        comparison_score: float,
        similarity_score: float,
    ) -> float:
        _validate_score_weights(_COMPARISON_WEIGHT, _SIMILARITY_WEIGHT)
        return round(
            (_COMPARISON_WEIGHT * comparison_score) + (_SIMILARITY_WEIGHT * similarity_score),
            6,
        )

    async def _score_pairwise(
        self,
        *,
        query_text: str,
        miner_response: str,
        reference_response: str,
    ) -> float:
        miner_first = await self._judge_pair(
            query_text=query_text,
            first_answer=miner_response,
            second_answer=reference_response,
        )
        reference_first = await self._judge_pair(
            query_text=query_text,
            first_answer=reference_response,
            second_answer=miner_response,
        )
        miner_wins = 0
        if miner_first.preferred_position == "first":
            miner_wins += 1
        if reference_first.preferred_position == "second":
            miner_wins += 1
        return miner_wins / 2.0

    async def _judge_pair(
        self,
        *,
        query_text: str,
        first_answer: str,
        second_answer: str,
    ) -> _PairwisePreference:
        user_prompt = (
            f"Query:\n{query_text}\n\n"
            f"Answer 1:\n{first_answer}\n\n"
            f"Answer 2:\n{second_answer}\n\n"
            'Return JSON with {"preferred_position":"first"} or {"preferred_position":"second"}.'
        )
        request = LlmRequest(
            provider=self._config.provider,
            model=self._config.model,
            messages=(
                LlmMessage(
                    role="system",
                    content=(LlmMessageContentPart.input_text(_PAIRWISE_SYSTEM_PROMPT),),
                ),
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text(user_prompt),),
                ),
            ),
            output_mode="structured",
            output_schema=_PairwisePreference,
            postprocessor=pydantic_postprocessor(_PairwisePreference),
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            reasoning_effort=self._config.reasoning_effort,
            timeout_seconds=self._config.timeout_seconds,
            internal_metadata={"use_case": "miner_task_pairwise_judge"},
        )
        response = await self._llm.invoke(request)
        parsed = response.postprocessed
        if parsed is None:
            raise RuntimeError("pairwise judge did not return structured output")
        return _PairwisePreference.model_validate(parsed)

    async def _score_similarity(
        self,
        *,
        miner_response: str,
        reference_response: str,
    ) -> float:
        miner_vector, reference_vector = await self._embed_pair(
            miner_response=miner_response,
            reference_response=reference_response,
        )
        return _normalize_similarity(_cosine_similarity(miner_vector, reference_vector))

    async def _embed_pair(
        self,
        *,
        miner_response: str,
        reference_response: str,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        miner_vector, reference_vector = await asyncio.gather(
            self._embeddings.embed(miner_response),
            self._embeddings.embed(reference_response),
        )
        if len(miner_vector) != len(reference_vector):
            raise RuntimeError(
                "embedding dimensions mismatch for miner-response/reference-answer comparison"
            )
        return miner_vector, reference_vector


def _validate_score_weights(comparison_weight: float, similarity_weight: float) -> None:
    total_weight = comparison_weight + similarity_weight
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"scoring weights must sum to 1.0; got {total_weight}")


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise RuntimeError("embedding vectors must have positive norm")
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)


def _normalize_similarity(cosine_similarity: float) -> float:
    normalized_similarity = (cosine_similarity + 1.0) / 2.0
    return round(max(0.0, min(1.0, normalized_similarity)), 6)


__all__ = [
    "EvaluationScoringConfig",
    "EvaluationScoringService",
    "TextEmbeddingPort",
]
