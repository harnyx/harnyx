from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer, Response
from harnyx_validator.application.services.evaluation_scoring import (
    EvaluationScoringConfig,
    EvaluationScoringService,
    _validate_score_weights,
)

pytestmark = pytest.mark.anyio("asyncio")


class StubLlmProvider:
    def __init__(self, preferences: list[str]) -> None:
        self._preferences = preferences
        self.requests: list[object] = []

    async def invoke(self, request: object) -> object:
        self.requests.append(request)
        if not self._preferences:
            raise RuntimeError("missing pairwise preference")
        return SimpleNamespace(postprocessed={"preferred_position": self._preferences.pop(0)})

    async def aclose(self) -> None:
        return None


class StubEmbeddingClient:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self._vectors = vectors

    async def embed(self, text: str) -> tuple[float, ...]:
        return self._vectors[text]


class OverlapTrackingEmbeddingClient:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self._vectors = vectors
        self.active_calls = 0
        self.max_active_calls = 0

    async def embed(self, text: str) -> tuple[float, ...]:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        await asyncio.sleep(0.01)
        self.active_calls -= 1
        return self._vectors[text]


async def test_scoring_service_combines_pairwise_and_similarity_scores() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    service = EvaluationScoringService(
        llm_provider=StubLlmProvider(["first", "second"]),
        embedding_client=StubEmbeddingClient(
            {
                "Miner says 42.": (1.0, 0.0),
                "The answer is 42.": (1.0, 0.0),
            },
        ),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(task=task, response=Response(text="Miner says 42."))

    assert score.comparison_score == pytest.approx(1.0)
    assert score.similarity_score == pytest.approx(1.0)
    assert score.total_score == pytest.approx(1.0)


async def test_scoring_service_records_split_pairwise_decision() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Summarize the result."),
        reference_answer=ReferenceAnswer(text="Reference summary."),
    )
    service = EvaluationScoringService(
        llm_provider=StubLlmProvider(["first", "first"]),
        embedding_client=StubEmbeddingClient(
            {
                "Miner summary.": (1.0, 0.0),
                "Reference summary.": (0.0, 1.0),
            },
        ),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(task=task, response=Response(text="Miner summary."))

    assert score.comparison_score == pytest.approx(0.5)
    assert score.similarity_score == pytest.approx(0.5)
    assert score.total_score == pytest.approx(0.5)


async def test_scoring_service_normalizes_negative_cosine_similarity() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="State the opposite."),
        reference_answer=ReferenceAnswer(text="Reference answer."),
    )
    service = EvaluationScoringService(
        llm_provider=StubLlmProvider(["second", "first"]),
        embedding_client=StubEmbeddingClient(
            {
                "Miner opposite.": (1.0, 0.0),
                "Reference answer.": (-1.0, 0.0),
            },
        ),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(task=task, response=Response(text="Miner opposite."))

    assert score.comparison_score == pytest.approx(0.0)
    assert score.similarity_score == pytest.approx(0.0)
    assert score.total_score == pytest.approx(0.0)


async def test_score_fails_explicitly_when_embeddings_are_unavailable_at_score_time() -> None:
    from harnyx_validator.infrastructure.scoring.vertex_embedding import MissingTextEmbeddingClient

    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Explain the answer."),
        reference_answer=ReferenceAnswer(text="Reference answer."),
    )
    service = EvaluationScoringService(
        llm_provider=StubLlmProvider(["first", "second"]),
        embedding_client=MissingTextEmbeddingClient(
            "GCP_PROJECT_ID must be configured for validator run scoring embeddings"
        ),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    with pytest.raises(RuntimeError, match="GCP_PROJECT_ID must be configured"):
        await service.score(task=task, response=Response(text="Miner answer."))


async def test_scoring_service_embeds_miner_and_reference_concurrently() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Compare the answers."),
        reference_answer=ReferenceAnswer(text="Reference answer."),
    )
    embeddings = OverlapTrackingEmbeddingClient(
        {
            "Miner answer.": (1.0, 0.0),
            "Reference answer.": (1.0, 0.0),
        }
    )
    service = EvaluationScoringService(
        llm_provider=StubLlmProvider(["first", "second"]),
        embedding_client=embeddings,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    await service.score(task=task, response=Response(text="Miner answer."))

    assert embeddings.max_active_calls == 2


def test_validate_score_weights_requires_sum_of_one() -> None:
    with pytest.raises(RuntimeError, match="scoring weights must sum to 1.0"):
        _validate_score_weights(0.5, 0.6)
