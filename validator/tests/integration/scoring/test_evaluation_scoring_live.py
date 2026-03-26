from __future__ import annotations

from uuid import uuid4

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer, Response
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_resolver
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_validator.application.services.evaluation_scoring import (
    EvaluationScoringConfig,
    EvaluationScoringService,
)
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


class StubEmbeddingClient:
    async def embed(self, _text: str) -> tuple[float, ...]:
        return (1.0, 0.0, 0.0)


class RecordingProvider(LlmProviderPort):
    def __init__(self, delegate: LlmProviderPort) -> None:
        self._delegate = delegate
        self.requests: list[AbstractLlmRequest] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        return await self._delegate.invoke(request)

    async def aclose(self) -> None:
        await self._delegate.aclose()


async def test_evaluation_scoring_live_uses_real_structured_chutes_flow() -> None:
    settings = Settings.load()
    provider_name = settings.llm.scoring_llm_provider

    resolve_provider = build_cached_llm_provider_resolver(
        llm_settings=settings.llm,
        vertex_settings=settings.vertex,
    )
    llm_provider = RecordingProvider(resolve_provider(provider_name))
    service = EvaluationScoringService(
        llm_provider=llm_provider,
        embedding_client=StubEmbeddingClient(),
        config=EvaluationScoringConfig(
            provider=provider_name,
            model=bootstrap._SCORING_LLM_MODEL,
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            temperature=0.0,
            max_output_tokens=256,
            timeout_seconds=float(settings.llm.scoring_llm_timeout_seconds),
        ),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the capital of France?"),
        reference_answer=ReferenceAnswer(text="Paris is the capital of France."),
    )

    try:
        score = await service.score(
            task=task,
            response=Response(text="Paris is the capital of France."),
        )
    finally:
        await llm_provider.aclose()

    assert len(llm_provider.requests) == 2
    assert all(request.output_mode == "structured" for request in llm_provider.requests)
    assert score.scoring_version == "v1"
    assert 0.0 <= score.comparison_score <= 1.0
    assert score.similarity_score == pytest.approx(1.0)
    assert 0.0 <= score.total_score <= 1.0
