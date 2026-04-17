from __future__ import annotations

import json
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
        self.responses: list[LlmResponse] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        response = await self._delegate.invoke(request)
        self.responses.append(response)
        return response

    async def aclose(self) -> None:
        await self._delegate.aclose()

async def test_evaluation_scoring_live_uses_real_structured_runtime_flow() -> None:
    base_settings = Settings.load()
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps(
                        {"scoring": {bootstrap._SCORING_LLM_MODEL: "bedrock"}}
                    )
                }
            )
        }
    )
    scoring_route = bootstrap._resolve_scoring_judge_route(settings)

    resolve_provider = build_cached_llm_provider_resolver(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    llm_provider = RecordingProvider(resolve_provider(scoring_route.provider))
    service = EvaluationScoringService(
        llm_provider=llm_provider,
        embedding_client=StubEmbeddingClient(),
        config=EvaluationScoringConfig(
            provider=scoring_route.provider,
            model=scoring_route.model,
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            temperature=0.0,
            max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
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
    assert all(request.provider == scoring_route.provider for request in llm_provider.requests)
    assert all(request.model == scoring_route.model for request in llm_provider.requests)
    assert score.scoring_version == "v1"
    assert 0.0 <= score.comparison_score <= 1.0
    assert score.similarity_score == pytest.approx(1.0)
    assert 0.0 <= score.total_score <= 1.0
    observed_reasoning = [
        response.choices[0].message.reasoning
        for response in llm_provider.responses
        if response.choices and response.choices[0].message.reasoning is not None
    ]
    if observed_reasoning:
        assert score.reasoning is not None
        assert score.reasoning.text is not None
        assert score.reasoning.text.strip()
        if score.reasoning.reasoning_tokens is not None:
            assert score.reasoning.reasoning_tokens >= 0
