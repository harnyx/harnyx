from __future__ import annotations

import json
from uuid import uuid4

import pytest

from harnyx_commons.domain.miner_task import (
    AnswerCitation,
    CitationExcerpt,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
)
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.miner_task_scoring import (
    EvaluationScoringConfig,
    EvaluationScoringService,
)
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]
_GEMMA_MODEL = "google/gemma-4-31B-turbo-TEE"
_GEMMA_ROUTE_TARGET = "openrouter"
_QWEN36_MODEL = "Qwen/Qwen3.6-27B-TEE"
_QWEN36_ROUTE_TARGET = "openrouter"
_KIMI_MODEL = "moonshotai/Kimi-K2.5-TEE"
_KIMI_ROUTE_TARGET = "bedrock"


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


@pytest.mark.parametrize(
    ("model", "route_target", "fast", "request_count"),
    (
        (_GEMMA_MODEL, _GEMMA_ROUTE_TARGET, False, 2),
        (_GEMMA_MODEL, _GEMMA_ROUTE_TARGET, True, 1),
        (_QWEN36_MODEL, _QWEN36_ROUTE_TARGET, False, 2),
        (_QWEN36_MODEL, _QWEN36_ROUTE_TARGET, True, 1),
    ),
)
@pytest.mark.parametrize("scoring_contract", ["native-v1", "f1-deep-v1"])
async def test_evaluation_scoring_live_uses_real_structured_runtime_flow(
    model: str,
    route_target: str,
    fast: bool,
    request_count: int,
    scoring_contract: str,
) -> None:
    base_settings = Settings.load()
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={"llm_model_provider_overrides_json": json.dumps({"scoring": {model: route_target}})},
            ),
        }
    )
    scoring_route = bootstrap._resolve_scoring_judge_route(settings, model=model)
    assert scoring_route.provider == route_target
    assert scoring_route.model == model

    registry = build_cached_llm_provider_registry(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    routed_provider = build_routed_llm_provider(
        surface="scoring",
        default_provider=settings.llm.scoring_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "openrouter", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )
    llm_provider = RecordingProvider(routed_provider)
    service = EvaluationScoringService(
        llm_provider=llm_provider,
        config=EvaluationScoringConfig(
            provider=settings.llm.scoring_llm_provider,
            model=scoring_route.model,
            fallback_models=(),
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            temperature=0.0,
            max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
        ),
    )
    citations = (
        AnswerCitation(
            url="https://example.com/geography",
            excerpts=(
                CitationExcerpt(start=0, end=31, text="Paris is the capital of France."),
                CitationExcerpt(start=100, end=131, text="France has its capital in Paris"),
            ),
        ),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(
            text="Return the capital of France.",
            fast=fast,
            output_schema={
                "type": "object",
                "properties": {"capital": {"type": "string"}},
                "required": ["capital"],
                "additionalProperties": False,
            },
        ),
        reference_answer=ReferenceAnswer(text='{"capital":"Paris"}', citations=citations),
    )

    try:
        score = await service.score(
            task=task,
            response=Response(output={"capital": "Paris"}, citations=citations),
            scoring_contract=scoring_contract,
        )
    finally:
        await registry.aclose()

    if not fast:
        for request in llm_provider.requests:
            _, payload_json = request.messages[1].content[0].text.split("Payload:\n", 1)
            payload = json.loads(payload_json)
            if "answers" not in payload:
                continue
            for answer in payload["answers"]:
                assert answer["validated_citations"] == [
                    {
                        "url": "https://example.com/geography",
                        "excerpts": [e.text for e in citations[0].excerpts],
                    }
                ]
    assert len(llm_provider.requests) == request_count + (1 if not fast and scoring_contract == "f1-deep-v1" else 0)
    assert all(request.output_mode == "structured" for request in llm_provider.requests)
    assert all(request.provider == settings.llm.scoring_llm_provider for request in llm_provider.requests)
    assert all(request.model == scoring_route.model for request in llm_provider.requests)
    assert all('"output_contract"' in request.messages[1].content[0].text for request in llm_provider.requests)
    assert all(response.metadata is not None for response in llm_provider.responses)
    assert all(response.metadata["selected_provider"] == scoring_route.provider for response in llm_provider.responses)
    assert all(response.metadata["selected_model"] == scoring_route.model for response in llm_provider.responses)
    assert score.scoring_version == (
        "f1-deep-v1" if scoring_contract == "f1-deep-v1" else "deepsearchqa-f1-v1" if fast else "v1"
    )
    if scoring_contract == "f1-deep-v1":
        assert score.score_breakdown.f1_score == 1.0
        assert score.score_breakdown.f1_score_evidence is not None
    assert 0.0 <= score.comparison_score <= 1.0
    assert score.total_score == pytest.approx(score.comparison_score)
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


@pytest.mark.parametrize(
    ("model", "route_target"),
    ((_KIMI_MODEL, _KIMI_ROUTE_TARGET),),
)
async def test_evaluation_scoring_live_accepts_fallback_candidate_route(
    model: str,
    route_target: str,
) -> None:
    base_settings = Settings.load()
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={
                    "llm_model_provider_overrides_json": json.dumps({"scoring": {model: route_target}}),
                }
            ),
        }
    )
    scoring_route = bootstrap._resolve_scoring_judge_route(settings, model=model)
    assert scoring_route.provider == route_target
    assert scoring_route.model == model

    registry = build_cached_llm_provider_registry(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    routed_provider = build_routed_llm_provider(
        surface="scoring",
        default_provider=settings.llm.scoring_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "openrouter", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )
    llm_provider = RecordingProvider(routed_provider)
    service = EvaluationScoringService(
        llm_provider=llm_provider,
        config=EvaluationScoringConfig(
            provider=settings.llm.scoring_llm_provider,
            model=scoring_route.model,
            fallback_models=(),
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            temperature=0.0,
            max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
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
        await registry.aclose()

    assert len(llm_provider.requests) == 2
    assert all(request.output_mode == "structured" for request in llm_provider.requests)
    assert all(request.provider == settings.llm.scoring_llm_provider for request in llm_provider.requests)
    assert all(request.model == scoring_route.model for request in llm_provider.requests)
    assert all(response.metadata is not None for response in llm_provider.responses)
    assert all(response.metadata["selected_provider"] == route_target for response in llm_provider.responses)
    assert all(response.metadata["selected_model"] == scoring_route.model for response in llm_provider.responses)
    assert score.scoring_version == "v1"
    assert 0.0 <= score.comparison_score <= 1.0
    assert score.total_score == pytest.approx(score.comparison_score)
