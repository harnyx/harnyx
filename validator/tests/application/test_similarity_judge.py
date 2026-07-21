from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest
from harnyx_validator.application.similarity_judge import SimilarityJudge, SimilarityJudgeConfig

pytestmark = pytest.mark.anyio("asyncio")


class StubLlmProvider:
    def __init__(self) -> None:
        self.requests: list[AbstractLlmRequest] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            id="stub-response",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(),
                        reasoning="candidate changes retrieval strategy",
                    ),
                ),
            ),
            usage=LlmUsage(reasoning_tokens=17),
            postprocessed={
                "classification": "novel",
                "reasoning": (
                    "The candidate changes the retrieval strategy by adding a cross-source "
                    "verification step before synthesis."
                ),
                "mechanism_change": "cross-source verification before synthesis",
            },
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        return None


class SequenceLlmProvider:
    def __init__(self, outcomes: list[LlmResponse | Exception]) -> None:
        self._outcomes = outcomes
        self.requests: list[AbstractLlmRequest] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        if not self._outcomes:
            raise RuntimeError("missing similarity outcome")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        return None


def _similarity_payload(request: AbstractLlmRequest) -> dict[str, object]:
    user_prompt = request.messages[1].content[0].text
    _, payload_json = user_prompt.split("Payload:\n", 1)
    return json.loads(payload_json)


def _similarity_response(
    *,
    classification: str = "novel",
    reasoning: str = "The candidate changes retrieval strategy.",
    mechanism_change: str | None = "retrieval strategy change",
    reasoning_text: str | None = "candidate changes retrieval strategy",
    reasoning_tokens: int | None = 17,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    metadata: dict[str, object] | None = None,
    finish_reason: str = "stop",
) -> LlmResponse:
    return LlmResponse(
        id="stub-response",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(),
                    reasoning=reasoning_text,
                ),
            ),
        ),
        usage=LlmUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        postprocessed=_similarity_postprocessed(
            classification=classification,
            reasoning=reasoning,
            mechanism_change=mechanism_change,
        ),
        metadata=metadata,
        finish_reason=finish_reason,
    )


def _similarity_postprocessed(
    *,
    classification: str,
    reasoning: str | None,
    mechanism_change: str | None,
) -> dict[str, str | None]:
    postprocessed: dict[str, str | None] = {"classification": classification}
    if reasoning is not None:
        postprocessed["reasoning"] = reasoning
    if mechanism_change is not None:
        postprocessed["mechanism_change"] = mechanism_change
    return postprocessed


def _raw_similarity_response(text: str) -> LlmResponse:
    return LlmResponse(
        id="raw-response",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart.input_text(text),),
                ),
            ),
        ),
        usage=LlmUsage(),
        finish_reason="stop",
    )


async def test_similarity_judge_returns_classification_and_validator_reasoning() -> None:
    llm = StubLlmProvider()
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="moonshotai/Kimi-K2.5-TEE",
            temperature=None,
            max_output_tokens=20480,
            reasoning_effort="high",
            timeout_seconds=300.0,
        ),
    )
    request = SimilarityJudgeRequest(
        batch_id=uuid4(),
        candidate_artifact_id=uuid4(),
        reference_artifact_id=uuid4(),
        candidate_miner_uid=20,
        reference_miner_uid=10,
        reference_script="def answer(): return 'old'",
        candidate_diff="+ def answer(): return 'new'",
    )

    result = await service.judge(request)

    assert result.classification == "novel"
    assert (
        result.reasoning
        == "The candidate changes the retrieval strategy by adding a cross-source verification step before synthesis.\n"
        "Mechanism change: cross-source verification before synthesis"
    )
    assert result.reasoning_tokens == 17
    assert result.model == "moonshotai/Kimi-K2.5-TEE"
    assert result.provider == "chutes"
    assert result.judge_usage is not None
    assert result.judge_usage.call_count == 1
    assert result.judge_usage.reasoning_tokens == 17
    llm_request = llm.requests[0]
    assert llm_request.provider == "chutes"
    assert llm_request.model == "moonshotai/Kimi-K2.5-TEE"
    assert llm_request.output_mode == "structured"
    assert llm_request.reasoning_effort == "high"
    assert llm_request.timeout_seconds == 300.0
    assert llm_request.use_case == "miner_task_similarity_judge"
    payload = _similarity_payload(llm_request)
    assert payload["reference"]["script"] == "def answer(): return 'old'"
    assert payload["candidate"]["diff_against_reference"] == "+ def answer(): return 'new'"
    assert llm_request.output_schema.__name__ == "_SimilarityClassificationModel"
    assert llm_request.postprocessor is not None


async def test_similarity_judge_structured_output_contract_rejects_invalid_shapes() -> None:
    llm = StubLlmProvider()
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(provider="chutes", model="google/gemma-4-31B-turbo-TEE"),
    )

    await service.judge(
        SimilarityJudgeRequest(
            batch_id=uuid4(),
            candidate_artifact_id=uuid4(),
            reference_artifact_id=uuid4(),
            candidate_miner_uid=20,
            reference_miner_uid=10,
            reference_script="def answer(): return 'old'",
            candidate_diff="+ def answer(): return 'new'",
        )
    )

    postprocessor = llm.requests[0].postprocessor
    assert postprocessor is not None

    missing_reasoning = postprocessor(_raw_similarity_response('{"classification":"duplicate"}'))
    assert missing_reasoning.ok is False
    assert missing_reasoning.retryable is True

    blank_reasoning = postprocessor(_raw_similarity_response('{"classification":"duplicate","reasoning":"   "}'))
    assert blank_reasoning.ok is False
    assert blank_reasoning.retryable is True

    extra_field = postprocessor(
        _raw_similarity_response('{"classification":"duplicate","reasoning":"same mechanism","extra":"no"}')
    )
    assert extra_field.ok is False
    assert extra_field.retryable is True

    missing_mechanism = postprocessor(
        _raw_similarity_response('{"classification":"novel","reasoning":"adds a verifier"}')
    )
    assert missing_mechanism.ok is False
    assert missing_mechanism.retryable is True

    contradictory_duplicate = postprocessor(
        _raw_similarity_response(
            '{"classification":"duplicate","reasoning":"same mechanism",'
            '"mechanism_change":"iterative retrieval"}'
        )
    )
    assert contradictory_duplicate.ok is False
    assert contradictory_duplicate.retryable is True


async def test_similarity_judge_postprocessor_accepts_duplicate_with_reasoning() -> None:
    llm = StubLlmProvider()
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(provider="chutes", model="google/gemma-4-31B-turbo-TEE"),
    )

    await service.judge(
        SimilarityJudgeRequest(
            batch_id=uuid4(),
            candidate_artifact_id=uuid4(),
            reference_artifact_id=uuid4(),
            candidate_miner_uid=20,
            reference_miner_uid=10,
            reference_script="def answer(): return 'old'",
            candidate_diff="+ def answer(): return 'new'",
        )
    )

    postprocessor = llm.requests[0].postprocessor
    assert postprocessor is not None
    result = postprocessor(
        _raw_similarity_response(
            '{"classification":"duplicate","reasoning":"Only token budget changed; '
            'no mechanism-level behavior changed."}'
        )
    )

    assert result.ok is True


async def test_similarity_judge_postprocessor_accepts_novel_with_mechanism_reasoning() -> None:
    llm = StubLlmProvider()
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(provider="chutes", model="google/gemma-4-31B-turbo-TEE"),
    )

    await service.judge(
        SimilarityJudgeRequest(
            batch_id=uuid4(),
            candidate_artifact_id=uuid4(),
            reference_artifact_id=uuid4(),
            candidate_miner_uid=20,
            reference_miner_uid=10,
            reference_script="def answer(): return 'old'",
            candidate_diff="+ def answer(): return 'new'",
        )
    )

    postprocessor = llm.requests[0].postprocessor
    assert postprocessor is not None
    result = postprocessor(
        _raw_similarity_response(
            '{"classification":"novel","reasoning":"Adds verification before synthesis.",'
            '"mechanism_change":"verification before synthesis"}'
        )
    )

    assert result.ok is True


async def test_similarity_judge_rejects_postprocessed_novel_without_mechanism_change() -> None:
    llm = SequenceLlmProvider(
        [
            _similarity_response(
                classification="novel",
                reasoning="Adds verification.",
                mechanism_change="",
            )
        ]
    )
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(provider="chutes", model="google/gemma-4-31B-turbo-TEE"),
    )

    with pytest.raises(ValidationError):
        await service.judge(
            SimilarityJudgeRequest(
                batch_id=uuid4(),
                candidate_artifact_id=uuid4(),
                reference_artifact_id=uuid4(),
                candidate_miner_uid=20,
                reference_miner_uid=10,
                reference_script="def answer(): return 'old'",
                candidate_diff="+ def answer(): return 'new'",
            )
        )


async def test_similarity_judge_rejects_structured_output_truncated_by_the_provider_cap() -> None:
    llm = SequenceLlmProvider([_similarity_response(finish_reason="length")])
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(provider="chutes", model="google/gemma-4-31B-turbo-TEE"),
    )

    with pytest.raises(RuntimeError, match="incomplete response.*length"):
        await service.judge(
            SimilarityJudgeRequest(
                batch_id=uuid4(),
                candidate_artifact_id=uuid4(),
                reference_artifact_id=uuid4(),
                candidate_miner_uid=20,
                reference_miner_uid=10,
                reference_script="def answer(): return 'old'",
                candidate_diff="+ def answer(): return 'new'",
            )
        )


async def test_similarity_judge_keeps_reasoning_effort_on_request_without_typed_thinking() -> None:
    llm = StubLlmProvider()
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
            reasoning_effort="high",
        ),
    )

    await service.judge(
        SimilarityJudgeRequest(
            batch_id=uuid4(),
            candidate_artifact_id=uuid4(),
            reference_artifact_id=uuid4(),
            candidate_miner_uid=20,
            reference_miner_uid=10,
            reference_script="def answer(): return 'old'",
            candidate_diff="+ def answer(): return 'new'",
        )
    )

    llm_request = llm.requests[0]
    assert llm_request.reasoning_effort == "high"
    assert llm_request.thinking is None


async def test_similarity_judge_tries_next_candidate_after_true_retry_exhaustion() -> None:
    retry_policy = RetryPolicy(attempts=3, initial_ms=1, max_ms=10, jitter=0.0)
    llm = SequenceLlmProvider(
        [
            LlmRetryExhaustedError("primary exhausted"),
            _similarity_response(
                prompt_tokens=31,
                completion_tokens=13,
                total_tokens=44,
                metadata={
                    "selected_provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
                    "selected_model": "google/gemma-4-31B-turbo-TEE",
                    "actual_cost_usd": 0.04,
                }
            ),
        ]
    )
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="moonshotai/Kimi-K2.5-TEE",
            fallback_models=("google/gemma-4-31B-turbo-TEE",),
            retry_policy=retry_policy,
        ),
    )

    result = await service.judge(
        SimilarityJudgeRequest(
            batch_id=uuid4(),
            candidate_artifact_id=uuid4(),
            reference_artifact_id=uuid4(),
            candidate_miner_uid=20,
            reference_miner_uid=10,
            reference_script="def answer(): return 'old'",
            candidate_diff="+ def answer(): return 'new'",
        )
    )

    assert result.classification == "novel"
    assert result.provider == "custom-openai-compatible:gemma4-cloud-run-turbo"
    assert result.model == "google/gemma-4-31B-turbo-TEE"
    assert result.judge_usage is not None
    assert result.judge_usage.prompt_tokens == 31
    assert result.judge_usage.completion_tokens == 13
    assert result.judge_usage.total_tokens == 44
    assert result.judge_usage.actual_cost_usd == pytest.approx(0.04)
    assert [request.model for request in llm.requests] == [
        "moonshotai/Kimi-K2.5-TEE",
        "google/gemma-4-31B-turbo-TEE",
    ]
    assert [request.retry_policy for request in llm.requests] == [retry_policy, retry_policy]


async def test_similarity_judge_counts_exhausted_primary_usage_before_fallback_success() -> None:
    primary_error = LlmRetryExhaustedError(
        "primary exhausted",
        response=_similarity_response(
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={
                "selected_provider": "chutes",
                "selected_model": "moonshotai/Kimi-K2.5-TEE",
                "actual_cost_usd": 0.02,
            },
        ),
    )
    llm = SequenceLlmProvider(
        [
            primary_error,
            _similarity_response(
                prompt_tokens=31,
                completion_tokens=13,
                total_tokens=44,
                metadata={
                    "selected_provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
                    "selected_model": "google/gemma-4-31B-turbo-TEE",
                    "actual_cost_usd": 0.04,
                },
            ),
        ]
    )
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="moonshotai/Kimi-K2.5-TEE",
            fallback_models=("google/gemma-4-31B-turbo-TEE",),
        ),
    )

    result = await service.judge(
        SimilarityJudgeRequest(
            batch_id=uuid4(),
            candidate_artifact_id=uuid4(),
            reference_artifact_id=uuid4(),
            candidate_miner_uid=20,
            reference_miner_uid=10,
            reference_script="def answer(): return 'old'",
            candidate_diff="+ def answer(): return 'new'",
        )
    )

    assert result.provider == "custom-openai-compatible:gemma4-cloud-run-turbo"
    assert result.model == "google/gemma-4-31B-turbo-TEE"
    assert result.judge_usage is not None
    assert result.judge_usage.call_count == 2
    assert result.judge_usage.prompt_tokens == 38
    assert result.judge_usage.completion_tokens == 17
    assert result.judge_usage.total_tokens == 55
    assert result.judge_usage.reasoning_tokens == 19
    assert result.judge_usage.actual_cost_usd == pytest.approx(0.06)
    assert [request.model for request in llm.requests] == [
        "moonshotai/Kimi-K2.5-TEE",
        "google/gemma-4-31B-turbo-TEE",
    ]


async def test_similarity_judge_preserves_retry_tokens_when_actual_cost_total_unavailable() -> None:
    primary_error = LlmRetryExhaustedError(
        "primary exhausted",
        response=_similarity_response(
            reasoning_text=None,
            reasoning_tokens=5,
            prompt_tokens=18,
            completion_tokens=9,
            total_tokens=27,
            metadata={
                "selected_provider": "chutes",
                "selected_model": "moonshotai/Kimi-K2.5-TEE",
                "billable_response_count": 2,
            },
        ),
    )
    llm = SequenceLlmProvider(
        [
            primary_error,
            _similarity_response(
                prompt_tokens=31,
                completion_tokens=13,
                total_tokens=44,
                metadata={
                    "selected_provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
                    "selected_model": "google/gemma-4-31B-turbo-TEE",
                    "actual_cost_usd": 0.04,
                },
            ),
        ]
    )
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="moonshotai/Kimi-K2.5-TEE",
            fallback_models=("google/gemma-4-31B-turbo-TEE",),
        ),
    )

    result = await service.judge(
        SimilarityJudgeRequest(
            batch_id=uuid4(),
            candidate_artifact_id=uuid4(),
            reference_artifact_id=uuid4(),
            candidate_miner_uid=20,
            reference_miner_uid=10,
            reference_script="def answer(): return 'old'",
            candidate_diff="+ def answer(): return 'new'",
        )
    )

    assert result.judge_usage is not None
    assert result.judge_usage.call_count == 3
    assert result.judge_usage.prompt_tokens == 49
    assert result.judge_usage.completion_tokens == 22
    assert result.judge_usage.total_tokens == 71
    assert result.judge_usage.reasoning_tokens == 22
    assert result.judge_usage.actual_cost_usd is None


async def test_similarity_judge_carries_failed_usage_when_final_fallback_has_no_response() -> None:
    primary_error = LlmRetryExhaustedError(
        "primary exhausted",
        response=_similarity_response(
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={
                "selected_provider": "chutes",
                "selected_model": "moonshotai/Kimi-K2.5-TEE",
                "actual_cost_usd": 0.02,
            },
        ),
    )
    fallback_error = LlmRetryExhaustedError("fallback exhausted")
    service = SimilarityJudge(
        llm_provider=SequenceLlmProvider([primary_error, fallback_error]),
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="moonshotai/Kimi-K2.5-TEE",
            fallback_models=("google/gemma-4-31B-turbo-TEE",),
        ),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await service.judge(
            SimilarityJudgeRequest(
                batch_id=uuid4(),
                candidate_artifact_id=uuid4(),
                reference_artifact_id=uuid4(),
                candidate_miner_uid=20,
                reference_miner_uid=10,
                reference_script="def answer(): return 'old'",
                candidate_diff="+ def answer(): return 'new'",
            )
        )

    assert raised.value is fallback_error
    assert raised.value.judge_usage.call_count == 1
    assert raised.value.judge_usage.prompt_tokens == 7
    assert raised.value.judge_usage.completion_tokens == 4
    assert raised.value.judge_usage.total_tokens == 11
    assert raised.value.judge_usage.reasoning_tokens == 2
    assert raised.value.judge_usage.actual_cost_usd == pytest.approx(0.02)


async def test_similarity_judge_does_not_advance_after_non_retryable_failure() -> None:
    llm = SequenceLlmProvider([RuntimeError("provider rejected request")])
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="moonshotai/Kimi-K2.5-TEE",
            fallback_models=("google/gemma-4-31B-turbo-TEE",),
        ),
    )

    with pytest.raises(RuntimeError, match="provider rejected request"):
        await service.judge(
            SimilarityJudgeRequest(
                batch_id=uuid4(),
                candidate_artifact_id=uuid4(),
                reference_artifact_id=uuid4(),
                candidate_miner_uid=20,
                reference_miner_uid=10,
                reference_script="def answer(): return 'old'",
                candidate_diff="+ def answer(): return 'new'",
            )
        )

    assert [request.model for request in llm.requests] == ["moonshotai/Kimi-K2.5-TEE"]
