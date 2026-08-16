from __future__ import annotations

import json
import logging
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.llm.provider import LlmProviderError, LlmRetryExhaustedError
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
from harnyx_commons.observability.logging import ExtrasFormatter
from harnyx_validator.application import similarity_judge as similarity_judge_module
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
                        reasoning="candidate replaces the research controller",
                    ),
                ),
            ),
            usage=LlmUsage(reasoning_tokens=17),
            postprocessed=_similarity_postprocessed(
                classification="novel",
                reasoning=(
                    "The candidate replaces one tool loop with explicit planning, retrieval, "
                    "fact-table verification, and synthesis stages."
                ),
                mechanism_change="staged controller with verified fact-table evidence",
            ),
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
) -> dict[str, object]:
    status_by_classification = {
        "duplicate": "preserved",
        "near_duplicate": "localized_change",
        "notable_change": "substantial_same_root_change",
        "novel": "replaced",
    }
    status = status_by_classification[classification]
    postprocessed: dict[str, object] = {
        "classification": classification,
        "mechanism_change": mechanism_change,
        "ordinary_case_path": "answer follows the reachable test path",
        "architecture_assessment": {
            "primary_controller": {
                "status": status,
                "evidence": "controller evidence",
            },
            "evidence_state_and_flow": {
                "status": status,
                "evidence": "evidence-flow evidence",
            },
            "answer_production_path": {
                "status": status,
                "evidence": "answer-path evidence",
            },
        },
    }
    if reasoning is not None:
        postprocessed["reasoning"] = reasoning
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
        == "The candidate replaces one tool loop with explicit planning, retrieval, fact-table verification, "
        "and synthesis stages.\n"
        "Ordinary successful path: answer follows the reachable test path\n"
        "Architecture assessment:\n"
        "- Primary controller [replaced]: controller evidence\n"
        "- Evidence state and flow [replaced]: evidence-flow evidence\n"
        "- Answer-production path [replaced]: answer-path evidence\n"
        "Mechanism change: staged controller with verified fact-table evidence"
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
    output_schema = llm_request.output_schema
    assert output_schema.__name__ == "_SimilarityClassificationModel"
    json_schema = output_schema.model_json_schema()
    assert "mechanism_change" in json_schema["required"]
    assert "ordinary_case_path" in json_schema["required"]
    assert "architecture_assessment" in json_schema["required"]
    mechanism_change_schema = json_schema["properties"]["mechanism_change"]
    assert {"type": "string"} in mechanism_change_schema["anyOf"]
    assert {"type": "null"} in mechanism_change_schema["anyOf"]
    assert llm_request.postprocessor is not None


async def test_similarity_judge_forwards_configured_request_extra_for_selected_model() -> None:
    model = "deepseek-ai/DeepSeek-V4-Flash-0731-TEE"
    request_extra = {"provider": {"ignore": ["unreliable-provider"]}}
    llm = StubLlmProvider()
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="openrouter",
            model=model,
            request_extra_by_model={model: request_extra},
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

    assert llm.requests[0].extra == request_extra


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

    missing_duplicate_mechanism = postprocessor(
        _raw_similarity_response('{"classification":"duplicate","reasoning":"same mechanism"}')
    )
    assert missing_duplicate_mechanism.ok is False
    assert missing_duplicate_mechanism.retryable is True

    contradictory_duplicate = postprocessor(
        _raw_similarity_response(
            '{"classification":"duplicate","reasoning":"same mechanism",'
            '"mechanism_change":"iterative retrieval"}'
        )
    )
    assert contradictory_duplicate.ok is False
    assert contradictory_duplicate.retryable is True

    empty_duplicate_mechanism = postprocessor(
        _raw_similarity_response(
            '{"classification":"duplicate","reasoning":"same mechanism","mechanism_change":""}'
        )
    )
    assert empty_duplicate_mechanism.ok is False
    assert empty_duplicate_mechanism.retryable is True

    whitespace_duplicate_mechanism = postprocessor(
        _raw_similarity_response(
            '{"classification":"duplicate","reasoning":"same mechanism","mechanism_change":"   "}'
        )
    )
    assert whitespace_duplicate_mechanism.ok is False
    assert whitespace_duplicate_mechanism.retryable is True


@pytest.mark.parametrize(
    ("classification", "invalid_status"),
    [
        ("duplicate", "localized_change"),
        ("near_duplicate", "substantial_same_root_change"),
        ("notable_change", "localized_change"),
        ("novel", "substantial_same_root_change"),
    ],
)
async def test_similarity_judge_rejects_classifications_above_declared_architectural_evidence(
    classification: str,
    invalid_status: str,
) -> None:
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
    payload = _similarity_postprocessed(
        classification=classification,
        reasoning="classification explanation",
        mechanism_change=None if classification == "duplicate" else "reachable behavior change",
    )
    assessment = cast(dict[str, dict[str, str]], payload["architecture_assessment"])
    for dimension in assessment.values():
        dimension["status"] = invalid_status

    result = postprocessor(_raw_similarity_response(json.dumps(payload)))

    assert result.ok is False
    assert result.retryable is True


async def test_similarity_judge_postprocessor_accepts_duplicate_with_explicit_null_mechanism() -> None:
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
            json.dumps(
                _similarity_postprocessed(
                    classification="duplicate",
                    reasoning="Only token budget changed; no mechanism-level behavior changed.",
                    mechanism_change=None,
                )
            )
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
            json.dumps(
                _similarity_postprocessed(
                    classification="novel",
                    reasoning=(
                        "Replaces one tool loop with a staged "
                        "plan-retrieve-verify-synthesize controller."
                    ),
                    mechanism_change="staged controller and verified fact-table evidence",
                )
            )
        )
    )

    assert result.ok is True


async def test_similarity_judge_postprocessor_accepts_notable_change_with_same_architectural_root() -> None:
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
            json.dumps(
                _similarity_postprocessed(
                    classification="notable_change",
                    reasoning=(
                        "Adds planning and verification stages while retaining the same controller "
                        "and evidence path."
                    ),
                    mechanism_change=(
                        "substantial planning and verification stages within the same architecture"
                    ),
                )
            )
        )
    )

    assert result.ok is True


async def test_similarity_judge_tries_next_candidate_after_invalid_structured_response() -> None:
    llm = SequenceLlmProvider(
        [
            _similarity_response(
                classification="novel",
                reasoning="Adds verification.",
                mechanism_change="",
            ),
            _similarity_response(),
        ]
    )
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
            fallback_models=("moonshotai/Kimi-K3-TEE",),
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
    assert [request.model for request in llm.requests] == [
        "google/gemma-4-31B-turbo-TEE",
        "moonshotai/Kimi-K3-TEE",
    ]


async def test_similarity_judge_tries_next_candidate_after_incomplete_response() -> None:
    llm = SequenceLlmProvider(
        [
            _similarity_response(finish_reason="length"),
            _similarity_response(),
        ]
    )
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
            fallback_models=("moonshotai/Kimi-K3-TEE",),
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
    assert [request.model for request in llm.requests] == [
        "google/gemma-4-31B-turbo-TEE",
        "moonshotai/Kimi-K3-TEE",
    ]


async def test_similarity_judge_retains_tokens_after_invalid_actual_cost_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_validator.application.similarity_judge")
    llm = SequenceLlmProvider(
        [
            _similarity_response(
                reasoning_tokens=2,
                prompt_tokens=7,
                completion_tokens=4,
                total_tokens=11,
                metadata={"actual_cost_usd": True},
            ),
            _similarity_response(
                reasoning_tokens=5,
                prompt_tokens=31,
                completion_tokens=13,
                total_tokens=44,
                metadata={"actual_cost_usd": 0.04},
            ),
        ]
    )
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
            fallback_models=("moonshotai/Kimi-K3-TEE",),
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
    assert result.judge_usage is not None
    assert result.judge_usage.call_count == 2
    assert result.judge_usage.prompt_tokens == 38
    assert result.judge_usage.completion_tokens == 17
    assert result.judge_usage.total_tokens == 55
    assert result.judge_usage.reasoning_tokens == 7
    assert result.judge_usage.actual_cost_usd is None
    assert [request.model for request in llm.requests] == [
        "google/gemma-4-31B-turbo-TEE",
        "moonshotai/Kimi-K3-TEE",
    ]
    degraded_record = next(
        record
        for record in caplog.records
        if record.message == "similarity_judge.failed_candidate_actual_cost_unavailable"
    )
    assert degraded_record.data["model"] == "google/gemma-4-31B-turbo-TEE"
    assert degraded_record.data["failure_reason"] == "actual_cost_usd must be numeric when supplied"


async def test_similarity_judge_omits_unusable_failed_usage_and_still_advances() -> None:
    llm = SequenceLlmProvider(
        [
            _similarity_response(
                prompt_tokens=-1,
                completion_tokens=4,
                total_tokens=3,
                metadata={"actual_cost_usd": True},
            ),
            _similarity_response(
                prompt_tokens=31,
                completion_tokens=13,
                total_tokens=44,
                metadata={"actual_cost_usd": 0.04},
            ),
        ]
    )
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
            fallback_models=("moonshotai/Kimi-K3-TEE",),
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
    assert result.judge_usage.call_count == 1
    assert result.judge_usage.prompt_tokens == 31
    assert result.judge_usage.completion_tokens == 13
    assert result.judge_usage.total_tokens == 44
    assert result.judge_usage.actual_cost_usd == pytest.approx(0.04)
    assert len(llm.requests) == 2


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


async def test_similarity_judge_tries_next_candidate_after_non_retryable_provider_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K_SERVICE", "validator")
    caplog.set_level(logging.WARNING, logger="harnyx_validator.application.similarity_judge")
    llm = SequenceLlmProvider(
        [
            LlmProviderError(
                "http_400: context length exceeded",
                effective_provider="custom-openai-compatible:glm-cloud-run",
                effective_model="zai-org/GLM-5.2-TEE",
            ),
            _similarity_response(),
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

    assert result.classification == "novel"
    assert [request.model for request in llm.requests] == [
        "moonshotai/Kimi-K2.5-TEE",
        "google/gemma-4-31B-turbo-TEE",
    ]
    fallback_record = next(
        record for record in caplog.records if record.message == "similarity_judge.candidate_failed"
    )
    assert fallback_record.data == {
        "model": "zai-org/GLM-5.2-TEE",
        "provider": "custom-openai-compatible:glm-cloud-run",
        "exception_type": "LlmProviderError",
        "failure_reason": "http_400: context length exceeded",
    }
    formatted_record = json.loads(ExtrasFormatter().format(fallback_record))
    assert formatted_record["data"] == fallback_record.data


@pytest.mark.parametrize(
    "implementation_error",
    [RuntimeError("unexpected accounting runtime error"), ValueError("unexpected accounting value error")],
)
async def test_similarity_judge_does_not_hide_accounting_implementation_failures(
    monkeypatch: pytest.MonkeyPatch,
    implementation_error: Exception,
) -> None:
    def _raise_implementation_error(*args: object, **kwargs: object) -> None:
        raise implementation_error

    monkeypatch.setattr(
        similarity_judge_module,
        "judge_usage_from_response",
        _raise_implementation_error,
    )
    llm = SequenceLlmProvider([_similarity_response(), _similarity_response()])
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="moonshotai/Kimi-K2.5-TEE",
            fallback_models=("google/gemma-4-31B-turbo-TEE",),
        ),
    )

    with pytest.raises(type(implementation_error), match=str(implementation_error)):
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

    assert len(llm.requests) == 1


async def test_similarity_judge_does_not_hide_failed_usage_recovery_implementation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_implementation_error(*args: object, **kwargs: object) -> None:
        raise ValueError("unexpected failed usage recovery error")

    monkeypatch.setattr(
        similarity_judge_module,
        "judge_usage_without_actual_cost_from_response",
        _raise_implementation_error,
    )
    primary_response = _similarity_response(metadata={"actual_cost_usd": True})
    llm = SequenceLlmProvider(
        [
            LlmProviderError("primary response metadata failed", response=primary_response),
            _similarity_response(),
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

    with pytest.raises(ValueError, match="unexpected failed usage recovery error"):
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

    assert len(llm.requests) == 1


async def test_similarity_judge_does_not_hide_programming_failures_with_model_fallback() -> None:
    llm = SequenceLlmProvider([RuntimeError("unexpected implementation failure")])
    service = SimilarityJudge(
        llm_provider=llm,
        config=SimilarityJudgeConfig(
            provider="chutes",
            model="moonshotai/Kimi-K2.5-TEE",
            fallback_models=("google/gemma-4-31B-turbo-TEE",),
        ),
    )

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
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
