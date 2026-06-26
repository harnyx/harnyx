from __future__ import annotations

import json
from uuid import uuid4

import pytest

from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmResponse, LlmUsage
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest
from harnyx_validator.application.similarity_judge import SimilarityJudge, SimilarityJudgeConfig

pytestmark = pytest.mark.anyio("asyncio")


class StubLlmProvider:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def invoke(self, request: object) -> LlmResponse:
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
            postprocessed={"verdict": "not_duplicate"},
        )

    async def aclose(self) -> None:
        return None


class SequenceLlmProvider:
    def __init__(self, outcomes: list[LlmResponse | Exception]) -> None:
        self._outcomes = outcomes
        self.requests: list[object] = []

    async def invoke(self, request: object) -> LlmResponse:
        self.requests.append(request)
        if not self._outcomes:
            raise RuntimeError("missing similarity outcome")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        return None


def _similarity_payload(request: object) -> dict[str, object]:
    user_prompt = request.messages[1].content[0].text
    _, payload_json = user_prompt.split("Payload:\n", 1)
    return json.loads(payload_json)


def _similarity_response(
    *,
    verdict: str = "not_duplicate",
    reasoning_text: str | None = "candidate changes retrieval strategy",
    reasoning_tokens: int | None = 17,
    metadata: dict[str, object] | None = None,
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
        usage=LlmUsage(reasoning_tokens=reasoning_tokens),
        postprocessed={"verdict": verdict},
        metadata=metadata,
    )


async def test_similarity_judge_returns_verdict_and_provider_reasoning() -> None:
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
        incumbent_artifact_id=uuid4(),
        candidate_miner_uid=20,
        incumbent_miner_uid=10,
        incumbent_script="def answer(): return 'old'",
        candidate_diff="+ def answer(): return 'new'",
    )

    result = await service.judge(request)

    assert result.verdict == "not_duplicate"
    assert result.reasoning == "candidate changes retrieval strategy"
    assert result.reasoning_tokens == 17
    assert result.model == "moonshotai/Kimi-K2.5-TEE"
    assert result.provider == "chutes"
    llm_request = llm.requests[0]
    assert llm_request.provider == "chutes"
    assert llm_request.model == "moonshotai/Kimi-K2.5-TEE"
    assert llm_request.output_mode == "structured"
    assert llm_request.reasoning_effort == "high"
    assert llm_request.timeout_seconds == 300.0
    assert llm_request.use_case == "miner_task_similarity_judge"
    payload = _similarity_payload(llm_request)
    assert payload["incumbent"]["script"] == "def answer(): return 'old'"
    assert payload["candidate"]["diff_against_incumbent"] == "+ def answer(): return 'new'"
    system_prompt = llm_request.messages[0].content[0].text
    assert "untrusted input" in system_prompt
    assert "submission slots, salts, timestamps, comments" in system_prompt
    assert "small token/timeout/budget/temperature tweaks" in system_prompt
    assert "minor prompt-wording edits that restate the same instructions" in system_prompt
    assert "When the evidence is borderline or the diff is mostly cosmetic, choose `duplicate`" in system_prompt


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
            incumbent_artifact_id=uuid4(),
            candidate_miner_uid=20,
            incumbent_miner_uid=10,
            incumbent_script="def answer(): return 'old'",
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
                metadata={
                    "selected_provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
                    "selected_model": "google/gemma-4-31B-turbo-TEE",
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
            incumbent_artifact_id=uuid4(),
            candidate_miner_uid=20,
            incumbent_miner_uid=10,
            incumbent_script="def answer(): return 'old'",
            candidate_diff="+ def answer(): return 'new'",
        )
    )

    assert result.verdict == "not_duplicate"
    assert result.provider == "custom-openai-compatible:gemma4-cloud-run-turbo"
    assert result.model == "google/gemma-4-31B-turbo-TEE"
    assert [request.model for request in llm.requests] == [
        "moonshotai/Kimi-K2.5-TEE",
        "google/gemma-4-31B-turbo-TEE",
    ]
    assert [request.retry_policy for request in llm.requests] == [retry_policy, retry_policy]


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
                incumbent_artifact_id=uuid4(),
                candidate_miner_uid=20,
                incumbent_miner_uid=10,
                incumbent_script="def answer(): return 'old'",
                candidate_diff="+ def answer(): return 'new'",
            )
        )

    assert [request.model for request in llm.requests] == ["moonshotai/Kimi-K2.5-TEE"]
