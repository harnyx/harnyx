from __future__ import annotations

import json
from uuid import uuid4

import pytest

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


def _similarity_payload(request: object) -> dict[str, object]:
    user_prompt = request.messages[1].content[0].text
    _, payload_json = user_prompt.split("Payload:\n", 1)
    return json.loads(payload_json)


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
