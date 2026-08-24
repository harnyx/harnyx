from __future__ import annotations

import json
from uuid import uuid4

import pytest

from harnyx_commons.domain.miner_task import (
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
)
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmResponse, LlmUsage
from harnyx_commons.miner_task_scoring import (
    EvaluationScoringConfig,
    EvaluationScoringService,
)

pytestmark = pytest.mark.anyio("asyncio")


class StubLlmProvider:
    def __init__(
        self,
        pairwise_results: list[tuple[str, str | None, int | None]],
    ) -> None:
        self._pairwise_results = pairwise_results
        self.requests: list[object] = []

    async def invoke(self, request: object) -> object:
        self.requests.append(request)
        if not self._pairwise_results:
            raise RuntimeError("missing pairwise preference")
        preferred_position, reasoning_text, reasoning_tokens = self._pairwise_results.pop(0)
        return _pairwise_response(
            preferred_position=preferred_position,
            reasoning_text=reasoning_text,
            reasoning_tokens=reasoning_tokens,
        )

    async def aclose(self) -> None:
        return None


class AliasStubLlmProvider:
    def __init__(self, chosen_answers: list[str]) -> None:
        self._chosen_answers = chosen_answers
        self.requests: list[object] = []

    async def invoke(self, request: object) -> object:
        self.requests.append(request)
        if not self._chosen_answers:
            raise RuntimeError("missing pairwise preference")
        chosen_answer = self._chosen_answers.pop(0)
        return LlmResponse(
            id="stub-response",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(),
                        reasoning=None,
                    ),
                ),
            ),
            usage=LlmUsage(),
            postprocessed={"chosen_answer": chosen_answer},
        )

    async def aclose(self) -> None:
        return None


def _pairwise_payload(request: object) -> dict[str, object]:
    user_prompt = request.messages[1].content[0].text
    _, payload_json = user_prompt.split("Payload:\n", 1)
    return json.loads(payload_json)


def _pairwise_response(
    *,
    preferred_position: str,
    reasoning_text: str | None,
    reasoning_tokens: int | None,
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
        postprocessed={"preferred_position": preferred_position},
    )


async def test_scoring_service_includes_retry_policy_on_pairwise_requests() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    retry_policy = RetryPolicy(attempts=6, initial_ms=30_000, max_ms=300_000, jitter=0.2)
    llm = StubLlmProvider([("first", None, None), ("second", None, None)])
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model", retry_policy=retry_policy),
    )

    await service.score(task=task, response=Response(text="Miner says 42."))

    assert tuple(request.retry_policy for request in llm.requests) == (retry_policy, retry_policy)
