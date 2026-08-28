from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

import harnyx_commons.miner_task_scoring as miner_task_scoring
from harnyx_commons.domain.miner_task import (
    AnswerCitation,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScorerReasoning,
)
from harnyx_commons.llm.provider import BaseLlmProvider, LlmRetryExhaustedError
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment
from harnyx_commons.miner_task_scoring import (
    _MAX_RENDERED_CITATIONS,
    EvaluationScoringConfig,
    EvaluationScoringService,
    _PairwisePreference,
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


class SequenceLlmProvider:
    def __init__(self, outcomes: list[LlmResponse | Exception]) -> None:
        self._outcomes = outcomes
        self.requests: list[object] = []
        self.requests_by_side: dict[str, list[object]] = {"miner_first": [], "reference_first": []}

    async def invoke(self, request: object) -> LlmResponse:
        self.requests.append(request)
        self.requests_by_side[_pairwise_side(request)].append(request)
        if not self._outcomes:
            raise RuntimeError("missing pairwise outcome")
        outcome = self._outcomes.pop(0)
        return _resolve_llm_outcome(outcome)

    async def aclose(self) -> None:
        return None


class FastSequenceLlmProvider:
    def __init__(self, outcomes: list[LlmResponse | Exception]) -> None:
        self._outcomes = outcomes
        self.requests: list[AbstractLlmRequest] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        if not self._outcomes:
            raise RuntimeError("missing fast scoring outcome")
        return _resolve_llm_outcome(self._outcomes.pop(0))

    async def aclose(self) -> None:
        return None


class RetryWrappedSequenceLlmProvider(BaseLlmProvider):
    def __init__(self, outcomes: list[LlmResponse | Exception], *, hold_sides: tuple[str, ...] = ()) -> None:
        super().__init__(provider_label="chutes")
        self._outcomes = outcomes
        self._hold_sides = set(hold_sides)
        self._release_held = asyncio.Event()
        self.requests: list[AbstractLlmRequest] = []
        self.requests_by_side: dict[str, list[AbstractLlmRequest]] = {"miner_first": [], "reference_first": []}

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        async def _call(current_request: AbstractLlmRequest) -> LlmResponse:
            self.requests.append(current_request)
            side = _pairwise_side(current_request)
            self.requests_by_side[side].append(current_request)
            if side in self._hold_sides:
                await self._release_held.wait()
            if not self._outcomes:
                raise RuntimeError("missing pairwise outcome")
            outcome = self._outcomes.pop(0)
            return _resolve_llm_outcome(outcome)

        return await self._call_with_retry(
            request,
            call_coro=_call,
            verifier=lambda _: (True, False, None),
            policy=request.retry_policy,
        )


class ConcurrentPairwiseLlmProvider:
    def __init__(
        self,
        outcomes_by_side_and_model: dict[tuple[str, str], list[LlmResponse | Exception]],
        *,
        wait_for_both_primary: bool = False,
        hold_sides: tuple[str, ...] = (),
    ) -> None:
        self._outcomes = outcomes_by_side_and_model
        self._wait_for_both_primary = wait_for_both_primary
        self._hold_sides = set(hold_sides)
        self._both_primary_started = asyncio.Event()
        self._started_changed = asyncio.Event()
        self._release_held = asyncio.Event()
        self.requests: list[object] = []
        self.requests_by_side: dict[str, list[object]] = {"miner_first": [], "reference_first": []}
        self.started_primary_sides: set[str] = set()
        self.cancelled_sides: set[str] = set()
        self.in_flight = 0
        self.max_in_flight = 0

    def release_held_sides(self) -> None:
        self._release_held.set()

    async def wait_for_started_sides(self, sides: set[str]) -> None:
        while not sides.issubset(self.started_primary_sides):
            self._started_changed.clear()
            await self._started_changed.wait()

    async def invoke(self, request: object) -> LlmResponse:
        side = _pairwise_side(request)
        self.requests.append(request)
        self.requests_by_side[side].append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if request.model == "primary-judge" or request.model == "judge-model":
                self.started_primary_sides.add(side)
                self._started_changed.set()
                if self.started_primary_sides == {"miner_first", "reference_first"}:
                    self._both_primary_started.set()
                if self._wait_for_both_primary:
                    await self._both_primary_started.wait()
            if side in self._hold_sides:
                await self._release_held.wait()
            outcomes = self._outcomes.get((side, request.model))
            if not outcomes:
                raise RuntimeError(f"missing pairwise outcome for {side}/{request.model}")
            return _resolve_llm_outcome(outcomes.pop(0))
        except asyncio.CancelledError:
            self.cancelled_sides.add(side)
            raise
        finally:
            self.in_flight -= 1

    async def aclose(self) -> None:
        return None


def _resolve_llm_outcome(outcome: LlmResponse | Exception) -> LlmResponse:
    if type(outcome) is LlmResponse:
        return outcome
    raise outcome


def _pairwise_payload(request: object) -> dict[str, object]:
    user_prompt = request.messages[1].content[0].text
    _, payload_json = user_prompt.split("Payload:\n", 1)
    return json.loads(payload_json)


def _fast_payload(request: AbstractLlmRequest) -> dict[str, object]:
    user_prompt = request.messages[1].content[0].text
    _, payload_json = user_prompt.split("Payload:\n", 1)
    return json.loads(payload_json)


def _pairwise_side(request: object) -> str:
    payload = _pairwise_payload(request)
    first_answer = payload["answers"][0]["answer_text"]
    return "miner_first" if first_answer.startswith("Miner") else "reference_first"


def _pairwise_response(
    *,
    preferred_position: str,
    reasoning_text: str | None,
    reasoning_tokens: int | None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
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
        usage=LlmUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        postprocessed={"preferred_position": preferred_position},
        metadata=metadata,
    )


def _fast_response(
    *,
    expected_components: list[dict[str, object]],
    excessive_components: list[dict[str, object]],
    reasoning_text: str | None = None,
    reasoning_tokens: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    metadata: dict[str, object] | None = None,
) -> LlmResponse:
    return LlmResponse(
        id="fast-stub-response",
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
        postprocessed={
            "expected_components": expected_components,
            "excessive_components": excessive_components,
        },
        metadata=metadata,
    )


def _raw_structured_response(payload: object) -> LlmResponse:
    return LlmResponse(
        id="raw-structured-response",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text=json.dumps(payload)),),
                ),
            ),
        ),
        usage=LlmUsage(),
    )


async def test_scoring_service_returns_pairwise_score_directly() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    service = EvaluationScoringService(
        llm_provider=StubLlmProvider([("first", None, None), ("second", None, None)]),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(task=task, response=Response(text="Miner says 42."))

    assert score.comparison_score == pytest.approx(1.0)
    assert score.total_score == pytest.approx(1.0)


async def test_plain_pairwise_prompt_remains_free_of_structured_output_instructions() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    llm = StubLlmProvider([("first", None, None), ("second", None, None)])
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    await service.score(task=task, response=Response(text="Miner says 42."))

    payload = _pairwise_payload(llm.requests[0])

    assert "output_contract" not in payload


async def test_structured_pairwise_payload_preserves_exact_public_output_contract() -> None:
    """Future failure: judge payload must retain field meanings and constraints visible to miners."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Candidate result",
        "type": "object",
        "properties": {
            "candidate": {
                "type": "string",
                "description": "Candidate name as printed in the source.",
                "minLength": 1,
            },
            "explanation": {
                "type": "string",
                "description": "Explain the researched result and cite each material claim with [[n]].",
            },
            "scores": {
                "type": "array",
                "description": "Atomic integer scores in requested order; do not add citation syntax.",
                "minItems": 1,
                "items": {"type": "integer", "minimum": 0},
            },
        },
        "required": ["candidate", "explanation", "scores"],
        "additionalProperties": False,
    }
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Return the candidate and scores.", output_schema=schema),
        reference_answer=ReferenceAnswer(
            text='{"candidate":"A","explanation":"Supported result [[1]].","scores":[1,2]}'
        ),
    )
    llm = StubLlmProvider([("first", None, None), ("second", None, None)])
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    await service.score(
        task=task,
        response=Response(output={"candidate": "A", "explanation": "Supported result [[1]].", "scores": [1, 2]}),
    )

    payload = _pairwise_payload(llm.requests[0])
    assert payload["output_contract"] == schema
    assert "output_schema" not in payload


async def test_scoring_service_cancels_sibling_when_miner_first_exhausts() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    miner_first_error = LlmRetryExhaustedError(
        "miner first exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={"selected_provider": "chutes", "selected_model": "judge-model"},
        ),
    )
    llm = ConcurrentPairwiseLlmProvider(
        {("miner_first", "judge-model"): [miner_first_error], ("reference_first", "judge-model"): []},
        hold_sides=("reference_first",),
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await asyncio.wait_for(
            service.score(task=task, response=Response(text="Miner says 42.")),
            timeout=1.0,
        )

    assert raised.value is miner_first_error
    assert "reference_first" in llm.cancelled_sides
    assert raised.value.judge_usage.call_count == 1
    assert raised.value.evaluation_trace.scoring_judge_retry_reasons == ("interrupted",)
    assert raised.value.evaluation_trace.scoring_judge_status == "exhausted"


async def test_scoring_service_cancels_sibling_when_pairwise_fails_without_retry_exhaustion() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    miner_first_error = RuntimeError("provider rejected request")
    llm = ConcurrentPairwiseLlmProvider(
        {("miner_first", "judge-model"): [miner_first_error], ("reference_first", "judge-model"): []},
        hold_sides=("reference_first",),
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    with pytest.raises(RuntimeError, match="provider rejected request") as raised:
        await asyncio.wait_for(
            service.score(task=task, response=Response(text="Miner says 42.")),
            timeout=1.0,
        )

    assert raised.value is miner_first_error
    assert "reference_first" in llm.cancelled_sides
    assert not hasattr(raised.value, "judge_usage")
    assert raised.value.evaluation_trace.scoring_judge_retry_reasons == ("interrupted",)
    assert raised.value.evaluation_trace.scoring_judge_status == "failed"


async def test_scoring_service_runs_swapped_pairwise_calls_concurrently() -> None:
    """Future failure: serial judge calls would double pairwise scoring latency."""
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    llm = ConcurrentPairwiseLlmProvider(
        {
            ("miner_first", "judge-model"): [
                _pairwise_response(preferred_position="first", reasoning_text=None, reasoning_tokens=None)
            ],
            ("reference_first", "judge-model"): [
                _pairwise_response(preferred_position="second", reasoning_text=None, reasoning_tokens=None)
            ],
        },
        wait_for_both_primary=True,
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    result = await asyncio.wait_for(
        service.score(task=task, response=Response(text="Miner says 42.")),
        timeout=1.0,
    )

    assert result.score_breakdown.total_score == pytest.approx(1.0)
    assert llm.max_in_flight == 2
    assert len(llm.requests_by_side["miner_first"]) == 1
    assert len(llm.requests_by_side["reference_first"]) == 1
    assert result.judge_usage.call_count == 2
    assert result.evaluation_trace is not None
    assert result.evaluation_trace.scoring_judge_attempt_count == 2


async def test_scoring_service_prefers_existing_non_retryable_error_when_both_sides_fail() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    miner_first_error = LlmRetryExhaustedError(
        "miner first exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={"selected_provider": "chutes", "selected_model": "judge-model"},
        ),
    )
    non_retryable_error = RuntimeError("provider rejected request")
    llm = ConcurrentPairwiseLlmProvider(
        {
            ("miner_first", "judge-model"): [miner_first_error],
            ("reference_first", "judge-model"): [non_retryable_error],
        },
        wait_for_both_primary=True,
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    with pytest.raises(RuntimeError, match="provider rejected request") as raised:
        await asyncio.wait_for(
            service.score(task=task, response=Response(text="Miner says 42.")),
            timeout=1.0,
        )

    assert raised.value is non_retryable_error
    assert raised.value.evaluation_trace.scoring_judge_status == "failed"


async def test_scoring_service_reselects_failure_after_sibling_beats_cancellation_cleanup(monkeypatch) -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    miner_first_error = LlmRetryExhaustedError(
        "miner first exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={"selected_provider": "chutes", "selected_model": "judge-model"},
        ),
    )
    reference_first_error = RuntimeError("provider rejected request")
    llm = ConcurrentPairwiseLlmProvider(
        {
            ("miner_first", "judge-model"): [miner_first_error],
            ("reference_first", "judge-model"): [reference_first_error],
        },
        hold_sides=("reference_first",),
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )
    original_cancel = miner_task_scoring._cancel_unfinished_pairwise_tasks

    async def release_sibling_then_cancel(pair_tasks: object) -> None:
        llm.release_held_sides()
        await asyncio.sleep(0)
        await original_cancel(pair_tasks)

    monkeypatch.setattr(miner_task_scoring, "_cancel_unfinished_pairwise_tasks", release_sibling_then_cancel)

    with pytest.raises(RuntimeError, match="provider rejected request") as raised:
        await service.score(task=task, response=Response(text="Miner says 42."))

    assert raised.value is reference_first_error
    assert raised.value.evaluation_trace.scoring_judge_status == "failed"


async def test_scoring_service_reports_retry_exhausted_when_both_sides_exhaust() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    miner_first_error = LlmRetryExhaustedError(
        "miner first exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={"selected_provider": "chutes", "selected_model": "judge-model"},
        ),
    )
    reference_first_error = LlmRetryExhaustedError(
        "reference first exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=4,
            prompt_tokens=13,
            completion_tokens=7,
            total_tokens=20,
            metadata={"selected_provider": "chutes", "selected_model": "judge-model"},
        ),
    )
    llm = ConcurrentPairwiseLlmProvider(
        {
            ("miner_first", "judge-model"): [miner_first_error],
            ("reference_first", "judge-model"): [reference_first_error],
        },
        wait_for_both_primary=True,
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await asyncio.wait_for(
            service.score(task=task, response=Response(text="Miner says 42.")),
            timeout=1.0,
        )

    assert raised.value is miner_first_error
    assert raised.value.evaluation_trace.scoring_judge_status == "exhausted"


async def test_scoring_service_propagates_external_cancellation() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    llm = ConcurrentPairwiseLlmProvider(
        {("miner_first", "judge-model"): [], ("reference_first", "judge-model"): []},
        hold_sides=("miner_first", "reference_first"),
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    scoring_task = asyncio.create_task(service.score(task=task, response=Response(text="Miner says 42.")))
    await asyncio.wait_for(llm.wait_for_started_sides({"miner_first", "reference_first"}), timeout=1.0)
    scoring_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await scoring_task

    assert llm.cancelled_sides == {"miner_first", "reference_first"}


async def test_scoring_service_tries_next_candidate_after_true_retry_exhaustion() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    llm = SequenceLlmProvider(
        [
            LlmRetryExhaustedError("primary exhausted"),
            _pairwise_response(preferred_position="first", reasoning_text=None, reasoning_tokens=None),
            _pairwise_response(preferred_position="second", reasoning_text=None, reasoning_tokens=None),
        ]
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(
            provider="chutes",
            model="primary-judge",
            fallback_models=("fallback-judge",),
        ),
    )

    score = await service.score(task=task, response=Response(text="Miner says 42."))

    assert score.comparison_score == pytest.approx(1.0)
    assert [request.model for request in llm.requests_by_side["miner_first"]] == [
        "primary-judge",
        "fallback-judge",
    ]
    assert [request.model for request in llm.requests_by_side["reference_first"]] == ["primary-judge"]


async def test_scoring_service_does_not_advance_after_non_retryable_failure() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    llm = SequenceLlmProvider([RuntimeError("provider rejected request")])
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(
            provider="chutes",
            model="primary-judge",
            fallback_models=("fallback-judge",),
        ),
    )

    with pytest.raises(RuntimeError, match="provider rejected request"):
        await service.score(task=task, response=Response(text="Miner says 42."))

    assert [request.model for request in llm.requests_by_side["miner_first"]] == ["primary-judge"]
    assert "fallback-judge" not in {request.model for request in llm.requests_by_side["miner_first"]}


async def test_scoring_service_attaches_partial_judge_usage_to_second_pair_failure() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    retry_error = LlmRetryExhaustedError(
        "second pair exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={"selected_provider": "chutes", "selected_model": "judge-model", "actual_cost_usd": 0.02},
        ),
    )
    llm = SequenceLlmProvider(
        [
            _pairwise_response(
                preferred_position="first",
                reasoning_text=None,
                reasoning_tokens=3,
                prompt_tokens=11,
                completion_tokens=5,
                total_tokens=16,
                metadata={"selected_provider": "chutes", "selected_model": "judge-model", "actual_cost_usd": 0.01},
            ),
            retry_error,
        ]
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await service.score(task=task, response=Response(text="Miner says 42."))

    assert raised.value is retry_error
    assert raised.value.judge_usage.call_count == 2
    assert raised.value.judge_usage.prompt_tokens == 18
    assert raised.value.judge_usage.completion_tokens == 9
    assert raised.value.judge_usage.total_tokens == 27
    assert raised.value.judge_usage.reasoning_tokens == 5
    assert raised.value.judge_usage.actual_cost_usd == pytest.approx(0.03)


async def test_scoring_service_attaches_failed_usage_when_first_pair_exhausts() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    retry_error = LlmRetryExhaustedError(
        "first pair exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={"selected_provider": "chutes", "selected_model": "judge-model", "actual_cost_usd": 0.02},
        ),
    )
    service = EvaluationScoringService(
        llm_provider=ConcurrentPairwiseLlmProvider(
            {("miner_first", "judge-model"): [retry_error], ("reference_first", "judge-model"): []},
            hold_sides=("reference_first",),
        ),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await service.score(task=task, response=Response(text="Miner says 42."))

    assert raised.value is retry_error
    assert raised.value.judge_usage.call_count == 1
    assert raised.value.judge_usage.prompt_tokens == 7
    assert raised.value.judge_usage.completion_tokens == 4
    assert raised.value.judge_usage.total_tokens == 11
    assert raised.value.judge_usage.reasoning_tokens == 2
    assert raised.value.judge_usage.actual_cost_usd == pytest.approx(0.02)


async def test_scoring_service_preserves_retry_tokens_when_actual_cost_total_unavailable() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    retry_error = LlmRetryExhaustedError(
        "first pair exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=5,
            prompt_tokens=18,
            completion_tokens=9,
            total_tokens=27,
            metadata={
                "selected_provider": "chutes",
                "selected_model": "judge-model",
                "billable_response_count": 2,
            },
        ),
    )
    service = EvaluationScoringService(
        llm_provider=ConcurrentPairwiseLlmProvider(
            {("miner_first", "judge-model"): [retry_error], ("reference_first", "judge-model"): []},
            hold_sides=("reference_first",),
        ),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await service.score(task=task, response=Response(text="Miner says 42."))

    assert raised.value is retry_error
    assert raised.value.judge_usage.call_count == 2
    assert raised.value.judge_usage.prompt_tokens == 18
    assert raised.value.judge_usage.completion_tokens == 9
    assert raised.value.judge_usage.total_tokens == 27
    assert raised.value.judge_usage.reasoning_tokens == 5
    assert raised.value.judge_usage.actual_cost_usd is None


async def test_scoring_service_counts_exhausted_primary_usage_before_fallback_success() -> None:
    """Future failure: fallback success must not erase the exhausted attempt's cost."""
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    primary_error = LlmRetryExhaustedError(
        "primary exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={"selected_provider": "chutes", "selected_model": "primary-judge", "actual_cost_usd": 0.02},
        ),
    )
    llm = SequenceLlmProvider(
        [
            primary_error,
            _pairwise_response(
                preferred_position="first",
                reasoning_text=None,
                reasoning_tokens=3,
                prompt_tokens=11,
                completion_tokens=5,
                total_tokens=16,
                metadata={"selected_provider": "chutes", "selected_model": "fallback-judge", "actual_cost_usd": 0.01},
            ),
            _pairwise_response(
                preferred_position="second",
                reasoning_text=None,
                reasoning_tokens=4,
                prompt_tokens=13,
                completion_tokens=7,
                total_tokens=20,
                metadata={"selected_provider": "chutes", "selected_model": "primary-judge", "actual_cost_usd": 0.03},
            ),
        ]
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(
            provider="chutes",
            model="primary-judge",
            fallback_models=("fallback-judge",),
        ),
    )

    result = await service.score(task=task, response=Response(text="Miner says 42."))

    assert result.score_breakdown.comparison_score == pytest.approx(1.0)
    assert result.judge_usage.call_count == 3
    assert result.judge_usage.prompt_tokens == 31
    assert result.judge_usage.completion_tokens == 16
    assert result.judge_usage.total_tokens == 47
    assert result.judge_usage.reasoning_tokens == 9
    assert result.judge_usage.actual_cost_usd == pytest.approx(0.06)
    assert [request.model for request in llm.requests_by_side["miner_first"]] == [
        "primary-judge",
        "fallback-judge",
    ]
    assert [request.model for request in llm.requests_by_side["reference_first"]] == ["primary-judge"]


async def test_scoring_service_preserves_selected_provider_model_routes_across_fallbacks() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    primary_error = LlmRetryExhaustedError(
        "primary exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={
                "selected_provider": "chutes",
                "selected_model": "primary-judge",
                "attempts": 2,
                "retry_reasons": ("transport_error: provider transport failed",),
                "latency_ms_total": 123.45,
            },
        ),
    )
    glm_error = LlmRetryExhaustedError(
        "glm exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=3,
            prompt_tokens=11,
            completion_tokens=5,
            total_tokens=16,
            metadata={
                "selected_provider": "vertex",
                "selected_model": "zai-org/GLM-5.2-TEE",
                "attempts": 2,
                "retry_reasons": ("rate_limited: provider capacity exceeded",),
                "latency_ms_total": 300.0,
            },
        ),
    )
    llm = SequenceLlmProvider(
        [
            primary_error,
            glm_error,
            _pairwise_response(
                preferred_position="first",
                reasoning_text=None,
                reasoning_tokens=4,
                prompt_tokens=13,
                completion_tokens=7,
                total_tokens=20,
                metadata={
                    "selected_provider": "bedrock",
                    "selected_model": "moonshotai/Kimi-K2.6-TEE",
                    "attempts": 1,
                    "latency_ms_total": 100.0,
                },
            ),
            _pairwise_response(
                preferred_position="second",
                reasoning_text=None,
                reasoning_tokens=4,
                prompt_tokens=13,
                completion_tokens=7,
                total_tokens=20,
                metadata={
                    "selected_provider": "chutes",
                    "selected_model": "primary-judge",
                    "attempts": 1,
                    "latency_ms_total": 200.0,
                },
            ),
        ]
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(
            provider="chutes",
            model="primary-judge",
            fallback_models=("zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE"),
        ),
    )

    result = await service.score(task=task, response=Response(text="Miner says 42."))

    assert result.evaluation_trace is not None
    assert result.evaluation_trace.scoring_judge_selected_routes == (
        "chutes/primary-judge",
        "vertex/zai-org/GLM-5.2-TEE",
        "bedrock/moonshotai/Kimi-K2.6-TEE",
    )
    assert result.evaluation_trace.scoring_judge_attempt_count == 6
    assert result.evaluation_trace.scoring_judge_retry_count == 2
    assert result.evaluation_trace.scoring_judge_retry_reasons == ("transport_error", "rate_limited")
    assert result.evaluation_trace.scoring_judge_duration_ms == pytest.approx(723.45)
    assert result.evaluation_trace.scoring_judge_status == "ok"


async def test_scoring_service_carries_failed_usage_when_final_fallback_has_no_response() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    primary_error = LlmRetryExhaustedError(
        "primary exhausted",
        response=_pairwise_response(
            preferred_position="first",
            reasoning_text=None,
            reasoning_tokens=2,
            prompt_tokens=7,
            completion_tokens=4,
            total_tokens=11,
            metadata={"selected_provider": "chutes", "selected_model": "primary-judge", "actual_cost_usd": 0.02},
        ),
    )
    fallback_error = LlmRetryExhaustedError(
        "fallback exhausted",
        attempts=2,
        retry_reasons=("timeout while waiting for provider",),
        latency_ms_total=250.0,
    )
    service = EvaluationScoringService(
        llm_provider=ConcurrentPairwiseLlmProvider(
            {
                ("miner_first", "primary-judge"): [primary_error],
                ("miner_first", "fallback-judge"): [fallback_error],
                ("reference_first", "primary-judge"): [],
            },
            hold_sides=("reference_first",),
        ),
        config=EvaluationScoringConfig(
            provider="chutes",
            model="primary-judge",
            fallback_models=("fallback-judge",),
        ),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await service.score(task=task, response=Response(text="Miner says 42."))

    assert raised.value is fallback_error
    assert raised.value.judge_usage.call_count == 1
    assert raised.value.judge_usage.prompt_tokens == 7
    assert raised.value.judge_usage.completion_tokens == 4
    assert raised.value.judge_usage.total_tokens == 11
    assert raised.value.judge_usage.reasoning_tokens == 2
    assert raised.value.judge_usage.actual_cost_usd == pytest.approx(0.02)
    assert raised.value.evaluation_trace.scoring_judge_selected_routes == (
        "chutes/primary-judge",
        "chutes/fallback-judge",
    )
    assert raised.value.evaluation_trace.scoring_judge_attempt_count == 3
    assert raised.value.evaluation_trace.scoring_judge_retry_count == 1
    assert raised.value.evaluation_trace.scoring_judge_retry_reasons == ("timeout", "interrupted")
    assert raised.value.evaluation_trace.scoring_judge_duration_ms == pytest.approx(250.0)
    assert raised.value.evaluation_trace.scoring_judge_status == "exhausted"


async def test_scoring_service_counts_accumulated_retry_usage_from_exhausted_provider_response() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    service = EvaluationScoringService(
        llm_provider=RetryWrappedSequenceLlmProvider(
            [
                _pairwise_response(
                    preferred_position="first",
                    reasoning_text=None,
                    reasoning_tokens=2,
                    prompt_tokens=7,
                    completion_tokens=4,
                    total_tokens=11,
                    metadata={"selected_provider": "chutes", "selected_model": "judge-model", "actual_cost_usd": 0.02},
                ),
                _pairwise_response(
                    preferred_position="first",
                    reasoning_text=None,
                    reasoning_tokens=3,
                    prompt_tokens=11,
                    completion_tokens=5,
                    total_tokens=16,
                    metadata={"selected_provider": "chutes", "selected_model": "judge-model", "actual_cost_usd": 0.01},
                ),
            ],
            hold_sides=("reference_first",),
        ),
        config=EvaluationScoringConfig(
            provider="chutes",
            model="judge-model",
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0),
        ),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await service.score(task=task, response=Response(text="Miner says 42."))

    assert raised.value.judge_usage.call_count == 2
    assert raised.value.judge_usage.prompt_tokens == 18
    assert raised.value.judge_usage.completion_tokens == 9
    assert raised.value.judge_usage.total_tokens == 27
    assert raised.value.judge_usage.reasoning_tokens == 5
    assert raised.value.judge_usage.actual_cost_usd == pytest.approx(0.03)


async def test_scoring_service_records_split_pairwise_decision() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Summarize the result."),
        reference_answer=ReferenceAnswer(text="Reference summary."),
    )
    service = EvaluationScoringService(
        llm_provider=StubLlmProvider([("first", None, None), ("first", None, None)]),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(task=task, response=Response(text="Miner summary."))

    assert score.comparison_score == pytest.approx(0.5)
    assert score.total_score == pytest.approx(0.5)


async def test_scoring_service_keeps_reasoning_effort_on_request_without_typed_thinking() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    llm = StubLlmProvider([("first", None, None), ("second", None, None)])
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
            reasoning_effort="high",
        ),
    )

    await service.score(task=task, response=Response(text="Miner says 42."))

    request = llm.requests[0]
    assert request.reasoning_effort == "high"
    assert request.thinking is None


async def test_scoring_service_includes_notes_in_both_pairwise_orders() -> None:
    """Future failure: either judge ordering must retain each answer's public note."""
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Which answer is better?"),
        reference_answer=ReferenceAnswer(text="Reference answer.", note="Reference qualification."),
    )
    llm = StubLlmProvider([("first", None, None), ("second", None, None)])
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    await service.score(
        task=task,
        response=Response(text="Miner answer.", note="Miner qualification."),
    )

    forward_payload = _pairwise_payload(llm.requests[0])
    reverse_payload = _pairwise_payload(llm.requests[1])
    assert [answer["note"] for answer in forward_payload["answers"]] == [
        "Miner qualification.",
        "Reference qualification.",
    ]
    assert [answer["note"] for answer in reverse_payload["answers"]] == [
        "Reference qualification.",
        "Miner qualification.",
    ]


def test_structured_object_renders_deterministically_in_judge_answer_text() -> None:
    rendered = miner_task_scoring._render_answer_for_judge(
        position="first",
        answer=Response(output={"z": [1, None], "a": True}),
    )

    assert rendered["answer_text"] == '{"a":true,"z":[1,null]}'


def test_structured_string_renders_as_json_string_not_legacy_text() -> None:
    rendered = miner_task_scoring._render_answer_for_judge(
        position="first",
        answer=Response(output="structured string"),
    )

    assert rendered["answer_text"] == '"structured string"'


def test_evaluation_scoring_config_default_timeout_is_300_seconds() -> None:
    config = EvaluationScoringConfig(provider="chutes", model="judge-model")

    assert config.timeout_seconds == pytest.approx(300.0)


async def test_scoring_service_preserves_positional_citations_and_caps_without_remapping() -> None:
    """Future failure: duplicate and unresolved positions must remain addressable by [[n]]."""
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Which answer is better?"),
        reference_answer=ReferenceAnswer(text="Reference answer."),
    )
    llm = StubLlmProvider([("first", None, None), ("second", None, None)])
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    citations = [
        AnswerCitation(url="https://same-source.example.com", title="Title A", note="Note A"),
        AnswerCitation(url="https://same-source.example.com", title="Title A", note="Note A"),
        None,
        AnswerCitation(url="https://same-source.example.com", title="Title B", note="Note B"),
        AnswerCitation(url="https://miner.example.com", note="Miner note"),
    ]
    citations.extend(
        AnswerCitation(url=f"https://extra-{index}.example.com") for index in range(_MAX_RENDERED_CITATIONS + 3)
    )

    await service.score(task=task, response=Response(text="Miner answer.", citations=tuple(citations)))

    payload = _pairwise_payload(llm.requests[0])
    validated_citations = payload["answers"][0]["validated_citations"]
    assert len(validated_citations) == _MAX_RENDERED_CITATIONS
    assert validated_citations[:4] == [
        {"url": "https://same-source.example.com", "title": "Title A", "note": "Note A"},
        {"url": "https://same-source.example.com", "title": "Title A", "note": "Note A"},
        None,
        {"url": "https://same-source.example.com", "title": "Title B", "note": "Note B"},
    ]
    assert (
        validated_citations.count({"url": "https://same-source.example.com", "title": "Title A", "note": "Note A"}) == 2
    )


async def test_scoring_service_persists_joined_reasoning_trace_and_token_total() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Which answer is better?"),
        reference_answer=ReferenceAnswer(text="Reference answer."),
    )
    service = EvaluationScoringService(
        llm_provider=StubLlmProvider(
            [
                ("first", "Miner-first reasoning trace.", 11),
                ("second", "Reference-first reasoning trace.", 7),
            ]
        ),
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(task=task, response=Response(text="Miner answer."))

    assert score.reasoning == ScorerReasoning(
        text="Miner-first reasoning trace.\n\n---\n\nReference-first reasoning trace.",
        reasoning_tokens=18,
    )


def test_pairwise_preference_accepts_chosen_answer_alias() -> None:
    parsed = _PairwisePreference.model_validate({"chosen_answer": "first"})

    assert parsed.preferred_position == "first"


async def test_scoring_service_accepts_chosen_answer_alias_from_live_shape() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What is the answer?"),
        reference_answer=ReferenceAnswer(text="The answer is 42."),
    )
    service = EvaluationScoringService(
        llm_provider=AliasStubLlmProvider(["first", "second"]),
        config=EvaluationScoringConfig(provider="vertex-maas", model="judge-model"),
    )

    score = await service.score(task=task, response=Response(text="Miner says 42."))

    assert score.comparison_score == pytest.approx(1.0)
    assert score.total_score == pytest.approx(1.0)


async def test_fast_scoring_uses_one_citation_free_component_judgment() -> None:
    """Future failure: fast scoring must stay correctness-only and one-directional."""
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Name all four required values.", fast=True),
        reference_answer=ReferenceAnswer(
            text="Alpha, beta, gamma, delta.",
            note="The order is not significant.",
            citations=(AnswerCitation(url="https://reference.example"),),
        ),
    )
    llm = FastSequenceLlmProvider(
        [
            _fast_response(
                expected_components=[
                    {"component_id": "alpha", "is_correct": True},
                    {"component_id": "beta", "is_correct": True},
                    {"component_id": "gamma", "is_correct": False},
                    {"component_id": "delta", "is_correct": False},
                ],
                excessive_components=[{"component_id": "epsilon"}],
                reasoning_text="Component comparison complete.",
                reasoning_tokens=9,
            )
        ]
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(
        task=task,
        response=Response(
            text="Alpha, beta, epsilon.",
            note="Gamma may also apply.",
            citations=(AnswerCitation(url="https://miner.example"),),
        ),
    )

    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert tuple(message.role for message in request.messages) == ("system", "user")
    assert request.use_case == "miner_task_fast_correctness_judge"
    assert request.output_schema is FastJudgeAssessment
    payload = _fast_payload(request)
    assert payload == {
        "query": "Name all four required values.",
        "reference_answer": "Alpha, beta, gamma, delta.",
        "reference_note": "The order is not significant.",
        "miner_answer": "Alpha, beta, epsilon.",
        "miner_note": "Gamma may also apply.",
    }
    assert "citation" not in json.dumps(payload).lower()
    assert score.comparison_score == pytest.approx(round(4 / 7, 6))
    assert score.total_score == pytest.approx(round(4 / 7, 6))
    assert score.scoring_version == "deepsearchqa-f1-v1"
    assert score.reasoning == ScorerReasoning(text="Component comparison complete.", reasoning_tokens=9)


async def test_fast_scoring_returns_zero_when_no_expected_component_is_correct() -> None:
    """Future failure: zero correct components must produce a deterministic zero score."""
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Name the first two planets from the Sun.", fast=True),
        reference_answer=ReferenceAnswer(text="Mercury and Venus."),
    )
    llm = FastSequenceLlmProvider(
        [
            _fast_response(
                expected_components=[
                    {"component_id": "Mercury", "is_correct": False},
                    {"component_id": "Venus", "is_correct": False},
                ],
                excessive_components=[{"component_id": "Mars"}],
            )
        ]
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(task=task, response=Response(text="Mars."))

    assert score.comparison_score == 0.0
    assert score.total_score == 0.0
    assert score.scoring_version == "deepsearchqa-f1-v1"


async def test_fast_scoring_preserves_structured_answer_contract_and_note_rules() -> None:
    """Future failure: structured fast answers and public notes must retain their exact roles."""
    task = MinerTask(
        task_id=uuid4(),
        query=Query(
            text="Return the capital.",
            output_schema={
                "type": "object",
                "properties": {"capital": {"type": "string"}},
                "required": ["capital"],
                "additionalProperties": False,
            },
            fast=True,
        ),
        reference_answer=ReferenceAnswer(text='{"capital":"Paris"}', note="Use the English city name."),
    )
    llm = FastSequenceLlmProvider(
        [
            _fast_response(
                expected_components=[{"component_id": "capital", "is_correct": True}],
                excessive_components=[],
            )
        ]
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )

    score = await service.score(
        task=task,
        response=Response(output={"capital": "Paris"}, note="No regional qualifier is needed."),
    )

    request = llm.requests[0]
    payload = _fast_payload(request)
    assert payload["output_contract"] == task.query.output_schema
    assert payload["miner_answer"] == '{"capital":"Paris"}'
    system_prompt = request.messages[0].content[0].text
    assert "Use `reference_note` only to qualify" in system_prompt
    assert "`miner_note` is optional supplementary correctness content" in system_prompt
    assert "cannot replace a missing component" in system_prompt
    assert request.messages[1].content[0].text.startswith("Payload:\n{")
    assert score.total_score == pytest.approx(1.0)


async def test_fast_scoring_schema_rejects_invalid_component_sets_with_retry_feedback() -> None:
    """Future failure: malformed component judgments must enter structured-output recovery."""
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Return the answer.", fast=True),
        reference_answer=ReferenceAnswer(text="Expected."),
    )
    llm = FastSequenceLlmProvider(
        [
            _fast_response(
                expected_components=[{"component_id": "expected", "is_correct": True}],
                excessive_components=[],
            )
        ]
    )
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(provider="chutes", model="judge-model"),
    )
    await service.score(task=task, response=Response(text="Expected."))
    request = llm.requests[0]
    invalid_payloads = (
        {
            "expected_components": [{"component_id": "expected", "is_correct": True, "extra": "no"}],
            "excessive_components": [],
        },
        {
            "expected_components": [{"component_id": "expected", "is_correct": 1}],
            "excessive_components": [],
        },
        {
            "expected_components": [{"component_id": "   ", "is_correct": True}],
            "excessive_components": [],
        },
        {
            "expected_components": [
                {"component_id": "Expected", "is_correct": True},
                {"component_id": " expected ", "is_correct": False},
            ],
            "excessive_components": [],
        },
        {
            "expected_components": [{"component_id": "expected", "is_correct": True}],
            "excessive_components": [
                {"component_id": "Extra"},
                {"component_id": " extra "},
            ],
        },
        {
            "expected_components": [{"component_id": "Expected", "is_correct": True}],
            "excessive_components": [{"component_id": " expected "}],
        },
    )

    for invalid_payload in invalid_payloads:
        result = request.postprocessor(_raw_structured_response(invalid_payload))
        assert result.ok is False
        assert result.retryable is True
        assert result.recovery is not None
        assert result.recovery.kind == "retry_with_feedback"


async def test_fast_scoring_reuses_fallback_usage_trace_and_terminal_failure() -> None:
    """Future failure: fast mode must retain scorer fallback accounting and exhaustion."""
    primary_error = LlmRetryExhaustedError(
        "primary exhausted",
        response=_fast_response(
            expected_components=[{"component_id": "answer", "is_correct": False}],
            excessive_components=[],
            prompt_tokens=7,
            completion_tokens=3,
            total_tokens=10,
            metadata={"selected_provider": "chutes", "selected_model": "primary-judge"},
        ),
        attempts=2,
        retry_reasons=("structured output invalid",),
    )
    fallback_response = _fast_response(
        expected_components=[{"component_id": "answer", "is_correct": True}],
        excessive_components=[],
        prompt_tokens=11,
        completion_tokens=4,
        total_tokens=15,
        metadata={"selected_provider": "chutes", "selected_model": "fallback-judge"},
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Return the answer.", fast=True),
        reference_answer=ReferenceAnswer(text="Answer."),
    )
    llm = FastSequenceLlmProvider([primary_error, fallback_response])
    service = EvaluationScoringService(
        llm_provider=llm,
        config=EvaluationScoringConfig(
            provider="chutes",
            model="primary-judge",
            fallback_models=("fallback-judge",),
        ),
    )

    result = await service.score(task=task, response=Response(text="Answer."))

    assert [request.model for request in llm.requests] == ["primary-judge", "fallback-judge"]
    assert result.judge_usage.call_count == 2
    assert result.judge_usage.total_tokens == 25
    assert result.evaluation_trace is not None
    assert result.evaluation_trace.scoring_judge_selected_routes == (
        "chutes/primary-judge",
        "chutes/fallback-judge",
    )
    assert result.evaluation_trace.scoring_judge_status == "ok"

    terminal_llm = FastSequenceLlmProvider(
        [LlmRetryExhaustedError("primary exhausted"), LlmRetryExhaustedError("fallback exhausted")]
    )
    terminal_service = EvaluationScoringService(
        llm_provider=terminal_llm,
        config=EvaluationScoringConfig(
            provider="chutes",
            model="primary-judge",
            fallback_models=("fallback-judge",),
        ),
    )
    with pytest.raises(LlmRetryExhaustedError, match="fallback exhausted"):
        await terminal_service.score(task=task, response=Response(text="Answer."))
