from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from harnyx_commons.domain_tweak_generation import (
    DomainTweakAdkPhase,
    DomainTweakAdkRunConfig,
    DomainTweakAdkRunner,
    DomainTweakAdkTurn,
    DomainTweakBatchGenerationConfig,
    DomainTweakBatchGenerationPipeline,
    DomainTweakReferenceAnswerPhasePolicy,
)
from harnyx_commons.domain_tweak_generation.batch_pipeline import _question_attempt_windows
from harnyx_commons.domain_tweak_generation.types import DomainTweakAdkEventSummary
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_generation import DomainTweakPairInput

pytestmark = pytest.mark.anyio("asyncio")

_TASK_IDS = (
    UUID("00000000-0000-0000-0000-000000000101"),
    UUID("00000000-0000-0000-0000-000000000102"),
)


async def test_batch_pipeline_selects_n_reviewed_questions_then_finalizes_only_selected() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            _question_payload(1),
            _question_payload(2),
            _review_payload(form_match=True),
            _review_payload(form_match=True),
            _answer_payload(1),
            _answer_payload(2),
        )
    )
    task_ids = iter(_TASK_IDS)
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
        task_id_factory=lambda: next(task_ids),
    )

    result = await pipeline.generate_batch(_pair_inputs(3), DomainTweakBatchGenerationConfig(target_count=2))

    assert not result.underfilled
    assert [item.pair_input.pair_id for item in result.selected_questions] == ["pair-001", "pair-002"]
    assert [item.task.task_id for item in result.finalized_tasks] == list(_TASK_IDS)
    assert result.rejected_attempts == ()
    assert result.failed_finalizations == ()
    assert executor.phases == [
        "question_generation",
        "question_generation",
        "form_review",
        "form_review",
        "reference_answer",
        "reference_answer",
    ]
    assert result.tool_usage.llm.call_count == 6
    assert result.tool_usage.search_tool.call_count == 6


async def test_batch_pipeline_skips_duplicate_reviewed_question_text_before_selection() -> None:
    duplicate_question = "Which players meet all constraints in domain 1?"
    executor = _FakeTurnExecutor(
        responses=(
            _question_payload(1),
            _question_payload(99, question=f"  {duplicate_question.upper()}  "),
            _review_payload(form_match=True),
            _review_payload(form_match=True),
            _question_payload(2),
            _review_payload(form_match=True),
            _answer_payload(1),
            _answer_payload(2),
        )
    )
    task_ids = iter(_TASK_IDS)
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
        task_id_factory=lambda: next(task_ids),
    )

    result = await pipeline.generate_batch(_pair_inputs(3), DomainTweakBatchGenerationConfig(target_count=2))

    assert not result.underfilled
    assert [item.pair_input.pair_id for item in result.selected_questions] == ["pair-001", "pair-003"]
    assert [item.pair_input.pair_id for item in result.rejected_attempts] == ["pair-002"]
    assert [item.task.query.text for item in result.finalized_tasks] == [
        "Which players meet all constraints in domain 1?",
        "Which players meet all constraints in domain 2?",
    ]
    assert executor.phases == [
        "question_generation",
        "question_generation",
        "form_review",
        "form_review",
        "question_generation",
        "form_review",
        "reference_answer",
        "reference_answer",
    ]
    assert result.tool_usage.llm.call_count == 8


async def test_batch_pipeline_reports_underfill_after_hard_attempt_cap() -> None:
    executor = _FakeTurnExecutor(responses=tuple(_no_generate_payload() for _ in range(12)))
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )

    result = await pipeline.generate_batch(_pair_inputs(12), DomainTweakBatchGenerationConfig(target_count=3))

    assert result.underfilled
    assert result.selected_questions == ()
    assert result.finalized_tasks == ()
    assert len(result.rejected_attempts) == 12
    assert result.failed_finalizations == ()
    assert set(executor.phases) == {"question_generation"}
    assert result.tool_usage.llm.call_count == 12


async def test_batch_pipeline_skips_reference_answer_phase_for_partial_question_underfill() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            _question_payload(1),
            _no_generate_payload(),
            _no_generate_payload(),
            _review_payload(form_match=True),
        )
    )
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )

    result = await pipeline.generate_batch(_pair_inputs(3), DomainTweakBatchGenerationConfig(target_count=3))

    assert result.underfilled
    assert [item.pair_input.pair_id for item in result.selected_questions] == ["pair-001"]
    assert result.finalized_tasks == ()
    assert len(result.rejected_attempts) == 2
    assert result.failed_finalizations == ()
    assert executor.phases == [
        "question_generation",
        "question_generation",
        "question_generation",
        "form_review",
    ]
    assert result.tool_usage.llm.call_count == 4


async def test_batch_pipeline_applies_phase_specific_defaults() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            _question_payload(1),
            _review_payload(form_match=True),
            _answer_payload(1),
        )
    )
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview", max_retries=99),
        runner=DomainTweakAdkRunner(turn_executor=executor),
        task_id_factory=lambda: _TASK_IDS[0],
    )

    result = await pipeline.generate_batch(_pair_inputs(1), DomainTweakBatchGenerationConfig(target_count=1))

    assert len(result.finalized_tasks) == 1
    observed = list(zip(executor.phases, executor.configs, strict=True))
    assert [
        (
            phase,
            config.max_retries,
            config.phase_timeout_seconds,
            config.soft_timeout_seconds,
            config.soft_timeout_interval_seconds,
        )
        for phase, config in observed
    ] == [
        ("question_generation", 2, 600.0, None, None),
        ("form_review", 2, 600.0, None, None),
        ("reference_answer", 1, 1800.0, 600.0, 300.0),
    ]
    assert DomainTweakReferenceAnswerPhasePolicy().invocation_attempt_cap(10) == 50


async def test_batch_pipeline_retries_a2_invocation_error_with_fresh_answer_phase() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            _question_payload(1),
            _review_payload(form_match=True),
            RuntimeError("transient ADK failure"),
            _answer_payload(1),
        )
    )
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
        task_id_factory=lambda: _TASK_IDS[0],
    )

    result = await pipeline.generate_batch(_pair_inputs(1), DomainTweakBatchGenerationConfig(target_count=1))

    assert not result.underfilled
    assert len(result.finalized_tasks) == 1
    assert result.failed_finalizations == ()
    assert executor.phases == [
        "question_generation",
        "form_review",
        "reference_answer",
        "reference_answer",
    ]
    assert result.tool_usage.llm.call_count == 3


async def test_batch_pipeline_retries_only_failed_a2_finalizations_until_n_tasks_are_finalized() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            _question_payload(1),
            _question_payload(2),
            _review_payload(form_match=True),
            _review_payload(form_match=True),
            _invalid_answer_payload(1),
            _answer_payload(2),
            _answer_payload(1),
        )
    )
    task_ids = iter(_TASK_IDS)
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
        task_id_factory=lambda: next(task_ids),
    )

    config = DomainTweakBatchGenerationConfig(
        target_count=2,
        reference_answer_policy=DomainTweakReferenceAnswerPhasePolicy(invocation_retries_per_answer=0),
    )

    result = await pipeline.generate_batch(_pair_inputs(2), config)

    assert not result.underfilled
    assert result.failed_finalizations == ()
    assert [item.task.query.text for item in result.finalized_tasks] == [
        "Which players meet all constraints in domain 1?",
        "Which players meet all constraints in domain 2?",
    ]
    assert result.reference_answer_finalization_attempt_count == 3
    assert result.reference_answer_retry_attempt_count == 1
    assert result.reference_answer_retry_round_count == 1
    assert executor.reference_answer_indexes == [1, 2, 1]


async def test_batch_pipeline_reports_underfill_after_failed_a2_retry_bound() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            _question_payload(1),
            _review_payload(form_match=True),
            _invalid_answer_payload(1),
            _invalid_answer_payload(1),
            _invalid_answer_payload(1),
            _invalid_answer_payload(1),
        )
    )
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )
    config = DomainTweakBatchGenerationConfig(
        target_count=1,
        reference_answer_policy=DomainTweakReferenceAnswerPhasePolicy(
            failed_finalization_retries_per_batch_item=3,
            invocation_retries_per_answer=0,
            hard_answer_attempt_cap_multiplier=4,
        ),
    )

    result = await pipeline.generate_batch(_pair_inputs(1), config)

    assert result.underfilled
    assert len(result.failed_finalizations) == 1
    failed_finalization = result.failed_finalizations[0]
    assert len(failed_finalization.reference_answer_results) == 4
    assert sum(len(item.attempts) for item in failed_finalization.reference_answer_results) == 8
    assert failed_finalization.tool_usage.llm.call_count == 4
    assert result.reference_answer_finalization_attempt_count == 4
    assert result.reference_answer_retry_attempt_count == 3
    assert result.reference_answer_retry_round_count == 3


async def test_batch_pipeline_stops_failed_a2_retries_when_invocation_budget_is_exhausted() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            _question_payload(1),
            _review_payload(form_match=True),
            _invalid_answer_payload(1),
        )
    )
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )
    config = DomainTweakBatchGenerationConfig(
        target_count=1,
        reference_answer_policy=DomainTweakReferenceAnswerPhasePolicy(
            failed_finalization_retries_per_batch_item=3,
            invocation_retries_per_answer=0,
            hard_answer_attempt_cap_multiplier=1,
        ),
    )

    result = await pipeline.generate_batch(_pair_inputs(1), config)

    assert result.underfilled
    assert len(result.failed_finalizations) == 1
    assert result.reference_answer_finalization_attempt_count == 1
    assert result.reference_answer_retry_attempt_count == 0
    assert result.reference_answer_retry_round_count == 1


async def test_batch_pipeline_runs_question_and_a2_with_fixed_bounded_concurrency() -> None:
    executor = _ConcurrencyTrackingExecutor()
    pipeline = DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )

    result = await pipeline.generate_batch(_pair_inputs(9), DomainTweakBatchGenerationConfig(target_count=9))

    assert not result.underfilled
    assert len(result.finalized_tasks) == 9
    assert [item.pair_input.pair_id for item in result.selected_questions] == [
        f"pair-{index:03d}" for index in range(1, 10)
    ]
    assert [item.task.query.text for item in result.finalized_tasks] == [
        f"Which players meet all constraints in domain {index}?" for index in range(1, 10)
    ]
    assert executor.max_active["question_generation"] == 8
    assert executor.max_active["reference_answer"] == 8
    assert max(executor.max_active.values()) == 8


def test_question_attempt_windows_are_deterministic_contiguous_passes() -> None:
    windows = _question_attempt_windows(
        _pair_inputs(40),
        DomainTweakBatchGenerationConfig(target_count=10).question_policy,
        target_count=10,
    )

    assert [[item.pair_id for item in window] for window in windows] == [
        [f"pair-{index:03d}" for index in range(1, 31)],
        [f"pair-{index:03d}" for index in range(31, 35)],
        [f"pair-{index:03d}" for index in range(35, 38)],
        [f"pair-{index:03d}" for index in range(38, 41)],
    ]


class _FakeTurnExecutor:
    def __init__(self, *, responses: tuple[str | BaseException, ...]) -> None:
        self._responses = list(responses)
        self.phases: list[DomainTweakAdkPhase] = []
        self.configs: list[DomainTweakAdkRunConfig] = []
        self.reference_answer_indexes: list[int] = []

    async def __call__(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        attempt_index: int,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
    ) -> DomainTweakAdkTurn:
        self.phases.append(phase)
        self.configs.append(config)
        if phase == "reference_answer":
            self.reference_answer_indexes.append(_index_from_prompt(prompt))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return DomainTweakAdkTurn(
            final_text=response,
            events=(
                DomainTweakAdkEventSummary(
                    is_final_response=True,
                    usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    function_call_names=("google_search_agent",),
                ),
            ),
        )


class _ConcurrencyTrackingExecutor:
    def __init__(self) -> None:
        self.active: dict[DomainTweakAdkPhase, int] = {
            "question_generation": 0,
            "form_review": 0,
            "reference_answer": 0,
        }
        self.max_active: dict[DomainTweakAdkPhase, int] = {
            "question_generation": 0,
            "form_review": 0,
            "reference_answer": 0,
        }
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        attempt_index: int,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
    ) -> DomainTweakAdkTurn:
        _ = (attempt_index, config, agent_instruction)
        async with self._lock:
            self.active[phase] += 1
            self.max_active[phase] = max(self.max_active[phase], self.active[phase])
        try:
            await asyncio.sleep(0.01)
        finally:
            async with self._lock:
                self.active[phase] -= 1
        return DomainTweakAdkTurn(
            final_text=_payload_for_phase_prompt(phase, prompt),
            events=(
                DomainTweakAdkEventSummary(
                    is_final_response=True,
                    usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    function_call_names=("google_search_agent",),
                ),
            ),
        )


def _payload_for_phase_prompt(phase: DomainTweakAdkPhase, prompt: str) -> str:
    index = _index_from_prompt(prompt)
    if phase == "question_generation":
        return _question_payload(index)
    if phase == "form_review":
        return _review_payload(form_match=True)
    return _answer_payload(index)


def _index_from_prompt(prompt: str) -> int:
    for line in prompt.splitlines():
        if line.startswith("Pair id: pair-"):
            return int(line.rsplit("-", maxsplit=1)[-1])
    marker = "Which players meet all constraints in domain "
    if marker in prompt:
        suffix = prompt.split(marker, maxsplit=1)[1]
        digits = []
        for char in suffix:
            if not char.isdigit():
                break
            digits.append(char)
        if digits:
            return int("".join(digits))
    raise AssertionError(f"could not extract pair index from prompt: {prompt[:200]}")


def _pair_inputs(count: int) -> tuple[DomainTweakPairInput, ...]:
    return tuple(
        DomainTweakPairInput(
            pair_id=f"pair-{index:03d}",
            deepsearchqa_form_target="Which films meet all constraints?",
            deepresearch9k_domain_target=f"Domain target {index}",
            timestamp=datetime(2026, 6, 23, tzinfo=UTC),
        )
        for index in range(1, count + 1)
    )


def _question_payload(index: int, *, question: str | None = None) -> str:
    return json.dumps(
        {
            "question": question or f"Which players meet all constraints in domain {index}?",
            "short_answer": f"Ada Example {index}; Ben Example {index}",
            "solution_plan": "- Find candidate pool\n- Intersect constraints",
        }
    )


def _review_payload(*, form_match: bool) -> str:
    return json.dumps(
        {
            "form_match": form_match,
            "false_premise_status": "none",
            "reviewer_feedback": "Form preserved." if form_match else "Form drifted.",
            "retry_recommended": not form_match,
        }
    )


def _answer_payload(index: int) -> str:
    return json.dumps(
        {
            "question": f"Which players meet all constraints in domain {index}?",
            "premise_assessment": "The premise is supported by the cited table.",
            "reference_answer": {
                "text": f"Ada Example {index} and Ben Example {index} meet all constraints.",
                "citations": [
                    {
                        "url": "https://example.com/table",
                        "title": "Official table",
                        "note": "Lists both qualifying players.",
                    }
                ],
            },
        }
    )


def _invalid_answer_payload(index: int) -> str:
    return f"reference answer for domain {index} without json"


def _no_generate_payload() -> str:
    return json.dumps(
        {
            "no_generate": True,
            "reason": "No grounded domain evidence supports the original form.",
            "retry_recommended": False,
        }
    )
