from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain_tweak_generation import (
    DomainTweakAdkPhaseResult,
    DomainTweakBatchGenerationConfig,
    DomainTweakBatchGenerationResult,
    DomainTweakFailedFinalization,
    DomainTweakFinalizedTask,
    DomainTweakMinerTaskDatasetBuilder,
    DomainTweakReviewedQuestion,
    finalized_tasks_from_domain_tweak_result,
)
from harnyx_commons.miner_task_generation import (
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionCandidate,
    MinerTaskDatasetRequest,
    MinerTaskModelSpec,
)

pytestmark = pytest.mark.anyio("asyncio")

_BATCH_ID = UUID("00000000-0000-0000-0000-000000000042")
_BATCH_CREATED_AT = datetime(2026, 6, 26, 5, 30, tzinfo=UTC)


async def test_domain_tweak_dataset_builder_returns_finalized_tasks() -> None:
    pair_source = _RecordingPairSource(_pair_inputs(8))
    batch_pipeline = _FakeBatchPipeline(finalized_count=2)
    builder = DomainTweakMinerTaskDatasetBuilder(pair_source=pair_source, batch_pipeline=batch_pipeline)

    tasks = await builder.build(_dataset_request(minimum_task_total=2))

    assert len(tasks) == 2
    assert [task.query.text for task in tasks] == ["Question 1?", "Question 2?"]
    assert pair_source.calls == [(_BATCH_ID, _BATCH_CREATED_AT, 8)]
    assert batch_pipeline.observed_target_count == 2
    assert [item.pair_id for item in batch_pipeline.observed_pair_inputs] == [
        f"pair-{index:03d}" for index in range(1, 9)
    ]


async def test_domain_tweak_dataset_builder_exposes_generation_result() -> None:
    pair_source = _RecordingPairSource(_pair_inputs(4))
    batch_pipeline = _FakeBatchPipeline(finalized_count=1)
    builder = DomainTweakMinerTaskDatasetBuilder(pair_source=pair_source, batch_pipeline=batch_pipeline)

    result = await builder.build_with_result(_dataset_request(minimum_task_total=1))

    assert isinstance(result, DomainTweakBatchGenerationResult)
    assert len(result.finalized_tasks) == 1


async def test_domain_tweak_dataset_builder_exposes_underfilled_generation_result() -> None:
    pair_source = _RecordingPairSource(_pair_inputs(8))
    batch_pipeline = _FakeBatchPipeline(finalized_count=1, underfilled=True)
    builder = DomainTweakMinerTaskDatasetBuilder(pair_source=pair_source, batch_pipeline=batch_pipeline)

    result = await builder.build_with_result(_dataset_request(minimum_task_total=2))

    assert result.underfilled is True
    assert len(result.finalized_tasks) == 1


async def test_domain_tweak_dataset_builder_fails_on_question_or_a2_underfill() -> None:
    pair_source = _RecordingPairSource(_pair_inputs(8))
    batch_pipeline = _FakeBatchPipeline(finalized_count=1, underfilled=True)
    builder = DomainTweakMinerTaskDatasetBuilder(pair_source=pair_source, batch_pipeline=batch_pipeline)

    with pytest.raises(RuntimeError, match="fewer finalized tasks"):
        await builder.build(_dataset_request(minimum_task_total=2))


async def test_domain_tweak_dataset_builder_fails_on_failed_finalizations() -> None:
    pair_source = _RecordingPairSource(_pair_inputs(4))
    batch_pipeline = _FakeBatchPipeline(finalized_count=1, failed_finalizations=1)
    builder = DomainTweakMinerTaskDatasetBuilder(pair_source=pair_source, batch_pipeline=batch_pipeline)

    with pytest.raises(RuntimeError, match="reference-answer finalization failed"):
        await builder.build(_dataset_request(minimum_task_total=1))


async def test_domain_tweak_dataset_builder_returns_tasks_after_recovered_a2_finalization() -> None:
    pair_source = _RecordingPairSource(_pair_inputs(4))
    batch_pipeline = _FakeBatchPipeline(
        finalized_count=1,
        failed_finalizations=0,
        reference_answer_finalization_attempt_count=2,
        reference_answer_retry_attempt_count=1,
    )
    builder = DomainTweakMinerTaskDatasetBuilder(pair_source=pair_source, batch_pipeline=batch_pipeline)

    tasks = await builder.build(_dataset_request(minimum_task_total=1))

    assert len(tasks) == 1


async def test_domain_tweak_dataset_builder_requires_batch_created_at() -> None:
    pair_source = _RecordingPairSource(_pair_inputs(4))
    batch_pipeline = _FakeBatchPipeline(finalized_count=1)
    builder = DomainTweakMinerTaskDatasetBuilder(pair_source=pair_source, batch_pipeline=batch_pipeline)

    request = _dataset_request(minimum_task_total=1).model_copy(update={"created_at": None})

    with pytest.raises(ValueError, match="requires request.created_at"):
        await builder.build_with_result(request)

    assert pair_source.calls == []


def test_finalized_tasks_from_domain_tweak_result_returns_tasks_for_complete_result() -> None:
    reviewed = (_reviewed_question(1), _reviewed_question(2))
    result = DomainTweakBatchGenerationResult(
        target_count=2,
        selected_questions=reviewed,
        finalized_tasks=tuple(_finalized_task(item, index) for index, item in enumerate(reviewed, start=1)),
        rejected_attempts=(),
        failed_finalizations=(),
        underfilled=False,
    )

    tasks = finalized_tasks_from_domain_tweak_result(result, target_count=2)

    assert [task.query.text for task in tasks] == ["Question 1?", "Question 2?"]


def test_finalized_tasks_from_domain_tweak_result_raises_for_underfill() -> None:
    reviewed = (_reviewed_question(1),)
    result = DomainTweakBatchGenerationResult(
        target_count=2,
        selected_questions=reviewed,
        finalized_tasks=(_finalized_task(reviewed[0], 1),),
        rejected_attempts=(),
        failed_finalizations=(),
        underfilled=True,
    )

    with pytest.raises(RuntimeError, match="fewer finalized tasks"):
        finalized_tasks_from_domain_tweak_result(result, target_count=2)


def test_finalized_tasks_from_domain_tweak_result_raises_for_failed_finalization() -> None:
    reviewed = (_reviewed_question(1),)
    result = DomainTweakBatchGenerationResult(
        target_count=1,
        selected_questions=reviewed,
        finalized_tasks=(_finalized_task(reviewed[0], 1),),
        rejected_attempts=(),
        failed_finalizations=(_failed_finalization(reviewed[0]),),
        underfilled=False,
    )

    with pytest.raises(RuntimeError, match="reference-answer finalization failed"):
        finalized_tasks_from_domain_tweak_result(result, target_count=1)


@dataclass(slots=True)
class _RecordingPairSource:
    pair_inputs: tuple[DomainTweakPairInput, ...]
    calls: list[tuple[UUID, datetime, int]]

    def __init__(self, pair_inputs: tuple[DomainTweakPairInput, ...]) -> None:
        self.pair_inputs = pair_inputs
        self.calls = []

    async def load_pair_inputs(
        self,
        *,
        batch_id: UUID,
        timestamp: datetime,
        requested_count: int,
    ) -> tuple[DomainTweakPairInput, ...]:
        self.calls.append((batch_id, timestamp, requested_count))
        return self.pair_inputs[:requested_count]


@dataclass(slots=True)
class _FakeBatchPipeline:
    finalized_count: int
    underfilled: bool = False
    failed_finalizations: int = 0
    reference_answer_finalization_attempt_count: int = 0
    reference_answer_retry_attempt_count: int = 0
    reference_answer_retry_round_count: int = 0
    observed_pair_inputs: tuple[DomainTweakPairInput, ...] = ()
    observed_target_count: int | None = None

    async def generate_batch(
        self,
        pair_inputs: Sequence[DomainTweakPairInput],
        config: DomainTweakBatchGenerationConfig,
    ) -> DomainTweakBatchGenerationResult:
        self.observed_pair_inputs = tuple(pair_inputs)
        self.observed_target_count = config.target_count
        selected_count = max(self.finalized_count, self.failed_finalizations)
        reviewed = tuple(_reviewed_question(index) for index in range(1, selected_count + 1))
        return DomainTweakBatchGenerationResult(
            target_count=config.target_count,
            selected_questions=reviewed,
            finalized_tasks=tuple(
                _finalized_task(item, index) for index, item in enumerate(reviewed[: self.finalized_count], start=1)
            ),
            rejected_attempts=(),
            failed_finalizations=tuple(_failed_finalization(item) for item in reviewed[: self.failed_finalizations]),
            reference_answer_finalization_attempt_count=self.reference_answer_finalization_attempt_count,
            reference_answer_retry_attempt_count=self.reference_answer_retry_attempt_count,
            reference_answer_retry_round_count=self.reference_answer_retry_round_count,
            underfilled=self.underfilled,
        )


def _dataset_request(*, minimum_task_total: int) -> MinerTaskDatasetRequest:
    spec = MinerTaskModelSpec(
        provider="vertex",
        model="unused-by-domain-tweak-builder",
        temperature=None,
        max_output_tokens=None,
    )
    return MinerTaskDatasetRequest(
        batch_id=_BATCH_ID,
        created_at=_BATCH_CREATED_AT,
        minimum_task_total=minimum_task_total,
        generation_task_buffer=99,
        generation_spec=spec,
        reference_spec=spec,
    )


def _pair_inputs(count: int) -> tuple[DomainTweakPairInput, ...]:
    return tuple(
        DomainTweakPairInput(
            pair_id=f"pair-{index:03d}",
            deepsearchqa_form_target="Which films meet all constraints?",
            deepresearch9k_domain_target=f"Domain target {index}",
            timestamp=datetime(2026, 6, 24, tzinfo=UTC),
        )
        for index in range(1, count + 1)
    )


def _reviewed_question(index: int) -> DomainTweakReviewedQuestion:
    phase_result = DomainTweakAdkPhaseResult(phase="question_generation", terminal_status="validated")
    return DomainTweakReviewedQuestion(
        pair_input=_pair_inputs(index)[-1],
        question_candidate=DomainTweakQuestionCandidate(
            question=f"Question {index}?",
            short_answer=f"Answer {index}",
            solution_plan="- Find sources\n- Compare constraints",
        ),
        form_review=DomainTweakFormReview(
            form_match=True,
            false_premise_status="none",
            reviewer_feedback="Form preserved.",
            retry_recommended=False,
        ),
        question_generation_result=phase_result,
        form_review_result=phase_result,
    )


def _finalized_task(reviewed_question: DomainTweakReviewedQuestion, index: int) -> DomainTweakFinalizedTask:
    return DomainTweakFinalizedTask(
        reviewed_question=reviewed_question,
        reference_answer_result=DomainTweakAdkPhaseResult(phase="reference_answer", terminal_status="validated"),
        task=MinerTask(
            task_id=uuid4(),
            query=Query(text=f"Question {index}?"),
            reference_answer=ReferenceAnswer(text=f"Answer {index}."),
        ),
    )


def _failed_finalization(reviewed_question: DomainTweakReviewedQuestion) -> DomainTweakFailedFinalization:
    return DomainTweakFailedFinalization(
        reviewed_question=reviewed_question,
        reference_answer_results=(
            DomainTweakAdkPhaseResult(phase="reference_answer", terminal_status="validation_failed"),
        ),
    )
