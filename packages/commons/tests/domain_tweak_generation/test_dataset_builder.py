from datetime import UTC, datetime
from uuid import UUID

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain_tweak_generation import DomainTweakBatchGenerationResult, DomainTweakFinalizedTask
from harnyx_commons.domain_tweak_generation.dataset_builder import DomainTweakMinerTaskDatasetBuilder
from harnyx_commons.miner_task_generation import MinerTaskDatasetRequest, MinerTaskModelSpec


class _Refill:
    def __init__(self) -> None:
        self.target_count: int | None = None

    async def generate_batch(self, **kwargs: object) -> DomainTweakBatchGenerationResult:
        target_count = kwargs["target_count"]
        assert isinstance(target_count, int)
        self.target_count = target_count
        return DomainTweakBatchGenerationResult(
            target_count=self.target_count,
            finalized_tasks=tuple(_finalized(index) for index in range(target_count)),
        )


def _finalized(index: int) -> DomainTweakFinalizedTask:
    return DomainTweakFinalizedTask(
        task=MinerTask(
            task_id=UUID(int=index + 1),
            query=Query(text=f"question {index}"),
            reference_answer=ReferenceAnswer(text=f"answer {index}"),
        ),
    )


@pytest.mark.anyio
async def test_builder_delegates_exact_requested_count_without_attempt_multiplier() -> None:
    """Future failure: dataset adaptation must not expand N into an N-multiple candidate budget."""
    refill = _Refill()
    builder = DomainTweakMinerTaskDatasetBuilder(
        refill_pipeline=refill,  # type: ignore[arg-type]
    )
    spec = MinerTaskModelSpec(
        provider="vertex",
        model="unused",
        temperature=None,
        max_output_tokens=None,
    )
    result = await builder.build_with_result(
        MinerTaskDatasetRequest(
            batch_id=UUID(int=1),
            created_at=datetime.now(UTC),
            minimum_task_total=7,
            generation_task_buffer=0,
            generation_spec=spec,
            reference_spec=spec,
        )
    )

    assert refill.target_count == 7
    assert result.target_count == 7
