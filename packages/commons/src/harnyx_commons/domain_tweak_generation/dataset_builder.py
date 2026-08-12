"""Stable Platform-facing adapter for exact-shortfall generation."""

from __future__ import annotations

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.domain_tweak_generation.contracts import (
    DomainTweakBatchGenerationResult,
    DomainTweakFinalizedTaskCallback,
    PortfolioCallCallback,
    SlotAttemptCallback,
)
from harnyx_commons.domain_tweak_generation.refill_pipeline import ShortfallRefillPipeline
from harnyx_commons.miner_task_generation import MinerTaskDatasetRequest


class DomainTweakMinerTaskDatasetBuilder:
    def __init__(
        self,
        *,
        refill_pipeline: ShortfallRefillPipeline,
        on_portfolio_completed: PortfolioCallCallback | None = None,
        on_attempt_completed: SlotAttemptCallback | None = None,
    ) -> None:
        self._refill_pipeline = refill_pipeline
        self._on_portfolio_completed = on_portfolio_completed
        self._on_attempt_completed = on_attempt_completed

    async def build(self, request: MinerTaskDatasetRequest) -> tuple[MinerTask, ...]:
        result = await self.build_with_result(request)
        return finalized_tasks_from_domain_tweak_result(result, target_count=request.minimum_task_total)

    async def build_with_result(
        self,
        request: MinerTaskDatasetRequest,
        *,
        on_finalized_task: DomainTweakFinalizedTaskCallback | None = None,
    ) -> DomainTweakBatchGenerationResult:
        if request.created_at is None:
            raise ValueError("domain-tweak generation requires request.created_at")
        return await self._refill_pipeline.generate_batch(
            target_count=request.minimum_task_total,
            on_finalized_task=on_finalized_task,
            on_portfolio_completed=self._on_portfolio_completed,
            on_attempt_completed=self._on_attempt_completed,
        )


def finalized_tasks_from_domain_tweak_result(
    result: DomainTweakBatchGenerationResult,
    *,
    target_count: int,
) -> tuple[MinerTask, ...]:
    if result.target_count != target_count or len(result.finalized_tasks) != target_count:
        raise RuntimeError(
            "domain-tweak generation result does not match the requested task count: "
            f"requested {target_count}, finalized {len(result.finalized_tasks)}"
        )
    return tuple(item.task for item in result.finalized_tasks)


__all__ = [
    "DomainTweakMinerTaskDatasetBuilder",
    "finalized_tasks_from_domain_tweak_result",
]
