"""Miner-task dataset adapter for the domain-tweak generation pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.domain_tweak_generation.pair_source import DomainTweakPairInputSource
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakBatchGenerationConfig,
    DomainTweakBatchGenerationResult,
)
from harnyx_commons.miner_task_generation import DomainTweakPairInput, MinerTaskDatasetRequest


class DomainTweakBatchPipelinePort(Protocol):
    async def generate_batch(
        self,
        pair_inputs: Sequence[DomainTweakPairInput],
        config: DomainTweakBatchGenerationConfig,
    ) -> DomainTweakBatchGenerationResult: ...


class DomainTweakMinerTaskDatasetBuilder:
    """Builds platform miner tasks through the public domain-tweak pipeline."""

    def __init__(
        self,
        *,
        pair_source: DomainTweakPairInputSource,
        batch_pipeline: DomainTweakBatchPipelinePort,
    ) -> None:
        self._pair_source = pair_source
        self._batch_pipeline = batch_pipeline

    async def build(self, request: MinerTaskDatasetRequest) -> tuple[MinerTask, ...]:
        result = await self.build_with_result(request)
        return finalized_tasks_from_domain_tweak_result(result, target_count=request.minimum_task_total)

    async def build_with_result(self, request: MinerTaskDatasetRequest) -> DomainTweakBatchGenerationResult:
        target_count = request.minimum_task_total
        config = DomainTweakBatchGenerationConfig(target_count=target_count)
        if request.created_at is None:
            raise ValueError("domain-tweak miner-task generation requires request.created_at")
        pair_inputs = await self._pair_source.load_pair_inputs(
            batch_id=request.batch_id,
            timestamp=request.created_at,
            requested_count=config.question_policy.hard_attempt_cap(target_count),
        )
        return await self._batch_pipeline.generate_batch(pair_inputs, config)


def finalized_tasks_from_domain_tweak_result(
    result: DomainTweakBatchGenerationResult,
    *,
    target_count: int,
) -> tuple[MinerTask, ...]:
    if result.underfilled or len(result.finalized_tasks) != target_count:
        if result.discarded_candidates:
            raise RuntimeError(
                "domain-tweak source-aware candidate finalization was discarded: "
                f"{len(result.discarded_candidates)} discarded candidate(s)"
            )
        raise RuntimeError(
            "domain-tweak generation produced fewer finalized tasks than requested: "
            f"requested {target_count}, finalized {len(result.finalized_tasks)}"
        )
    return tuple(finalized.task for finalized in result.finalized_tasks)


__all__ = [
    "DomainTweakBatchPipelinePort",
    "DomainTweakMinerTaskDatasetBuilder",
    "finalized_tasks_from_domain_tweak_result",
]
