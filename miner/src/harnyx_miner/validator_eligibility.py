"""Read validator admission eligibility from one finalized subnet snapshot."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Self

from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.core.chain_data.metagraph_info import SelectiveMetagraphIndex
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harnyx_commons.config.subtensor import SubtensorSettings


class ValidatorSnapshot(BaseModel):
    model_config = ConfigDict(strict=True, from_attributes=True, frozen=True)

    block: int = Field(ge=0)
    hotkeys: list[str]
    validator_permit: list[bool]

    @model_validator(mode="after")
    def aligned(self) -> Self:
        if len(self.hotkeys) != len(self.validator_permit) or len(set(self.hotkeys)) != len(self.hotkeys):
            raise ValueError("invalid subnet membership and permit vectors")
        return self

    def permits(self, hotkey: str) -> bool:
        return any(key == hotkey and permit for key, permit in zip(self.hotkeys, self.validator_permit, strict=True))


@dataclass(frozen=True)
class ValidatorEligibility:
    settings: SubtensorSettings
    timeout_seconds: float = 10

    async def read(self) -> ValidatorSnapshot:
        # A request owns its client, including failed initialization and cancellation.
        subtensor = AsyncSubtensor(network=self.settings.endpoint, retry_forever=False)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await subtensor.initialize()
                finalized_hash = await subtensor.substrate.get_chain_finalised_head()
                finalized_block = await subtensor.substrate.get_block_number(finalized_hash)
                info = await subtensor.get_metagraph_info(
                    netuid=self.settings.netuid,
                    block_hash=finalized_hash,
                    selected_indices=[
                        SelectiveMetagraphIndex.Block,
                        SelectiveMetagraphIndex.Hotkeys,
                        SelectiveMetagraphIndex.ValidatorPermit,
                    ],
                )
                snapshot = ValidatorSnapshot.model_validate(info)
                if snapshot.block != finalized_block:
                    raise ValueError("metagraph does not match the finalized block")
                return snapshot
        finally:
            async with asyncio.timeout(self.timeout_seconds):
                await subtensor.close()

    async def __call__(self, hotkey: str) -> bool:
        return (await self.read()).permits(hotkey)
