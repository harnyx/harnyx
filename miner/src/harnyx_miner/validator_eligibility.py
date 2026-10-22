"""Share finalized validator eligibility for fifteen minutes per reader."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic
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


@dataclass
class ValidatorEligibility:
    settings: SubtensorSettings
    timeout_seconds: float = 10
    _snapshot: ValidatorSnapshot | None = field(default=None, init=False, repr=False)
    _expires_at: float = field(default=0, init=False, repr=False)
    _retry_after: float = field(default=0, init=False, repr=False)
    _retry_delay: float = field(default=5, init=False, repr=False)
    _refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def read(self) -> ValidatorSnapshot:
        async with self._refresh_lock:
            now = monotonic()
            if self._snapshot is not None and now < self._expires_at:
                return self._snapshot
            if now < self._retry_after:
                raise RuntimeError("validator eligibility refresh is backing off")
            try:
                snapshot = await self._read_snapshot()
            except Exception:
                self._retry_after = monotonic() + self._retry_delay
                self._retry_delay = min(self._retry_delay * 2, 60)
                raise
            self._snapshot = snapshot
            # Include time spent reading the chain in the freshness budget.
            self._expires_at = now + 15 * 60
            self._retry_after = 0
            self._retry_delay = 5
            return snapshot

    async def _read_snapshot(self) -> ValidatorSnapshot:
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
