"""Read-only contract check against the configured chain and pinned Bittensor SDK."""

import asyncio

import pytest
from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.core.chain_data.metagraph_info import SelectiveMetagraphIndex

from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_miner.validator_eligibility import ValidatorEligibility

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_finalized_validator_snapshot_matches_pinned_sdk():
    settings = SubtensorSettings()
    observed = await ValidatorEligibility(settings).read()
    async with asyncio.timeout(30), AsyncSubtensor(network=settings.endpoint) as subtensor:
        expected = await subtensor.get_metagraph_info(
            netuid=settings.netuid,
            block=observed.block,
            selected_indices=[
                SelectiveMetagraphIndex.Block,
                SelectiveMetagraphIndex.Hotkeys,
                SelectiveMetagraphIndex.ValidatorPermit,
            ],
        )
        assert expected is not None
        assert observed.hotkeys == expected.hotkeys
        assert observed.validator_permit == expected.validator_permit
        for hotkey, permit in zip(expected.hotkeys, expected.validator_permit, strict=True):
            assert observed.permits(hotkey) is permit
        assert not observed.permits("not-a-registered-hotkey")
