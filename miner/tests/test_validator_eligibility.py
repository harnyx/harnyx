"""Admission must use complete permit data at a single finalized block."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_miner.validator_eligibility import ValidatorEligibility

pytestmark = pytest.mark.anyio("asyncio")


def chain(monkeypatch, **overrides):
    info = SimpleNamespace(block=42, hotkeys=["permitted", "registered"], validator_permit=[True, False])
    for key, value in overrides.items():
        setattr(info, key, value)
    client = SimpleNamespace(
        initialize=AsyncMock(),
        close=AsyncMock(),
        get_metagraph_info=AsyncMock(return_value=info),
        substrate=SimpleNamespace(
            get_chain_finalised_head=AsyncMock(return_value="finalized"), get_block_number=AsyncMock(return_value=42)
        ),
    )
    monkeypatch.setattr("harnyx_miner.validator_eligibility.AsyncSubtensor", Mock(return_value=client))
    return client, ValidatorEligibility(SubtensorSettings(), timeout_seconds=0.05)


@pytest.mark.parametrize("hotkey, expected", [("permitted", True), ("registered", False), ("unknown", False)])
async def test_registration_and_permit_are_required_at_finalized_hash(monkeypatch, hotkey, expected):
    client, reader = chain(monkeypatch)
    assert await reader(hotkey) is expected
    assert client.get_metagraph_info.call_args.kwargs["block_hash"] == "finalized"
    client.close.assert_awaited_once()


@pytest.mark.parametrize(
    "overrides",
    [
        {"block": 41},
        {"validator_permit": [True]},
        {"validator_permit": [1, False]},
        {"hotkeys": ["permitted", "permitted"]},
        {"hotkeys": None},
    ],
)
async def test_inconsistent_or_malformed_chain_data_fails_closed(monkeypatch, overrides):
    client, reader = chain(monkeypatch, **overrides)
    with pytest.raises(ValueError):
        await reader("permitted")
    client.close.assert_awaited_once()


@pytest.mark.parametrize("stage", ["initialize", "get_metagraph_info"])
async def test_stalled_chain_read_is_bounded_and_closes_client(monkeypatch, stage):
    client, reader = chain(monkeypatch)

    async def stall(*args, **kwargs):
        await asyncio.Event().wait()

    getattr(client, stage).side_effect = stall
    async with asyncio.timeout(1):
        with pytest.raises(TimeoutError):
            await reader("permitted")
    client.close.assert_awaited_once()
