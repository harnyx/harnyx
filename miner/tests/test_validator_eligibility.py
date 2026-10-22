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


@pytest.fixture
def clock(monkeypatch):
    clock = Mock(return_value=1000.0)
    monkeypatch.setattr("harnyx_miner.validator_eligibility.monotonic", clock)
    return clock


async def test_snapshot_is_shared_across_signers_until_fifteen_minute_expiry(monkeypatch, clock):
    client, reader = chain(monkeypatch)
    assert await reader("permitted")
    assert not await reader("registered")
    assert not await reader("unknown")
    clock.return_value += 899
    assert await reader("permitted")
    client.get_metagraph_info.assert_awaited_once()

    # Revocation and newly granted permits must both be observed on expiry.
    client.get_metagraph_info.return_value = SimpleNamespace(
        block=42, hotkeys=["permitted", "registered"], validator_permit=[False, True]
    )
    clock.return_value += 1
    assert not await reader("permitted")
    assert await reader("registered")
    assert client.get_metagraph_info.await_count == 2


@pytest.mark.parametrize("expired", [False, True])
async def test_concurrent_checks_share_one_refresh(monkeypatch, clock, expired):
    client, reader = chain(monkeypatch)
    if expired:
        await reader.read()
        clock.return_value += 900
    started, release = asyncio.Event(), asyncio.Event()

    async def initialize():
        started.set()
        await release.wait()

    client.initialize.side_effect = initialize
    checks = [asyncio.create_task(reader("permitted")) for _ in range(30)]
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        assert all(await asyncio.gather(*checks))
    finally:
        release.set()
        await asyncio.gather(*checks, return_exceptions=True)
    assert client.initialize.await_count == 1 + int(expired)
    assert client.close.await_count == 1 + int(expired)


async def test_expired_snapshot_cannot_admit_during_failed_refresh_or_backoff(monkeypatch, clock):
    client, reader = chain(monkeypatch)
    assert await reader("permitted")
    clock.return_value += 900
    client.initialize.side_effect = ConnectionError("RPC rate limited")
    results = await asyncio.gather(*(reader("permitted") for _ in range(30)), return_exceptions=True)
    assert all(isinstance(result, Exception) for result in results)
    assert isinstance(results[0], ConnectionError)
    assert client.initialize.await_count == 2
    assert client.close.await_count == 2

    # A recovery before the cooldown ends cannot trigger a new connection.
    client.initialize.side_effect = None
    clock.return_value += 4
    with pytest.raises(RuntimeError, match="backing off"):
        await reader("permitted")
    clock.return_value += 1
    assert await reader("permitted")
    assert client.initialize.await_count == 3


async def test_refresh_backoff_grows_to_one_minute_and_resets_after_recovery(monkeypatch, clock):
    client, reader = chain(monkeypatch)
    client.initialize.side_effect = ConnectionError("RPC rate limited")
    for attempts, delay in enumerate([5, 10, 20, 40, 60, 60], start=1):
        with pytest.raises(ConnectionError):
            await reader.read()
        clock.return_value += delay - 0.1
        with pytest.raises(RuntimeError, match="backing off"):
            await reader.read()
        assert client.initialize.await_count == attempts
        clock.return_value += 0.1

    client.initialize.side_effect = None
    await reader.read()
    clock.return_value += 900
    client.initialize.side_effect = ConnectionError("RPC rate limited again")
    with pytest.raises(ConnectionError):
        await reader.read()
    clock.return_value += 5
    client.initialize.side_effect = None
    assert await reader("permitted")


async def test_cancelled_refresh_closes_connection_and_allows_next_request(monkeypatch, clock):
    client, reader = chain(monkeypatch)
    started = asyncio.Event()

    async def initialize():
        started.set()
        await asyncio.Event().wait()

    client.initialize.side_effect = initialize
    refresh = asyncio.create_task(reader.read())
    await asyncio.wait_for(started.wait(), timeout=1)
    refresh.cancel()
    with pytest.raises(asyncio.CancelledError):
        await refresh
    client.close.assert_awaited_once()
    client.initialize.side_effect = None
    assert await reader("permitted")
    assert client.close.await_count == 2
