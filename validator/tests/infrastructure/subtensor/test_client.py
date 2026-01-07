from __future__ import annotations

from caster_commons.config.subtensor import SubtensorSettings
from caster_validator.infrastructure.subtensor.client import RuntimeSubtensorClient
from validator.tests.fixtures.subtensor import FakeSubtensorClient


def make_settings() -> SubtensorSettings:
    return SubtensorSettings(
        network="local",
        endpoint="ws://127.0.0.1:9945",
        netuid=1,
        wallet_name="validator",
        hotkey_name="default",
        wait_for_inclusion=False,
        wait_for_finalization=False,
        transaction_mode="immortal",
        transaction_period=None,
    )


def test_runtime_subtensor_client_uses_factory() -> None:
    fake = FakeSubtensorClient()
    client = RuntimeSubtensorClient(make_settings(), client_factory=lambda cfg: fake)

    client.connect()

    assert fake.connected is True


def test_runtime_subtensor_client_delegates_calls() -> None:
    fake = FakeSubtensorClient()
    snapshot = fake.metagraph = fake.metagraph.__class__(uids=(1, 2), hotkeys=("a", "b"))
    client = RuntimeSubtensorClient(make_settings(), client_factory=lambda cfg: fake)

    assert client.fetch_metagraph() == snapshot

    tx_hash = client.submit_weights({1: 0.6, 2: 0.4})
    assert fake.weight_updates == [{1: 0.6, 2: 0.4}]
    assert fake.fetch_weight(1) == 0.6
    assert tx_hash == fake.tx_hashes[-1]
