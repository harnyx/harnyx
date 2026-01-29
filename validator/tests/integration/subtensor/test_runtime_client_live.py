from __future__ import annotations

import time

import pytest

from caster_validator.application.scheduling.gate import chain_epoch_index, commitment_marker
from caster_validator.infrastructure.subtensor.client import RuntimeSubtensorClient
from caster_validator.runtime.settings import Settings

pytestmark = pytest.mark.subtensor_live


def _load_settings() -> Settings:
    settings = Settings.load()
    assert settings.subtensor.endpoint, "subtensor endpoint not configured"
    return settings


def test_runtime_client_live_commitment_and_weights() -> None:
    settings = _load_settings()

    try:
        import bittensor  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        pytest.fail(f"bittensor package not available: {exc}")

    client = RuntimeSubtensorClient(settings.subtensor)

    try:
        client.connect()
    except Exception as exc:  # pragma: no cover - network issues
        pytest.fail(f"unable to connect to subtensor: {exc}")

    validator_info = client.validator_info()
    assert validator_info.uid >= 0, "validator hotkey is not registered on the subnet"

    # Publish and confirm commitment using the canonical marker format.
    now_block = client.current_block()
    tempo = client.tempo(settings.subtensor.netuid)
    epoch = chain_epoch_index(at_block=now_block, netuid=settings.subtensor.netuid, tempo=tempo)
    commitment_payload = commitment_marker(validator_info.uid, epoch)
    client.publish_commitment(commitment_payload, blocks_until_reveal=1)

    fetched = None
    for _ in range(10):
        time.sleep(10)
        fetched = client.fetch_commitment(validator_info.uid)
        if fetched is not None:
            break
    assert fetched is not None, "commitment not retrievable"

    baseline_update = client.last_update_block(validator_info.uid)
    baseline_value = baseline_update if baseline_update is not None else -1

    # Choose a target miner uid ≠ validator uid
    metagraph = client.fetch_metagraph()
    try:
        target_uid = next(u for u in metagraph.uids if u != validator_info.uid)
    except StopIteration:
        pytest.fail("no miner UID available on this subnet to set weight for")

    # Submit weight and verify via adapter
    deadline = time.time() + 300
    last_submit_error: Exception | None = None
    while time.time() < deadline:
        try:
            client.submit_weights({target_uid: 1.0})
            last_submit_error = None
            break
        except RuntimeError as exc:
            last_submit_error = exc
            if "too soon to commit weights" not in str(exc):
                raise
            time.sleep(10)
    if last_submit_error is not None:
        raise last_submit_error

    observed = baseline_value
    while time.time() < deadline:
        time.sleep(5)
        latest = client.last_update_block(validator_info.uid)
        if latest is not None:
            observed = int(latest)
        if observed > baseline_value:
            break

    assert observed > baseline_value

    # TODO:
    # - Track reveal_round from submissions and use it to gate polling.
    # - Add helper utilities to shorten waits once commit/reveal settings are configurable here.
