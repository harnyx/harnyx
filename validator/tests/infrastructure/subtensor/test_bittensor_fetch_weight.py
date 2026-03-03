from __future__ import annotations

from typing import Any, cast

import pytest

from caster_commons.config.subtensor import SubtensorSettings
from caster_validator.application.ports.subtensor import ValidatorNodeInfo
from caster_validator.infrastructure.subtensor.bittensor import BittensorSubtensorClient


class _SubtensorStub:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def weights(self, *, netuid: int) -> object:
        del netuid
        return self._payload


def _make_settings() -> SubtensorSettings:
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


def _make_client(monkeypatch: pytest.MonkeyPatch, *, payload: object) -> BittensorSubtensorClient:
    client = BittensorSubtensorClient(_make_settings())
    monkeypatch.setattr(client, "_ensure_ready", lambda: None)
    client._subtensor = cast(Any, _SubtensorStub(payload))
    return client


def test_fetch_weight_returns_canonical_payload_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(
        monkeypatch,
        payload=[(0, [(0, 3), (1, 8)]), (1, [(0, 5), (1, 19)])],
    )
    monkeypatch.setattr(client, "validator_info", lambda: ValidatorNodeInfo(uid=0, version_key=None))

    assert client.fetch_weight(1) == 8.0


def test_fetch_weight_raises_runtime_error_for_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, payload=[(0, [(0, True)])])
    monkeypatch.setattr(client, "validator_info", lambda: ValidatorNodeInfo(uid=0, version_key=None))

    with pytest.raises(RuntimeError, match="invalid subtensor weights payload"):
        client.fetch_weight(0)


def test_fetch_weight_returns_zero_for_unregistered_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, payload=[(0, [(0, 7)]), (1, [(0, 9)])])
    monkeypatch.setattr(client, "validator_info", lambda: ValidatorNodeInfo(uid=-1, version_key=None))

    assert client.fetch_weight(0) == 0.0


def test_fetch_weight_returns_zero_for_missing_validator_source_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, payload=[(0, [(0, 7)]), (1, [(0, 9)])])
    monkeypatch.setattr(client, "validator_info", lambda: ValidatorNodeInfo(uid=7, version_key=None))

    assert client.fetch_weight(0) == 0.0


def test_fetch_weight_returns_zero_for_missing_target_in_validator_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, payload=[(0, [(0, 7)]), (1, [(0, 9)])])
    monkeypatch.setattr(client, "validator_info", lambda: ValidatorNodeInfo(uid=1, version_key=None))

    assert client.fetch_weight(1) == 0.0
    assert client.fetch_weight(99) == 0.0
