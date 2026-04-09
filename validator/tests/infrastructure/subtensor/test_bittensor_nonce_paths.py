from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_validator.infrastructure.subtensor.bittensor import BittensorSubtensorClient


def _make_settings() -> SubtensorSettings:
    return SubtensorSettings.model_construct(
        network="local",
        endpoint="ws://127.0.0.1:9945",
        netuid=1,
        wallet_name="validator",
        hotkey_name="default",
        wait_for_inclusion=True,
        wait_for_finalization=False,
        transaction_mode="immortal",
        transaction_period=None,
    )


class _SubstrateStub:
    def __init__(self) -> None:
        self.compose_calls: list[dict[str, object]] = []

    def compose_call(
        self,
        *,
        call_module: str,
        call_function: str,
        call_params: dict[str, object],
    ) -> dict[str, object]:
        call = {
            "call_module": call_module,
            "call_function": call_function,
            "call_params": call_params,
        }
        self.compose_calls.append(call)
        return call


class _SubtensorStub:
    def __init__(self, *, commit_reveal_enabled: bool) -> None:
        self._commit_reveal_enabled = commit_reveal_enabled
        self.substrate = _SubstrateStub()
        self.sign_calls: list[dict[str, object]] = []
        self.sign_side_effects: list[Exception | tuple[bool, str]] = []
        self.set_reveal_commitment_calls: list[dict[str, object]] = []
        self.set_weights_calls: list[dict[str, object]] = []
        self.validator_uid = 11
        self.blocks_since_last_update_value = 101
        self.weights_rate_limit_value = 100

    def set_reveal_commitment(
        self,
        *,
        wallet: object,
        netuid: int,
        data: str,
        blocks_until_reveal: int,
        period: int | None,
    ) -> tuple[bool, int]:
        self.set_reveal_commitment_calls.append(
            {
                "wallet": wallet,
                "netuid": netuid,
                "data": data,
                "blocks_until_reveal": blocks_until_reveal,
                "period": period,
            }
        )
        return True, 77

    def set_weights(
        self,
        *,
        wallet: object,
        netuid: int,
        weights: list[float],
        uids: list[int],
        wait_for_inclusion: bool,
        wait_for_finalization: bool,
        period: int | None,
    ) -> tuple[bool, str]:
        self.set_weights_calls.append(
            {
                "wallet": wallet,
                "netuid": netuid,
                "weights": weights,
                "uids": uids,
                "wait_for_inclusion": wait_for_inclusion,
                "wait_for_finalization": wait_for_finalization,
                "period": period,
            }
        )
        return True, "delegated-set-weights"

    def sign_and_send_extrinsic(self, **kwargs: object) -> tuple[bool, str]:
        self.sign_calls.append(dict(kwargs))
        if self.sign_side_effects:
            outcome = self.sign_side_effects.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return True, "signed-hotkey-extrinsic"

    def commit_reveal_enabled(self, *, netuid: int) -> bool:
        assert netuid == 1
        return self._commit_reveal_enabled

    def get_uid_for_hotkey_on_subnet(self, hotkey_ss58: str, netuid: int) -> int:
        assert hotkey_ss58
        assert netuid == 1
        return self.validator_uid

    def blocks_since_last_update(self, netuid: int, uid: int) -> int:
        assert netuid == 1
        assert uid == self.validator_uid
        return self.blocks_since_last_update_value

    def weights_rate_limit(self, netuid: int) -> int:
        assert netuid == 1
        return self.weights_rate_limit_value

    def get_current_block(self) -> int:
        return 123

    def get_subnet_hyperparameters(self, netuid: int, block: int) -> SimpleNamespace:
        assert netuid == 1
        assert block == 123
        return SimpleNamespace(tempo=360, commit_reveal_period=1)


def _make_wallet() -> object:
    hotkey = SimpleNamespace(
        ss58_address="5CihjJXv7CqQXXSYQoQLjjFH5f8GbrhhjM81KDiXTzgqrqiY",
        public_key=(b"\x11" * 32),
    )
    return SimpleNamespace(hotkey=hotkey)


def _make_client(monkeypatch: pytest.MonkeyPatch, *, subtensor: _SubtensorStub) -> BittensorSubtensorClient:
    client = BittensorSubtensorClient(_make_settings())
    monkeypatch.setattr(client, "_ensure_ready", lambda: None)
    client._subtensor = cast(Any, subtensor)
    client._wallet = cast(Any, _make_wallet())
    return client


def test_publish_commitment_uses_pool_aware_hotkey_nonce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    client = _make_client(monkeypatch, subtensor=subtensor)

    record = client.publish_commitment("marker", blocks_until_reveal=1)

    assert record.block == 123
    assert len(subtensor.sign_calls) == 1
    assert subtensor.sign_calls[0]["wallet"] == client._wallet
    assert subtensor.sign_calls[0]["wait_for_inclusion"] is False
    assert subtensor.sign_calls[0]["wait_for_finalization"] is True
    assert subtensor.sign_calls[0]["sign_with"] == "hotkey"
    assert subtensor.sign_calls[0]["use_nonce"] is True
    assert subtensor.sign_calls[0]["nonce_key"] == "hotkey"
    assert subtensor.sign_calls[0]["period"] is None
    call = cast(dict[str, object], subtensor.sign_calls[0]["call"])
    assert call["call_module"] == "Commitments"
    assert call["call_function"] == "set_commitment"
    params = cast(dict[str, object], call["call_params"])
    assert params["netuid"] == 1
    assert subtensor.set_reveal_commitment_calls == []


def test_submit_weights_uses_pool_aware_hotkey_nonce_when_commit_reveal_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    client = _make_client(monkeypatch, subtensor=subtensor)

    tx_hash = client.submit_weights({7: 1.0})

    assert tx_hash.startswith("reveal_round:")
    assert len(subtensor.sign_calls) == 1
    assert subtensor.sign_calls[0]["use_nonce"] is True
    assert subtensor.sign_calls[0]["nonce_key"] == "hotkey"
    assert subtensor.sign_calls[0]["sign_with"] == "hotkey"
    assert subtensor.set_weights_calls == []


def test_submit_weights_retries_commit_reveal_when_attempt_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.sign_side_effects = [RuntimeError("temporary rpc failure"), (True, "signed-hotkey-extrinsic")]
    client = _make_client(monkeypatch, subtensor=subtensor)

    tx_hash = client.submit_weights({7: 1.0})

    assert tx_hash.startswith("reveal_round:")
    assert len(subtensor.sign_calls) == 2


def test_submit_weights_keeps_existing_set_weights_delegate_when_commit_reveal_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    client = _make_client(monkeypatch, subtensor=subtensor)

    tx_hash = client.submit_weights({7: 1.0})

    assert tx_hash == "delegated-set-weights"
    assert subtensor.sign_calls == []
    assert len(subtensor.set_weights_calls) == 1
