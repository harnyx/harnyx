from __future__ import annotations

import socket
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

import harnyx_validator.infrastructure.subtensor.bittensor as bittensor_mod
from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_validator.application.ports.subtensor import WeightSubmissionTooEarlyError
from harnyx_validator.infrastructure.subtensor.bittensor import BittensorSubtensorClient


def _make_settings(*, wait_for_inclusion: bool = True, wait_for_finalization: bool = False) -> SubtensorSettings:
    return SubtensorSettings.model_construct(
        network="local",
        endpoint="ws://127.0.0.1:9945",
        netuid=1,
        wallet_name="validator",
        hotkey_name="default",
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
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
        self.network = "local"
        self._commit_reveal_enabled = commit_reveal_enabled
        self.substrate = _SubstrateStub()
        self.sign_calls: list[dict[str, object]] = []
        self.sign_side_effects: list[Exception | object] = []
        self.set_reveal_commitment_calls: list[dict[str, object]] = []
        self.set_weights_calls: list[dict[str, object]] = []
        self.set_weights_extrinsic_calls: list[dict[str, object]] = []
        self.set_weights_extrinsic_side_effects: list[Exception | object] = []
        self.validator_uid: int | None = 11
        self.blocks_since_last_update_value = 101
        self.weights_rate_limit_value: int | None = 100
        self.last_update_values: list[int] | None = [0] * 16
        self.current_block_value = 123

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

    def compose_call(
        self,
        *,
        call_module: str,
        call_function: str,
        call_params: dict[str, object],
    ) -> dict[str, object]:
        return self.substrate.compose_call(
            call_module=call_module,
            call_function=call_function,
            call_params=call_params,
        )

    def sign_and_send_extrinsic(self, **kwargs: object) -> object:
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

    def get_uid_for_hotkey_on_subnet(self, hotkey_ss58: str, netuid: int) -> int | None:
        assert hotkey_ss58
        assert netuid == 1
        return self.validator_uid

    def blocks_since_last_update(self, netuid: int, uid: int) -> int:
        assert netuid == 1
        self.blocks_since_last_update_calls.append(uid)
        assert uid == self.validator_uid
        return self.blocks_since_last_update_value

    def weights_rate_limit(self, netuid: int) -> int | None:
        assert netuid == 1
        return self.weights_rate_limit_value

    def get_current_block(self) -> int:
        return self.current_block_value

    def get_hyperparameter(self, param_name: str, netuid: int, block: int | None = None) -> object:
        assert netuid == 1
        del block
        if param_name == "LastUpdate":
            return self.last_update_values
        if param_name == "WeightsSetRateLimit":
            return self.weights_rate_limit_value
        if param_name == "CommitRevealWeightsEnabled":
            return self._commit_reveal_enabled
        raise AssertionError(param_name)

    def get_subnet_hyperparameters(self, netuid: int, block: int) -> SimpleNamespace:
        assert netuid == 1
        assert block == self.current_block_value
        return SimpleNamespace(tempo=360, commit_reveal_period=1)


def _make_wallet() -> object:
    hotkey = SimpleNamespace(
        ss58_address="5CihjJXv7CqQXXSYQoQLjjFH5f8GbrhhjM81KDiXTzgqrqiY",
        public_key=(b"\x11" * 32),
    )
    return SimpleNamespace(hotkey=hotkey)


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    subtensor: _SubtensorStub,
    settings: SubtensorSettings | None = None,
) -> BittensorSubtensorClient:
    client = BittensorSubtensorClient(settings or _make_settings())
    monkeypatch.setattr(client, "_ensure_ready", lambda: None)
    client._subtensor = cast(Any, subtensor)
    client._wallet = cast(Any, _make_wallet())
    return client


def _patch_set_weights_extrinsic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_set_weights_extrinsic(**kwargs: object) -> object:
        subtensor = cast(_SubtensorStub, kwargs["subtensor"])
        subtensor.set_weights_extrinsic_calls.append(dict(kwargs))
        if subtensor.set_weights_extrinsic_side_effects:
            outcome = subtensor.set_weights_extrinsic_side_effects.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        call = subtensor.compose_call(
            call_module="SubtensorModule",
            call_function="set_mechanism_weights",
            call_params={
                "netuid": kwargs["netuid"],
                "mecid": kwargs["mechid"],
                "dests": kwargs["uids"],
                "weights": kwargs["weights"],
                "version_key": kwargs["version_key"],
            },
        )
        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=kwargs["wallet"],
            wait_for_inclusion=kwargs["wait_for_inclusion"],
            wait_for_finalization=kwargs["wait_for_finalization"],
            use_nonce=True,
            period=kwargs["period"],
            sign_with="hotkey",
            nonce_key="hotkey",
        )

    monkeypatch.setattr(bittensor_mod, "set_weights_extrinsic", fake_set_weights_extrinsic)


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
    call = cast(dict[str, object], subtensor.sign_calls[0]["call"])
    assert call["call_module"] == "SubtensorModule"
    assert call["call_function"] == "commit_timelocked_mechanism_weights"
    params = cast(dict[str, object], call["call_params"])
    assert params["netuid"] == 1
    assert params["mecid"] == 0
    assert subtensor.set_weights_calls == []


def test_submit_weights_commit_reveal_waits_for_inclusion_when_settings_disable_all_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    client = _make_client(
        monkeypatch,
        subtensor=subtensor,
        settings=_make_settings(wait_for_inclusion=False, wait_for_finalization=False),
    )

    client.submit_weights({7: 1.0})

    assert subtensor.sign_calls[0]["wait_for_inclusion"] is True
    assert subtensor.sign_calls[0]["wait_for_finalization"] is False


def test_submit_weights_retries_commit_reveal_when_attempt_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.sign_side_effects = [RuntimeError("temporary rpc failure"), (True, "signed-hotkey-extrinsic")]
    client = _make_client(monkeypatch, subtensor=subtensor)

    tx_hash = client.submit_weights({7: 1.0})

    assert tx_hash.startswith("reveal_round:")
    assert len(subtensor.sign_calls) == 2


def test_submit_weights_does_not_precheck_validator_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.validator_uid = -1
    client = _make_client(monkeypatch, subtensor=subtensor)

    tx_hash = client.submit_weights({7: 1.0})

    assert tx_hash.startswith("reveal_round:")
    assert len(subtensor.sign_calls) == 1
    assert subtensor.set_reveal_commitment_calls == []
    assert subtensor.set_weights_calls == []


def test_commit_reveal_preserves_deterministic_return_after_transient_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transient = httpx.ConnectTimeout("connect timed out")
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.sign_side_effects = [transient] + [(False, "chain rejected")] * 5
    client = _make_client(monkeypatch, subtensor=subtensor)

    with pytest.raises(RuntimeError) as exc_info:
        client.submit_weights({7: 1.0})

    assert str(exc_info.value) == "set_weights failed: chain rejected"
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("connect failed"),
        httpx.ConnectError("connect failed"),
        TimeoutError("local timeout"),
        socket.gaierror(socket.EAI_NONAME, "permanent dns"),
        httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"),
    ],
)
def test_commit_reveal_does_not_treat_non_transient_transport_shapes_as_transient(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.sign_side_effects = [exc] * 5
    client = _make_client(monkeypatch, subtensor=subtensor)

    with pytest.raises(RuntimeError) as exc_info:
        client.submit_weights({7: 1.0})

    assert str(exc_info.value).startswith("set_weights failed:")
    assert exc_info.value.__cause__ is None


def test_commit_reveal_raises_wrapper_from_final_transient_after_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_exception = httpx.ConnectTimeout("final timeout")
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.sign_side_effects = [
        httpx.ConnectTimeout("timeout 1"),
        httpx.ConnectTimeout("timeout 2"),
        httpx.ConnectTimeout("timeout 3"),
        httpx.ConnectTimeout("timeout 4"),
        final_exception,
    ]
    client = _make_client(monkeypatch, subtensor=subtensor)

    with pytest.raises(RuntimeError, match="commit-reveal weight submission attempts failed") as exc_info:
        client.submit_weights({7: 1.0})

    assert exc_info.value.__cause__ is final_exception


def test_last_update_block_returns_none_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    subtensor.last_update_values = None
    client = _make_client(monkeypatch, subtensor=subtensor)

    assert client.last_update_block(11) is None


def test_last_update_block_allows_missing_uid_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    subtensor.last_update_values = []
    client = _make_client(monkeypatch, subtensor=subtensor)

    assert client.last_update_block(11) is None


def test_last_update_block_does_not_treat_zero_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    subtensor.last_update_values[11] = 0
    client = _make_client(monkeypatch, subtensor=subtensor)

    assert client.last_update_block(11) == 0


def test_submit_weights_plain_allows_exact_rate_limit_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_set_weights_extrinsic(monkeypatch)
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    subtensor.current_block_value = 200
    subtensor.last_update_values[11] = 100
    subtensor.weights_rate_limit_value = 100
    client = _make_client(monkeypatch, subtensor=subtensor)

    tx_hash = client.submit_weights({7: 1.0})

    assert tx_hash
    assert len(subtensor.sign_calls) == 1
    assert subtensor.sign_calls[0]["use_nonce"] is True
    assert subtensor.sign_calls[0]["nonce_key"] == "hotkey"
    assert subtensor.sign_calls[0]["sign_with"] == "hotkey"
    call = cast(dict[str, object], subtensor.sign_calls[0]["call"])
    assert call["call_module"] == "SubtensorModule"
    assert call["call_function"] == "set_mechanism_weights"
    assert subtensor.set_weights_extrinsic_calls[0]["mechid"] == 0
    assert subtensor.set_weights_extrinsic_calls[0]["mev_protection"] is False
    assert subtensor.set_weights_calls == []


def test_submit_weights_plain_waits_for_inclusion_when_settings_disable_all_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_set_weights_extrinsic(monkeypatch)
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    client = _make_client(
        monkeypatch,
        subtensor=subtensor,
        settings=_make_settings(wait_for_inclusion=False, wait_for_finalization=False),
    )

    client.submit_weights({7: 1.0})

    assert subtensor.set_weights_extrinsic_calls[0]["wait_for_inclusion"] is True
    assert subtensor.set_weights_extrinsic_calls[0]["wait_for_finalization"] is False


def test_submit_weights_retries_plain_set_weights_when_attempt_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_set_weights_extrinsic(monkeypatch)
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    subtensor.sign_side_effects = [(False, "temporary rpc failure"), (True, "signed-hotkey-extrinsic")]
    client = _make_client(monkeypatch, subtensor=subtensor)

    tx_hash = client.submit_weights({7: 1.0})

    assert tx_hash == "signed-hotkey-extrinsic"
    assert len(subtensor.sign_calls) == 2
    assert subtensor.set_weights_calls == []


def test_submit_weights_plain_raises_too_early_for_chain_cadence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_set_weights_extrinsic(monkeypatch)
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    subtensor.set_weights_extrinsic_side_effects = [
        SimpleNamespace(
            success=False,
            message="SubtensorModule.SettingWeightsTooFast",
            error={"name": "SettingWeightsTooFast"},
        )
    ]
    client = _make_client(monkeypatch, subtensor=subtensor)

    with pytest.raises(WeightSubmissionTooEarlyError):
        client.submit_weights({7: 1.0})

    assert len(subtensor.set_weights_extrinsic_calls) == 1


def test_submit_weights_plain_keeps_other_chain_errors_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_set_weights_extrinsic(monkeypatch)
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    subtensor.set_weights_extrinsic_side_effects = [
        SimpleNamespace(
            success=False,
            message="SubtensorModule.TxRateLimitExceeded",
            error={"name": "TxRateLimitExceeded"},
        )
    ] * 5
    client = _make_client(monkeypatch, subtensor=subtensor)

    with pytest.raises(RuntimeError, match="set_weights failed: SubtensorModule.TxRateLimitExceeded"):
        client.submit_weights({7: 1.0})

    assert len(subtensor.set_weights_extrinsic_calls) == 5


def test_submit_weights_commit_reveal_raises_too_early_for_chain_cadence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.sign_side_effects = [
        SimpleNamespace(
            success=False,
            message="SubtensorModule.CommittingWeightsTooFast",
            error={"name": "CommittingWeightsTooFast"},
        )
    ]
    client = _make_client(monkeypatch, subtensor=subtensor)

    with pytest.raises(WeightSubmissionTooEarlyError):
        client.submit_weights({7: 1.0})

    assert len(subtensor.sign_calls) == 1
    assert subtensor.set_reveal_commitment_calls == []
    assert subtensor.set_weights_calls == []


def test_submit_weights_commit_reveal_keeps_non_chain_too_early_text_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.sign_side_effects = [
        RuntimeError("diagnostic text mentioned SettingWeightsTooFast but was not a chain error")
    ] * 5
    client = _make_client(monkeypatch, subtensor=subtensor)

    with pytest.raises(RuntimeError, match="set_weights failed:"):
        client.submit_weights({7: 1.0})

    assert len(subtensor.sign_calls) == 5


def test_submit_weights_commit_reveal_raises_too_early_for_chain_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensor = _SubtensorStub(commit_reveal_enabled=True)
    subtensor.sign_side_effects = [RuntimeError({"name": "CommittingWeightsTooFast", "docs": []})]
    client = _make_client(monkeypatch, subtensor=subtensor)

    with pytest.raises(WeightSubmissionTooEarlyError):
        client.submit_weights({7: 1.0})

    assert len(subtensor.sign_calls) == 1


def test_submit_weights_uses_direct_plain_set_weights_extrinsic_when_commit_reveal_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_set_weights_extrinsic(monkeypatch)
    subtensor = _SubtensorStub(commit_reveal_enabled=False)
    client = _make_client(monkeypatch, subtensor=subtensor)

    tx_hash = client.submit_weights({7: 1.0})

    assert tx_hash
    assert len(subtensor.sign_calls) == 1
    assert subtensor.set_weights_extrinsic_calls[0]["mechid"] == 0
    assert subtensor.set_weights_calls == []
