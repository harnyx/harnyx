from __future__ import annotations

import bittensor as bt
import pytest

from harnyx_validator.infrastructure.subtensor.hotkey import ensure_wallet_hotkey_from_seed


def _make_wallet(tmp_path) -> bt.wallet.Wallet:
    return bt.wallet(name="validator", hotkey="default", path=str(tmp_path))


def test_ensure_wallet_hotkey_from_seed_creates_hotkey_when_missing(tmp_path) -> None:
    mnemonic = bt.Keypair.generate_mnemonic()
    wallet = _make_wallet(tmp_path)

    assert wallet.hotkey_file.exists_on_device() is False

    ensure_wallet_hotkey_from_seed(wallet, mnemonic)

    assert wallet.hotkey_file.exists_on_device() is True
    assert wallet.hotkey.ss58_address == bt.Keypair.create_from_mnemonic(mnemonic).ss58_address


def test_ensure_wallet_hotkey_from_seed_accepts_uri_form_seed(tmp_path) -> None:
    seed = "//Alice"
    wallet = _make_wallet(tmp_path)

    ensure_wallet_hotkey_from_seed(wallet, seed)

    assert wallet.hotkey.ss58_address == bt.Keypair.create_from_uri(seed).ss58_address


def test_ensure_wallet_hotkey_from_seed_is_idempotent_when_matches(tmp_path) -> None:
    mnemonic = bt.Keypair.generate_mnemonic()
    wallet = _make_wallet(tmp_path)

    ensure_wallet_hotkey_from_seed(wallet, mnemonic)
    expected_ss58 = wallet.hotkey.ss58_address

    ensure_wallet_hotkey_from_seed(wallet, mnemonic)

    assert wallet.hotkey.ss58_address == expected_ss58


def test_ensure_wallet_hotkey_from_seed_raises_when_mismatched(tmp_path) -> None:
    mnemonic = bt.Keypair.generate_mnemonic()
    other_mnemonic = bt.Keypair.generate_mnemonic()
    wallet = _make_wallet(tmp_path)

    ensure_wallet_hotkey_from_seed(wallet, mnemonic)

    with pytest.raises(RuntimeError, match="SUBTENSOR_HOTKEY_MNEMONIC"):
        ensure_wallet_hotkey_from_seed(wallet, other_mnemonic)


def test_create_wallet_raises_when_missing_mnemonic_and_keyfile(tmp_path, monkeypatch) -> None:
    from harnyx_commons.config.subtensor import SubtensorSettings
    from harnyx_validator.infrastructure.subtensor.hotkey import create_wallet

    original_wallet = bt.wallet

    def wallet_factory(*, name: str, hotkey: str) -> bt.wallet.Wallet:
        return original_wallet(name=name, hotkey=hotkey, path=str(tmp_path))

    monkeypatch.setattr("harnyx_validator.infrastructure.subtensor.hotkey.bt.wallet", wallet_factory)

    settings = SubtensorSettings.model_validate(
        {
            "SUBTENSOR_WALLET_NAME": "validator",
            "SUBTENSOR_HOTKEY_NAME": "default",
        }
    )

    with pytest.raises(RuntimeError, match="SUBTENSOR_HOTKEY_MNEMONIC"):
        create_wallet(settings)


@pytest.mark.parametrize("mnemonic_value", ["", "   "])
def test_create_wallet_uses_existing_hotkey_when_mnemonic_env_is_blank(
    tmp_path, monkeypatch, mnemonic_value: str
) -> None:
    from harnyx_validator.infrastructure.subtensor.hotkey import create_wallet
    from harnyx_validator.runtime.settings import Settings

    original_wallet = bt.wallet
    expected_hotkey = _make_wallet(tmp_path)
    ensure_wallet_hotkey_from_seed(expected_hotkey, "//Alice")

    def wallet_factory(*, name: str, hotkey: str) -> bt.wallet.Wallet:
        return original_wallet(name=name, hotkey=hotkey, path=str(tmp_path))

    monkeypatch.setattr("harnyx_validator.infrastructure.subtensor.hotkey.bt.wallet", wallet_factory)
    monkeypatch.setenv("SUBTENSOR_WALLET_NAME", "validator")
    monkeypatch.setenv("SUBTENSOR_HOTKEY_NAME", "default")
    monkeypatch.setenv("SUBTENSOR_HOTKEY_MNEMONIC", mnemonic_value)

    settings = Settings.load()
    wallet = create_wallet(settings.subtensor)

    assert wallet.hotkey.ss58_address == expected_hotkey.hotkey.ss58_address
