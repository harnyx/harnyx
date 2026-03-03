"""Hotkey initialization helpers for validator runtime."""

from __future__ import annotations

import bittensor as bt

from caster_commons.config.subtensor import SubtensorSettings


def create_wallet(settings: SubtensorSettings) -> bt.Wallet:
    wallet = bt.wallet(
        name=settings.wallet_name,
        hotkey=settings.hotkey_name,
    )
    mnemonic = settings.hotkey_mnemonic_value
    if mnemonic is not None:
        ensure_wallet_hotkey_from_mnemonic(wallet, mnemonic)
    elif not wallet.hotkey_file.exists_on_device():
        raise RuntimeError(
            "validator hotkey is not configured: set SUBTENSOR_HOTKEY_MNEMONIC or mount an existing hotkey file at "
            f"{wallet.hotkey_file.path}"
        )
    return wallet


def ensure_wallet_hotkey_from_mnemonic(wallet: bt.Wallet, mnemonic: str) -> None:
    expected_ss58 = bt.Keypair.create_from_mnemonic(mnemonic).ss58_address
    if not wallet.hotkey_file.exists_on_device():
        wallet.regenerate_hotkey(
            mnemonic=mnemonic,
            use_password=False,
            overwrite=False,
            suppress=True,
        )

    if not wallet.hotkey_file.exists_on_device():
        raise RuntimeError(f"wallet hotkey file was not created: {wallet.hotkey_file.path}")

    hotkey = wallet.hotkey
    if hotkey is None:
        raise RuntimeError("wallet hotkey is unavailable")
    actual_ss58 = hotkey.ss58_address
    if actual_ss58 != expected_ss58:
        raise RuntimeError(
            "SUBTENSOR_HOTKEY_MNEMONIC does not match existing hotkey file: "
            f"path={wallet.hotkey_file.path} expected_ss58={expected_ss58} actual_ss58={actual_ss58}"
        )


__all__ = ["create_wallet", "ensure_wallet_hotkey_from_mnemonic"]
