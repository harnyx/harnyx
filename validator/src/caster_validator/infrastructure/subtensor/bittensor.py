"""Bittensor-backed Subtensor client used by the validator runtime."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import bittensor as bt

from caster_commons.config.subtensor import SubtensorSettings
from caster_validator.application.ports.subtensor import (
    CommitmentRecord,
    MetagraphSnapshot,
    SubtensorClientPort,
    ValidatorNodeInfo,
)

logger = logging.getLogger("caster_validator.subtensor")


@dataclass(slots=True)
class BittensorSubtensorClient(SubtensorClientPort):
    """Synchronous wrapper around ``bt.Subtensor``."""

    settings: SubtensorSettings

    def __post_init__(self) -> None:
        self._subtensor: bt.Subtensor | None = None
        self._wallet: bt.wallet.Wallet | None = None

    # ------------------------------------------------------------------
    # lifecycle helpers

    def connect(self) -> None:
        self._ensure_ready()

    def close(self) -> None:
        subtensor = self._subtensor
        if subtensor is None:
            return
        try:
            subtensor.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.debug("subtensor close failed", exc_info=exc)
        finally:
            self._subtensor = None

    def _ensure_ready(self) -> None:
        if self._subtensor is None:
            self._subtensor = self._create_subtensor()
        if self._wallet is None:
            self._wallet = bt.wallet(
                name=self.settings.wallet_name,
                hotkey=self.settings.hotkey_name,
            )

    def _create_subtensor(self) -> bt.Subtensor:
        endpoint = self.settings.endpoint.strip()
        network_or_endpoint = endpoint or self.settings.network
        return bt.Subtensor(network=network_or_endpoint)

    def _require_subtensor(self) -> bt.Subtensor:
        if self._subtensor is None:
            raise RuntimeError("subtensor client not initialized")
        return self._subtensor

    def _require_wallet(self) -> bt.wallet.Wallet:
        if self._wallet is None:
            raise RuntimeError("wallet not initialized")
        return self._wallet

    # ------------------------------------------------------------------
    # port implementation

    def fetch_metagraph(self) -> MetagraphSnapshot:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        metagraph = subtensor.metagraph(self.settings.netuid)
        uids = tuple(int(uid) for uid in metagraph.uids)
        hotkeys = tuple(metagraph.hotkeys)
        return MetagraphSnapshot(uids=uids, hotkeys=hotkeys)

    def fetch_commitment(self, uid: int) -> CommitmentRecord | None:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        commitment = subtensor.get_revealed_commitment(
            netuid=self.settings.netuid,
            uid=uid,
        )
        if not commitment:
            return None
        latest_block, data = max((int(entry[0]), entry[1]) for entry in commitment)
        return CommitmentRecord(block=latest_block, data=str(data))

    def publish_commitment(
        self,
        data: str,
        *,
        blocks_until_reveal: int = 1,
    ) -> CommitmentRecord:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        wallet = self._require_wallet()
        success, _ = subtensor.set_reveal_commitment(
            wallet=wallet,
            netuid=self.settings.netuid,
            data=data,
            blocks_until_reveal=max(1, blocks_until_reveal),
            period=self.settings.transaction_period,
        )
        if not success:
            raise RuntimeError("set_reveal_commitment failed")
        block_number = self._read_block_number()
        return CommitmentRecord(block=block_number, data=data)

    def current_block(self) -> int:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        return int(subtensor.get_current_block())

    def last_update_block(self, uid: int) -> int | None:
        if uid < 0:
            return None
        self._ensure_ready()
        subtensor = self._require_subtensor()
        metagraph = subtensor.metagraph(self.settings.netuid)
        last_update = metagraph.last_update
        if last_update is None or uid >= len(last_update):
            return None
        return int(last_update[uid])

    def validator_info(self) -> ValidatorNodeInfo:
        snapshot = self.fetch_metagraph()
        wallet = self._require_wallet()
        hotkey = wallet.hotkey
        if hotkey is None:
            raise RuntimeError("wallet hotkey is unavailable")
        hotkey_addr = hotkey.ss58_address
        uid = -1
        if hotkey_addr and hotkey_addr in snapshot.hotkeys:
            uid = snapshot.hotkeys.index(hotkey_addr)
        version_key = self._query_version_key()
        return ValidatorNodeInfo(uid=uid, version_key=version_key)

    def submit_weights(self, weights: Mapping[int, float]) -> str:
        if not weights:
            raise ValueError("weights mapping must not be empty")
        self._ensure_ready()
        subtensor = self._require_subtensor()
        wallet = self._require_wallet()
        uids, normalized = self._normalize_weights(weights)
        kwargs = self._set_weights_kwargs(wallet, uids, normalized)
        logger.debug(
            "calling subtensor.set_weights",
            extra={"uids": uids, "wait_for_inclusion": self.settings.wait_for_inclusion},
        )
        success, message = subtensor.set_weights(**kwargs)
        logger.debug(
            "subtensor.set_weights returned",
            extra={"success": success, "message": message},
        )
        if not success:
            raise RuntimeError(f"set_weights failed: {message}")
        return str(message) if message is not None else ""

    def _normalize_weights(self, weights: Mapping[int, float]) -> tuple[list[int], list[float]]:
        ordered = sorted(weights.items(), key=lambda item: item[0])
        uids = [int(uid) for uid, _ in ordered]
        values = [float(score) for _, score in ordered]
        total = sum(values)
        if math.isclose(total, 0.0, abs_tol=1e-9):
            raise ValueError("weight totals must be greater than zero")
        normalized = [value / total for value in values]
        return uids, normalized

    def _set_weights_kwargs(
        self,
        wallet: bt.wallet.Wallet,
        uids: list[int],
        normalized: list[float],
    ) -> dict[str, object]:
        return {
            "wallet": wallet,
            "netuid": self.settings.netuid,
            "weights": normalized,
            "uids": uids,
            "wait_for_inclusion": self.settings.wait_for_inclusion,
            "wait_for_finalization": self.settings.wait_for_finalization,
            "period": self.settings.transaction_period,
        }

    def fetch_weight(self, uid: int) -> float:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        weights = subtensor.weights(netuid=self.settings.netuid)
        if uid < 0 or uid >= len(weights):
            return 0.0
        return float(weights[uid])

    def tempo(self, netuid: int) -> int:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        tempo = subtensor.tempo(netuid=netuid)
        if tempo is None:
            raise RuntimeError("tempo is unavailable")
        return int(tempo)

    def get_next_epoch_start_block(
        self,
        netuid: int,
        *,
        reference_block: int | None = None,
    ) -> int:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        next_block = subtensor.get_next_epoch_start_block(netuid, block=reference_block)
        if next_block is None:
            raise RuntimeError("unable to determine next epoch start block")
        return int(next_block)

    # ------------------------------------------------------------------
    # helpers

    def _read_block_number(self) -> int:
        try:
            subtensor = self._require_subtensor()
            return int(subtensor.get_current_block())
        except Exception:  # pragma: no cover - informational fallback
            return -1

    def _query_version_key(self) -> int | None:
        try:
            subtensor = self._require_subtensor()
            return int(subtensor.weights_version(self.settings.netuid))
        except Exception:  # pragma: no cover - optional metadata
            return None


__all__ = ["BittensorSubtensorClient"]
