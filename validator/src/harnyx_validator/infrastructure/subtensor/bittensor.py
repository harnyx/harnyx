"""Bittensor-backed Subtensor client used by the validator runtime."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import bittensor as bt
from bittensor.core.errors import MetadataError
from bittensor.core.settings import version_as_int
from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids
from bittensor_drand import get_encrypted_commit, get_encrypted_commitment
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_validator.application.ports.subtensor import (
    CommitmentRecord,
    MetagraphSnapshot,
    SubtensorClientPort,
    ValidatorNodeInfo,
)

from .hotkey import create_wallet

logger = logging.getLogger("harnyx_validator.subtensor")

_COMMIT_REVEAL_MAX_RETRIES = 5
_COMMIT_REVEAL_VERSION = 4
_DEFAULT_BLOCK_TIME_SECONDS = 12.0
_NO_WEIGHT_ATTEMPT_MESSAGE = "No attempt made. Perhaps it is too soon to commit weights!"


class _SubtensorWeightTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: int
    weight: int

    @model_validator(mode="before")
    @classmethod
    def _from_wire(cls, value: object) -> object:
        if isinstance(value, tuple) and len(value) == 2:
            uid, weight = value
            return {"uid": uid, "weight": weight}
        return value


class _SubtensorWeightRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_uid: int
    targets: list[_SubtensorWeightTarget]

    @model_validator(mode="before")
    @classmethod
    def _from_wire(cls, value: object) -> object:
        if isinstance(value, tuple) and len(value) == 2:
            source_uid, targets = value
            return {"source_uid": source_uid, "targets": targets}
        return value


class _SubtensorWeightsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[_SubtensorWeightRow]


@dataclass(slots=True)
class BittensorSubtensorClient(SubtensorClientPort):
    """Synchronous wrapper around ``bt.Subtensor``."""

    settings: SubtensorSettings

    def __post_init__(self) -> None:
        self._subtensor: bt.Subtensor | None = None
        self._wallet: bt.Wallet | None = None

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
            self._wallet = create_wallet(self.settings)

    def _create_subtensor(self) -> bt.Subtensor:
        endpoint = self.settings.endpoint.strip()
        network_or_endpoint = endpoint or self.settings.network
        return bt.Subtensor(network=network_or_endpoint)

    def _require_subtensor(self) -> bt.Subtensor:
        if self._subtensor is None:
            raise RuntimeError("subtensor client not initialized")
        return self._subtensor

    def _require_wallet(self) -> bt.Wallet:
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
        success, message = self._publish_commitment_extrinsic(
            subtensor=subtensor,
            data=data,
            blocks_until_reveal=max(1, blocks_until_reveal),
        )
        if not success:
            raise MetadataError(message)
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
        logger.debug(
            "calling subtensor.set_weights",
            extra={"uids": uids, "wait_for_inclusion": self.settings.wait_for_inclusion},
        )
        if subtensor.commit_reveal_enabled(netuid=self.settings.netuid):
            success, message = self._submit_commit_reveal_weights(
                subtensor=subtensor,
                wallet=wallet,
                uids=uids,
                normalized=normalized,
            )
        else:
            success, message = subtensor.set_weights(
                wallet=wallet,
                netuid=self.settings.netuid,
                weights=normalized,
                uids=uids,
                wait_for_inclusion=self.settings.wait_for_inclusion,
                wait_for_finalization=self.settings.wait_for_finalization,
                period=self.settings.transaction_period,
            )
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

    def fetch_weight(self, uid: int) -> float:
        if uid < 0:
            return 0.0
        self._ensure_ready()
        validator_uid = self.validator_info().uid
        if validator_uid < 0:
            return 0.0
        subtensor = self._require_subtensor()
        raw_weights = subtensor.weights(netuid=self.settings.netuid)
        try:
            payload = _SubtensorWeightsPayload.model_validate(
                {"rows": raw_weights},
                strict=True,
            )
        except ValidationError as exc:
            raise RuntimeError(f"invalid subtensor weights payload: {exc}") from exc

        row = next((candidate for candidate in payload.rows if candidate.source_uid == validator_uid), None)
        if row is None:
            return 0.0

        target = next((candidate for candidate in row.targets if candidate.uid == uid), None)
        if target is None:
            return 0.0
        return float(target.weight)

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

    def _submit_hotkey_extrinsic(
        self,
        *,
        subtensor: bt.Subtensor,
        call: Any,
        wait_for_inclusion: bool,
        wait_for_finalization: bool,
    ) -> tuple[bool, str]:
        wallet = self._require_wallet()
        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            sign_with="hotkey",
            use_nonce=True,
            nonce_key="hotkey",
            period=self.settings.transaction_period,
        )

    def _publish_commitment_extrinsic(
        self,
        *,
        subtensor: bt.Subtensor,
        data: str,
        blocks_until_reveal: int,
    ) -> tuple[bool, str]:
        encrypted, reveal_round = get_encrypted_commitment(
            data,
            blocks_until_reveal=blocks_until_reveal,
            block_time=_DEFAULT_BLOCK_TIME_SECONDS,
        )
        call = subtensor.substrate.compose_call(
            call_module="Commitments",
            call_function="set_commitment",
            call_params={
                "netuid": self.settings.netuid,
                "info": {
                    "fields": [[{"TimelockEncrypted": {"encrypted": encrypted, "reveal_round": reveal_round}}]]
                },
            },
        )
        return self._submit_hotkey_extrinsic(
            subtensor=subtensor,
            call=call,
            wait_for_inclusion=False,
            wait_for_finalization=True,
        )

    def _submit_commit_reveal_weights(
        self,
        *,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        uids: list[int],
        normalized: list[float],
    ) -> tuple[bool, str]:
        validator_uid = subtensor.get_uid_for_hotkey_on_subnet(
            wallet.hotkey.ss58_address,
            self.settings.netuid,
        )
        if validator_uid is None:
            return (
                False,
                f"Hotkey {wallet.hotkey.ss58_address} not registered in subnet {self.settings.netuid}",
            )

        blocks_since_last_update = subtensor.blocks_since_last_update(
            self.settings.netuid,
            int(validator_uid),
        )
        weights_rate_limit = subtensor.weights_rate_limit(self.settings.netuid)
        if (
            blocks_since_last_update is None
            or weights_rate_limit is None
            or int(blocks_since_last_update) <= int(weights_rate_limit)
        ):
            return False, _NO_WEIGHT_ATTEMPT_MESSAGE

        success = False
        message = _NO_WEIGHT_ATTEMPT_MESSAGE
        for _ in range(_COMMIT_REVEAL_MAX_RETRIES):
            try:
                call, reveal_round = self._build_commit_reveal_call(
                    subtensor=subtensor,
                    wallet=wallet,
                    uids=uids,
                    normalized=normalized,
                )
                success, message = self._submit_hotkey_extrinsic(
                    subtensor=subtensor,
                    call=call,
                    wait_for_inclusion=self.settings.wait_for_inclusion,
                    wait_for_finalization=self.settings.wait_for_finalization,
                )
            except Exception as exc:
                logger.warning("commit-reveal weight submission attempt failed", exc_info=exc)
                success = False
                message = str(exc)
            if success:
                return True, f"reveal_round:{reveal_round}"
        return False, message

    def _build_commit_reveal_call(
        self,
        *,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        uids: list[int],
        normalized: list[float],
    ) -> tuple[Any, int]:
        weight_uids, weight_vals = convert_and_normalize_weights_and_uids(uids, normalized)
        current_block = int(subtensor.get_current_block())
        hyperparameters = subtensor.get_subnet_hyperparameters(
            self.settings.netuid,
            block=current_block,
        )
        tempo = getattr(hyperparameters, "tempo", None)
        commit_reveal_period = getattr(hyperparameters, "commit_reveal_period", None)
        if tempo is None or commit_reveal_period is None:
            raise RuntimeError("subnet hyperparameters unavailable")
        hotkey = wallet.hotkey
        hotkey_public_key = hotkey.public_key if hotkey is not None else None
        if hotkey_public_key is None:
            raise RuntimeError("wallet hotkey public key is unavailable")
        commit, reveal_round = get_encrypted_commit(
            uids=weight_uids,
            weights=weight_vals,
            version_key=version_as_int,
            tempo=int(tempo),
            current_block=current_block,
            netuid=self.settings.netuid,
            subnet_reveal_period_epochs=int(commit_reveal_period),
            block_time=_DEFAULT_BLOCK_TIME_SECONDS,
            hotkey=hotkey_public_key,
        )
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="commit_timelocked_weights",
            call_params={
                "netuid": self.settings.netuid,
                "commit": commit,
                "reveal_round": reveal_round,
                "commit_reveal_version": _COMMIT_REVEAL_VERSION,
            },
        )
        return call, reveal_round

    def _read_block_number(self) -> int:
        try:
            subtensor = self._require_subtensor()
            return int(subtensor.get_current_block())
        except Exception:  # pragma: no cover - informational fallback
            return -1

    def _query_version_key(self) -> int | None:
        try:
            subtensor = self._require_subtensor()
            subtensor_any = cast(Any, subtensor)
            return int(subtensor_any.weights_version(self.settings.netuid))
        except Exception:  # pragma: no cover - optional metadata
            return None


__all__ = ["BittensorSubtensorClient"]
