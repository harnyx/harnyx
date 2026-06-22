"""Bittensor-backed Subtensor client used by the validator runtime."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import bittensor as bt
from bittensor.core.errors import MetadataError
from bittensor.core.extrinsics.pallets import SubtensorModule
from bittensor.core.extrinsics.weights import set_weights_extrinsic
from bittensor.core.settings import version_as_int
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import get_mechid_storage_index
from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids
from bittensor_drand import get_encrypted_commit, get_encrypted_commitment
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_validator.application.ports.subtensor import (
    CommitmentRecord,
    MetagraphSnapshot,
    SubtensorClientPort,
    ValidatorNodeInfo,
    WeightSubmissionTooEarlyError,
)
from harnyx_validator.infrastructure.transient_network import classify_transient_network_failure

from .hotkey import create_wallet

logger = logging.getLogger("harnyx_validator.subtensor")

_COMMIT_REVEAL_MAX_RETRIES = 5
_PLAIN_SET_WEIGHTS_MAX_RETRIES = 5
_COMMIT_REVEAL_VERSION = 4
_PRIMARY_MECHANISM_ID = 0
_DEFAULT_BLOCK_TIME_SECONDS = 12.0
_NO_WEIGHT_ATTEMPT_MESSAGE = "No attempt made. Perhaps it is too soon to commit weights!"
_CHAIN_TOO_EARLY_REFUSAL_ERROR_NAMES = frozenset(
    {
        "SettingWeightsTooFast",
        "CommittingWeightsTooFast",
    }
)
_LastUpdateValue: TypeAlias = int | None
_LastUpdateValues: TypeAlias = dict[int, _LastUpdateValue] | list[_LastUpdateValue]


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


@dataclass(frozen=True, slots=True)
class _WeightExtrinsicResult:
    success: bool
    message: str
    error_name: str | None = None


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
        last_update = self._query_last_update_values(
            subtensor=subtensor,
            netuid=self.settings.netuid,
        )
        if last_update is None:
            return None
        return self._last_update_for_uid(last_update, uid)

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
        commit_reveal_enabled = self._query_commit_reveal_enabled(
            subtensor=subtensor,
            netuid=self.settings.netuid,
        )
        if commit_reveal_enabled is None:
            raise RuntimeError("commit-reveal metadata unavailable")
        logger.debug(
            "submitting weights to subtensor",
            extra={"uids": uids, "wait_for_inclusion": self.settings.wait_for_inclusion},
        )
        if commit_reveal_enabled:
            success, message = self._submit_commit_reveal_weights(
                subtensor=subtensor,
                wallet=wallet,
                uids=uids,
                normalized=normalized,
            )
        else:
            success, message = self._submit_plain_weights(
                subtensor=subtensor,
                wallet=wallet,
                uids=uids,
                normalized=normalized,
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
    ) -> _WeightExtrinsicResult:
        wallet = self._require_wallet()
        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            sign_with="hotkey",
            use_nonce=True,
            nonce_key="hotkey",
            period=self.settings.transaction_period,
        )
        return self._extrinsic_response_result(response)

    def _weight_submission_wait_for_inclusion(self) -> bool:
        return self.settings.wait_for_inclusion or not self.settings.wait_for_finalization

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
        result = self._submit_hotkey_extrinsic(
            subtensor=subtensor,
            call=call,
            wait_for_inclusion=False,
            wait_for_finalization=True,
        )
        return result.success, result.message

    def _submit_commit_reveal_weights(
        self,
        *,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        uids: list[int],
        normalized: list[float],
    ) -> tuple[bool, str]:
        message = _NO_WEIGHT_ATTEMPT_MESSAGE
        transient_failure: Exception | None = None
        all_failures_transient = True
        for _ in range(_COMMIT_REVEAL_MAX_RETRIES):
            try:
                call, reveal_round = self._build_commit_reveal_call(
                    subtensor=subtensor,
                    wallet=wallet,
                    uids=uids,
                    normalized=normalized,
                )
                result = self._submit_hotkey_extrinsic(
                    subtensor=subtensor,
                    call=call,
                    wait_for_inclusion=self._weight_submission_wait_for_inclusion(),
                    wait_for_finalization=self.settings.wait_for_finalization,
                )
            except Exception as exc:
                if self._error_name_from_object(exc) in _CHAIN_TOO_EARLY_REFUSAL_ERROR_NAMES:
                    raise WeightSubmissionTooEarlyError(str(exc)) from exc
                cause = classify_transient_network_failure(exc)
                if cause is None:
                    all_failures_transient = False
                    transient_failure = None
                else:
                    transient_failure = exc
                logger.warning("commit-reveal weight submission attempt failed", exc_info=exc)
                message = str(exc)
                continue
            message = result.message
            if self._is_chain_too_early_refusal(result):
                raise WeightSubmissionTooEarlyError(result.message)
            if result.success:
                return True, f"reveal_round:{reveal_round}"
            all_failures_transient = False
            transient_failure = None
            logger.warning(
                "commit-reveal weight submission attempt failed",
                extra={"set_weights_message": message, "set_weights_error": result.error_name},
            )
        if all_failures_transient and transient_failure is not None:
            raise RuntimeError("commit-reveal weight submission attempts failed") from transient_failure
        return False, message

    def _submit_plain_weights(
        self,
        *,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        uids: list[int],
        normalized: list[float],
    ) -> tuple[bool, str]:
        message = _NO_WEIGHT_ATTEMPT_MESSAGE
        for _ in range(_PLAIN_SET_WEIGHTS_MAX_RETRIES):
            response = set_weights_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                netuid=self.settings.netuid,
                mechid=_PRIMARY_MECHANISM_ID,
                uids=uids,
                weights=normalized,
                version_key=version_as_int,
                mev_protection=False,
                wait_for_inclusion=self._weight_submission_wait_for_inclusion(),
                wait_for_finalization=self.settings.wait_for_finalization,
                period=self.settings.transaction_period,
            )
            result = self._extrinsic_response_result(response)
            message = result.message
            if self._is_chain_too_early_refusal(result):
                raise WeightSubmissionTooEarlyError(result.message)
            if result.success:
                return True, message
            logger.warning(
                "plain weight submission attempt failed",
                extra={"set_weights_message": message, "set_weights_error": result.error_name},
            )
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
            netuid=get_mechid_storage_index(
                netuid=self.settings.netuid,
                mechid=_PRIMARY_MECHANISM_ID,
            ),
            subnet_reveal_period_epochs=int(commit_reveal_period),
            block_time=_DEFAULT_BLOCK_TIME_SECONDS,
            hotkey=hotkey_public_key,
        )
        call = SubtensorModule(subtensor).commit_timelocked_mechanism_weights(
            netuid=self.settings.netuid,
            mecid=_PRIMARY_MECHANISM_ID,
            commit=commit,
            reveal_round=reveal_round,
            commit_reveal_version=_COMMIT_REVEAL_VERSION,
        )
        return call, reveal_round

    def _read_block_number(self) -> int:
        try:
            subtensor = self._require_subtensor()
            return int(subtensor.get_current_block())
        except Exception:  # pragma: no cover - informational fallback
            return -1

    @classmethod
    def _extrinsic_response_result(
        cls,
        response: ExtrinsicResponse | Sequence[object],
    ) -> _WeightExtrinsicResult:
        if isinstance(response, Sequence) and not isinstance(response, (str, bytes, bytearray)):
            success = bool(response[0]) if len(response) > 0 else False
            message = "" if len(response) < 2 or response[1] is None else str(response[1])
            return _WeightExtrinsicResult(success=success, message=message)

        success = bool(getattr(response, "success", False))
        raw_message = getattr(response, "message", None)
        message = "" if raw_message is None else str(raw_message)
        return _WeightExtrinsicResult(
            success=success,
            message=message,
            error_name=cls._extrinsic_error_name(response),
        )

    @staticmethod
    def _is_chain_too_early_refusal(result: _WeightExtrinsicResult) -> bool:
        return result.error_name in _CHAIN_TOO_EARLY_REFUSAL_ERROR_NAMES

    @classmethod
    def _extrinsic_error_name(cls, response: ExtrinsicResponse | object) -> str | None:
        error_name = cls._error_name_from_object(getattr(response, "error", None))
        if error_name is not None:
            return error_name
        return cls._error_name_from_object(getattr(response, "message", None))

    @classmethod
    def _error_name_from_object(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            mapping_value = cast(Mapping[object, object], value)
            for key in ("name", "error", "error_name"):
                nested = mapping_value.get(key)
                if isinstance(nested, str) and nested in _CHAIN_TOO_EARLY_REFUSAL_ERROR_NAMES:
                    return nested
            for nested in mapping_value.values():
                error_name = cls._error_name_from_object(nested)
                if error_name is not None:
                    return error_name
            return None
        if isinstance(value, BaseException):
            for nested in value.args:
                if isinstance(nested, Mapping):
                    error_name = cls._error_name_from_object(nested)
                    if error_name is not None:
                        return error_name
            for attr in ("error", "error_name"):
                nested = getattr(value, attr, None)
                error_name = cls._error_name_from_object(nested)
                if error_name is not None:
                    return error_name
            return None
        for attr in ("name", "error", "error_name"):
            nested = getattr(value, attr, None)
            if isinstance(nested, str) and nested in _CHAIN_TOO_EARLY_REFUSAL_ERROR_NAMES:
                return nested
        return None

    @staticmethod
    def _query_commit_reveal_enabled(*, subtensor: bt.Subtensor, netuid: int) -> bool | None:
        try:
            return bool(subtensor.commit_reveal_enabled(netuid=netuid))
        except Exception as exc:
            logger.debug("unable to read commit-reveal flag", exc_info=exc)
            return None

    @staticmethod
    def _query_last_update_values(*, subtensor: bt.Subtensor, netuid: int) -> _LastUpdateValues | None:
        try:
            values = subtensor.get_hyperparameter(param_name="LastUpdate", netuid=netuid)
        except Exception as exc:
            logger.debug("unable to read LastUpdate metadata", exc_info=exc)
            return None
        return BittensorSubtensorClient._normalize_last_update_values(values)

    @staticmethod
    def _normalize_last_update_values(values: object) -> _LastUpdateValues | None:
        if isinstance(values, Mapping):
            normalized: dict[int, _LastUpdateValue] = {}
            for key, value in values.items():
                if not isinstance(key, int) or (value is not None and not isinstance(value, int)):
                    return None
                normalized[key] = value
            return normalized
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            return None
        normalized_values: list[_LastUpdateValue] = []
        for value in values:
            if value is not None and not isinstance(value, int):
                return None
            normalized_values.append(value)
        return normalized_values

    @staticmethod
    def _last_update_for_uid(values: _LastUpdateValues, uid: int) -> int | None:
        if uid < 0:
            return None
        try:
            if isinstance(values, dict):
                value = values.get(uid)
            else:
                if uid >= len(values):
                    return None
                value = values[uid]
        except IndexError:
            return None
        return value

    def _query_version_key(self) -> int | None:
        try:
            subtensor = self._require_subtensor()
            subtensor_any = cast(Any, subtensor)
            return int(subtensor_any.weights_version(self.settings.netuid))
        except Exception:  # pragma: no cover - optional metadata
            return None


__all__ = ["BittensorSubtensorClient"]
