"""Runtime subtensor client that defers to the configured provider."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping

from caster_commons.config.subtensor import SubtensorSettings
from caster_validator.application.ports.subtensor import (
    CommitmentRecord,
    MetagraphSnapshot,
    SubtensorClientPort,
    ValidatorNodeInfo,
)

from .bittensor import BittensorSubtensorClient


class RuntimeSubtensorClient(SubtensorClientPort):
    """Concrete client used by the runtime, pluggable for tests."""

    def __init__(
        self,
        settings: SubtensorSettings,
        *,
        client_factory: Callable[[SubtensorSettings], SubtensorClientPort] | None = None,
    ) -> None:
        self._settings = settings
        self._factory = client_factory or (lambda cfg: BittensorSubtensorClient(cfg))
        self._client: SubtensorClientPort | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # helpers

    def _delegate(self) -> SubtensorClientPort:
        if self._client is None:
            self._client = self._factory(self._settings)
            self._client.connect()
        return self._client

    # ------------------------------------------------------------------
    # port implementation

    def connect(self) -> None:
        with self._lock:
            self._delegate().connect()

    def fetch_metagraph(self) -> MetagraphSnapshot:
        with self._lock:
            return self._delegate().fetch_metagraph()

    def fetch_commitment(self, uid: int) -> CommitmentRecord | None:
        with self._lock:
            return self._delegate().fetch_commitment(uid)

    def publish_commitment(
        self,
        data: str,
        *,
        blocks_until_reveal: int = 1,
    ) -> CommitmentRecord:
        with self._lock:
            return self._delegate().publish_commitment(
                data,
                blocks_until_reveal=blocks_until_reveal,
            )

    def current_block(self) -> int:
        with self._lock:
            return self._delegate().current_block()

    def last_update_block(self, uid: int) -> int | None:
        with self._lock:
            return self._delegate().last_update_block(uid)

    def validator_info(self) -> ValidatorNodeInfo:
        with self._lock:
            return self._delegate().validator_info()

    def submit_weights(self, weights: Mapping[int, float]) -> str:
        with self._lock:
            return self._delegate().submit_weights(weights)

    def fetch_weight(self, uid: int) -> float:
        with self._lock:
            return self._delegate().fetch_weight(uid)

    def tempo(self, netuid: int) -> int:
        with self._lock:
            return self._delegate().tempo(netuid)

    def get_next_epoch_start_block(
        self,
        netuid: int,
        *,
        reference_block: int | None = None,
    ) -> int:
        with self._lock:
            return self._delegate().get_next_epoch_start_block(
                netuid,
                reference_block=reference_block,
            )

    def close(self) -> None:
        with self._lock:
            if self._client is None:
                return
            self._client.close()


__all__ = ["RuntimeSubtensorClient"]
