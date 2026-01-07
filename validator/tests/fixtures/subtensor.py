from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from caster_validator.application.ports.subtensor import (
    CommitmentRecord,
    MetagraphSnapshot,
    SubtensorClientPort,
    ValidatorNodeInfo,
)


@dataclass
class FakeSubtensorClient(SubtensorClientPort):
    """Test double used to verify subtensor integrations."""

    metagraph: MetagraphSnapshot = field(
        default_factory=lambda: MetagraphSnapshot(uids=(), hotkeys=()),
    )
    commitments_by_uid: dict[int, CommitmentRecord] = field(default_factory=dict)
    validator_metadata: ValidatorNodeInfo = field(
        default_factory=lambda: ValidatorNodeInfo(uid=-1, version_key=None),
    )
    weight_updates: list[Mapping[int, float]] = field(default_factory=list)
    connected: bool = False
    weights: dict[int, float] = field(default_factory=dict)
    tx_hashes: list[str] = field(default_factory=list)
    current_block_height: int = 0
    last_update_by_uid: dict[int, int] = field(default_factory=dict)
    tempo_by_netuid: dict[int, int] = field(default_factory=dict)
    next_epoch_start_by_netuid: dict[int, int] = field(default_factory=dict)

    def connect(self) -> None:
        self.connected = True

    def fetch_metagraph(self) -> MetagraphSnapshot:
        return self.metagraph

    def fetch_commitment(self, uid: int) -> CommitmentRecord | None:
        if uid < 0:
            return None
        return self.commitments_by_uid.get(uid)

    def publish_commitment(
        self,
        data: str,
        *,
        blocks_until_reveal: int = 1,
    ) -> CommitmentRecord:
        record = CommitmentRecord(block=self.current_block_height, data=data)
        uid = self.validator_metadata.uid
        if uid >= 0:
            self.commitments_by_uid[uid] = record
        return record

    def current_block(self) -> int:
        return self.current_block_height

    def last_update_block(self, uid: int) -> int | None:
        return self.last_update_by_uid.get(uid)

    def validator_info(self) -> ValidatorNodeInfo:
        return self.validator_metadata

    def submit_weights(self, weights: Mapping[int, float]) -> str:
        self.weight_updates.append(dict(weights))
        self.weights.update({int(uid): float(value) for uid, value in weights.items()})
        tx_hash = f"0x{len(self.weight_updates):08x}"
        self.tx_hashes.append(tx_hash)
        uid = self.validator_metadata.uid
        if uid >= 0:
            self.last_update_by_uid[uid] = self.current_block_height
        return tx_hash

    def fetch_weight(self, uid: int) -> float:
        return float(self.weights.get(uid, 0.0))

    def tempo(self, netuid: int) -> int:
        return int(self.tempo_by_netuid.get(netuid, 360))

    def get_next_epoch_start_block(
        self,
        netuid: int,
        *,
        reference_block: int | None = None,
    ) -> int:
        if netuid in self.next_epoch_start_by_netuid:
            return int(self.next_epoch_start_by_netuid[netuid])
        tempo = max(1, self.tempo(netuid))
        ref = self.current_block_height if reference_block is None else reference_block
        cycle = tempo + 1
        offset = (ref + netuid + 1) % cycle
        if offset == 0:
            return ref
        return ref + (cycle - offset)


__all__ = ["FakeSubtensorClient"]
