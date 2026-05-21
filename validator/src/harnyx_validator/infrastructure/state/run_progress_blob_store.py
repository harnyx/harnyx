"""Blob storage for completed miner-task run submissions."""

from __future__ import annotations

import mmap
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Final
from uuid import UUID

from pydantic import TypeAdapter

from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission

_FRAME_HEADER_SIZE: Final[int] = 8
_SUBMISSION_ADAPTER: Final[TypeAdapter[MinerTaskRunSubmission]] = TypeAdapter(
    MinerTaskRunSubmission
)


@dataclass(frozen=True, slots=True)
class RunSubmissionBlobRef:
    batch_id: UUID
    sequence: int
    segment_name: str
    payload_offset: int
    payload_length: int
    frame_offset: int
    frame_length: int
    sha256: str


@dataclass(slots=True)
class _SegmentWriter:
    name: str
    path: Path
    next_offset: int


class RunSubmissionBlobStore:
    """Append-only length-framed blob store for completed run submissions."""

    def __init__(
        self,
        root_dir: Path,
        *,
        segment_size_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if segment_size_bytes <= _FRAME_HEADER_SIZE:
            raise ValueError("segment_size_bytes must be larger than the frame header")
        self.root_dir = root_dir.expanduser()
        self.segment_size_bytes = segment_size_bytes
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._writers_by_batch: dict[UUID, _SegmentWriter] = {}

    def append(
        self,
        *,
        batch_id: UUID,
        sequence: int,
        submission: MinerTaskRunSubmission,
    ) -> RunSubmissionBlobRef:
        payload = _SUBMISSION_ADAPTER.dump_json(submission)
        frame_length = _FRAME_HEADER_SIZE + len(payload)
        writer = self._writer_for(batch_id=batch_id, frame_length=frame_length)
        frame_offset = writer.next_offset
        payload_offset = frame_offset + _FRAME_HEADER_SIZE
        writer.path.parent.mkdir(parents=True, exist_ok=True)
        with writer.path.open("ab") as handle:
            handle.write(len(payload).to_bytes(_FRAME_HEADER_SIZE, byteorder="big"))
            handle.write(payload)
            handle.flush()
        writer.next_offset += frame_length
        return RunSubmissionBlobRef(
            batch_id=batch_id,
            sequence=sequence,
            segment_name=writer.name,
            payload_offset=payload_offset,
            payload_length=len(payload),
            frame_offset=frame_offset,
            frame_length=frame_length,
            sha256=sha256(payload).hexdigest(),
        )

    def read(self, ref: RunSubmissionBlobRef) -> MinerTaskRunSubmission:
        return self.read_many((ref,))[0]

    def read_many(
        self,
        refs: Sequence[RunSubmissionBlobRef],
    ) -> tuple[MinerTaskRunSubmission, ...]:
        if not refs:
            return ()
        ordered: list[MinerTaskRunSubmission | None] = [None] * len(refs)
        refs_by_segment: dict[tuple[UUID, str], list[tuple[int, RunSubmissionBlobRef]]] = {}
        for index, ref in enumerate(refs):
            refs_by_segment.setdefault((ref.batch_id, ref.segment_name), []).append((index, ref))

        for (batch_id, segment_name), segment_refs in refs_by_segment.items():
            segment_path = self._batch_dir(batch_id) / segment_name
            with segment_path.open("rb") as handle:
                with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
                    for index, ref in segment_refs:
                        payload = _payload_slice(mapped, ref)
                        ordered[index] = _SUBMISSION_ADAPTER.validate_json(payload)

        if any(item is None for item in ordered):
            raise RuntimeError("blob store read did not hydrate every requested submission")
        return tuple(item for item in ordered if item is not None)

    def delete_batch(self, batch_id: UUID) -> bool:
        writer_existed = self._writers_by_batch.pop(batch_id, None) is not None
        batch_dir = self._batch_dir(batch_id)
        if not batch_dir.exists():
            return writer_existed
        shutil.rmtree(batch_dir)
        return True

    def prune_stale_batch_dirs(
        self,
        *,
        cutoff: datetime,
        protected_batch_ids: frozenset[UUID],
    ) -> tuple[UUID, ...]:
        removed: list[UUID] = []
        if not self.root_dir.exists():
            return ()

        normalized_cutoff = _as_utc(cutoff)
        for child in self.root_dir.iterdir():
            batch_id = _uuid_from_name(child.name)
            if batch_id is None or batch_id in protected_batch_ids or not child.is_dir():
                continue
            if _latest_mtime(child) > normalized_cutoff:
                continue
            self._writers_by_batch.pop(batch_id, None)
            shutil.rmtree(child)
            removed.append(batch_id)
        return tuple(removed)

    def _writer_for(self, *, batch_id: UUID, frame_length: int) -> _SegmentWriter:
        writer = self._writers_by_batch.get(batch_id)
        if writer is not None and writer.next_offset + frame_length <= self.segment_size_bytes:
            return writer
        segment_index = 1 if writer is None else _segment_index(writer.name) + 1
        name = f"runs-{segment_index:06d}.blob"
        path = self._batch_dir(batch_id) / name
        next_offset = path.stat().st_size if path.exists() else 0
        writer = _SegmentWriter(name=name, path=path, next_offset=next_offset)
        self._writers_by_batch[batch_id] = writer
        return writer

    def _batch_dir(self, batch_id: UUID) -> Path:
        return self.root_dir / str(batch_id)


def _payload_slice(mapped: mmap.mmap, ref: RunSubmissionBlobRef) -> bytes:
    frame_end = ref.frame_offset + ref.frame_length
    payload_end = ref.payload_offset + ref.payload_length
    if frame_end > len(mapped) or payload_end > len(mapped):
        raise RuntimeError("blob ref points past end of segment")
    encoded_length = mapped[ref.frame_offset : ref.frame_offset + _FRAME_HEADER_SIZE]
    payload_length = int.from_bytes(encoded_length, byteorder="big")
    if payload_length != ref.payload_length:
        raise RuntimeError("blob frame length mismatch")
    payload = bytes(mapped[ref.payload_offset:payload_end])
    if sha256(payload).hexdigest() != ref.sha256:
        raise RuntimeError("blob payload checksum mismatch")
    return payload


def _segment_index(name: str) -> int:
    prefix = "runs-"
    suffix = ".blob"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise RuntimeError(f"unexpected run-progress segment name {name!r}")
    return int(name[len(prefix) : -len(suffix)])


def _uuid_from_name(name: str) -> UUID | None:
    try:
        return UUID(name)
    except ValueError:
        return None


def _latest_mtime(path: Path) -> datetime:
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        latest = max(latest, child.stat().st_mtime)
    return datetime.fromtimestamp(latest, UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["RunSubmissionBlobRef", "RunSubmissionBlobStore"]
