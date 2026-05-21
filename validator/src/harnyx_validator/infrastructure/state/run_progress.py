"""File-backed tracker for per-batch miner-task progress."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TypeAlias
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission
from harnyx_validator.application.ports.progress import (
    ProviderFailureEvidence,
    RunProgressPage,
    RunProgressSummary,
    SequencedRun,
)
from harnyx_validator.infrastructure.state.run_progress_blob_store import (
    RunSubmissionBlobRef,
    RunSubmissionBlobStore,
)

ProviderEvidenceSnapshot: TypeAlias = ProviderFailureEvidence


@dataclass(slots=True)
class _SessionRunContext:
    batch_id: UUID


@dataclass(slots=True)
class _ProviderEvidenceCounter:
    total_calls: int = 0
    failed_calls: int = 0
    failure_reason: str | None = None


@dataclass(slots=True)
class FileBackedRunProgress:
    storage_root: Path
    segment_size_bytes: int = 64 * 1024 * 1024
    batches_by_id: dict[UUID, MinerTaskBatchSpec] = field(default_factory=dict)
    expected_by_batch: dict[UUID, int] = field(default_factory=dict)
    submission_refs_by_batch: dict[
        UUID,
        dict[tuple[UUID, UUID], RunSubmissionBlobRef],
    ] = field(default_factory=dict)
    sequence_by_pair_by_batch: dict[UUID, dict[tuple[UUID, UUID], int]] = field(default_factory=dict)
    pair_by_sequence_by_batch: dict[UUID, dict[int, tuple[UUID, UUID]]] = field(default_factory=dict)
    next_sequence_by_batch: dict[UUID, int] = field(default_factory=dict)
    session_context_by_id: dict[UUID, _SessionRunContext] = field(default_factory=dict)
    provider_counters_by_batch: dict[
        UUID,
        dict[tuple[str, str], _ProviderEvidenceCounter],
    ] = field(default_factory=dict)
    failed_provider_keys_by_session: dict[UUID, set[tuple[str, str]]] = field(default_factory=dict)
    blob_store: RunSubmissionBlobStore = field(init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.storage_root = self.storage_root.expanduser()
        self.blob_store = RunSubmissionBlobStore(
            self.storage_root,
            segment_size_bytes=self.segment_size_bytes,
        )

    def register(self, batch: MinerTaskBatchSpec) -> None:
        with self._lock:
            existing = self.batches_by_id.get(batch.batch_id)
            if existing is not None:
                if existing != batch:
                    raise RuntimeError("batch_id already exists with different contents")
                return

            self.batches_by_id[batch.batch_id] = batch
            self.expected_by_batch[batch.batch_id] = len(batch.tasks) * len(batch.artifacts)
            self.submission_refs_by_batch.setdefault(batch.batch_id, {})
            self.sequence_by_pair_by_batch.setdefault(batch.batch_id, {})
            self.pair_by_sequence_by_batch.setdefault(batch.batch_id, {})
            self.next_sequence_by_batch.setdefault(batch.batch_id, 1)

    def record(self, result: MinerTaskRunSubmission) -> None:
        with self._lock:
            refs = self.submission_refs_by_batch.setdefault(result.batch_id, {})
            self._record_submission(
                batch_id=result.batch_id,
                refs=refs,
                result=result,
            )

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        submissions: Sequence[MinerTaskRunSubmission],
        provider_evidence: Sequence[ProviderEvidenceSnapshot] = (),
    ) -> None:
        with self._lock:
            self.register(batch)
            staged_refs = dict(self.submission_refs_by_batch.get(batch.batch_id, {}))
            staged_sequence_by_pair = dict(self.sequence_by_pair_by_batch.get(batch.batch_id, {}))
            staged_pair_by_sequence = dict(self.pair_by_sequence_by_batch.get(batch.batch_id, {}))
            next_sequence = int(self.next_sequence_by_batch.get(batch.batch_id, 1))
            for submission in submissions:
                if submission.batch_id != batch.batch_id:
                    raise RuntimeError("restored submission batch_id mismatch")
                next_sequence = self._record_submission(
                    batch_id=batch.batch_id,
                    refs=staged_refs,
                    result=submission,
                    sequence_by_pair=staged_sequence_by_pair,
                    pair_by_sequence=staged_pair_by_sequence,
                    next_sequence=next_sequence,
                    commit_next_sequence=False,
                )
            self.submission_refs_by_batch[batch.batch_id] = staged_refs
            self.sequence_by_pair_by_batch[batch.batch_id] = staged_sequence_by_pair
            self.pair_by_sequence_by_batch[batch.batch_id] = staged_pair_by_sequence
            self.next_sequence_by_batch[batch.batch_id] = next_sequence
            self.provider_counters_by_batch[batch.batch_id] = self._merged_provider_counters(
                batch.batch_id,
                provider_evidence,
            )

    def _merged_provider_counters(
        self,
        batch_id: UUID,
        provider_evidence: Sequence[ProviderEvidenceSnapshot],
    ) -> dict[tuple[str, str], _ProviderEvidenceCounter]:
        existing = self.provider_counters_by_batch.get(batch_id, {})
        merged = dict(existing)
        for entry in provider_evidence:
            key = _provider_model_key(provider=entry["provider"], model=entry["model"])
            restored = _ProviderEvidenceCounter(
                total_calls=entry["total_calls"],
                failed_calls=entry["failed_calls"],
                failure_reason=entry.get("failure_reason"),
            )
            current = merged.get(key)
            if current is None:
                merged[key] = restored
                continue
            merged[key] = _ProviderEvidenceCounter(
                total_calls=max(current.total_calls, restored.total_calls),
                failed_calls=max(current.failed_calls, restored.failed_calls),
                failure_reason=current.failure_reason or restored.failure_reason,
            )
        return merged

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        with self._lock:
            refs = self.submission_refs_by_batch.get(batch_id, {})
            return frozenset(refs)

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        with self._lock:
            self.session_context_by_id[session_id] = _SessionRunContext(batch_id=batch_id)

    def record_provider_call(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
    ) -> None:
        with self._lock:
            key = _provider_model_key(provider=provider, model=model)
            context = self.session_context_by_id.get(session_id)
            if context is None:
                return
            counter = self.provider_counters_by_batch.setdefault(context.batch_id, {}).setdefault(
                key,
                _ProviderEvidenceCounter(),
            )
            counter.total_calls += 1

    def record_provider_failure(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        with self._lock:
            key = _provider_model_key(provider=provider, model=model)
            context = self.session_context_by_id.get(session_id)
            if context is None:
                return
            counter = self.provider_counters_by_batch.setdefault(context.batch_id, {}).setdefault(
                key,
                _ProviderEvidenceCounter(),
            )
            counter.failed_calls += 1
            failure_reason = reason.strip()
            if failure_reason:
                counter.failure_reason = failure_reason
            keys = self.failed_provider_keys_by_session.setdefault(session_id, set())
            keys.add(key)

    def consume_provider_failures(self, session_id: UUID) -> tuple[ProviderEvidenceSnapshot, ...]:
        with self._lock:
            keys = self.failed_provider_keys_by_session.pop(session_id, None)
            if not keys:
                return ()
            context = self.session_context_by_id.get(session_id)
            if context is None:
                return ()
            snapshots: list[ProviderEvidenceSnapshot] = []
            for key in sorted(keys):
                snapshot = self._provider_evidence_snapshot(batch_id=context.batch_id, key=key)
                if snapshot is None:
                    continue
                snapshots.append(snapshot)
            return tuple(snapshots)

    def clear_task_session(self, session_id: UUID) -> None:
        with self._lock:
            self.session_context_by_id.pop(session_id, None)
            self.failed_provider_keys_by_session.pop(session_id, None)

    def provider_evidence(self, batch_id: UUID) -> tuple[ProviderEvidenceSnapshot, ...]:
        with self._lock:
            provider_counters = self.provider_counters_by_batch.get(batch_id, {})
            snapshots: list[ProviderEvidenceSnapshot] = []
            for provider, model in sorted(provider_counters):
                snapshot = self._provider_evidence_snapshot(batch_id=batch_id, key=(provider, model))
                if snapshot is None:
                    continue
                snapshots.append(snapshot)
            return tuple(snapshots)

    def summary(self, batch_id: UUID) -> RunProgressSummary:
        with self._lock:
            total = int(self.expected_by_batch.get(batch_id, 0))
            completed = len(self.submission_refs_by_batch.get(batch_id, {}))
            remaining = max(0, total - completed)
            return {
                "batch_id": batch_id,
                "total": total,
                "completed": completed,
                "remaining": remaining,
                "latest_sequence": self._latest_sequence(batch_id),
                "provider_evidence": self._provider_evidence_unlocked(batch_id),
            }

    def completed_run_page(
        self,
        batch_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> RunProgressPage:
        if after_sequence < 0:
            raise RuntimeError("after_sequence must be non-negative")
        if limit < 1:
            raise RuntimeError("limit must be positive")

        with self._lock:
            refs = self.submission_refs_by_batch.get(batch_id, {})
            pair_by_sequence = self.pair_by_sequence_by_batch.get(batch_id, {})
            latest_sequence = self._latest_sequence(batch_id)
            requested_sequences = tuple(range(after_sequence + 1, min(latest_sequence, after_sequence + limit) + 1))
            requested_refs: list[RunSubmissionBlobRef] = []
            for sequence in requested_sequences:
                pair = pair_by_sequence.get(sequence)
                if pair is None:
                    raise RuntimeError("progress sequence points at missing pair")
                ref = refs.get(pair)
                if ref is None:
                    raise RuntimeError("progress sequence points at missing result")
                requested_refs.append(ref)
            submissions = self.blob_store.read_many(requested_refs)

        items: list[SequencedRun] = [
            {"sequence": sequence, "submission": submission}
            for sequence, submission in zip(requested_sequences, submissions, strict=True)
        ]
        next_after_sequence = items[-1]["sequence"] if items else after_sequence
        return {
            "batch_id": batch_id,
            "after_sequence": after_sequence,
            "limit": limit,
            "latest_sequence": latest_sequence,
            "next_after_sequence": next_after_sequence,
            "has_more": next_after_sequence < latest_sequence,
            "items": tuple(items),
        }

    def discard_batch(self, batch_id: UUID) -> bool:
        with self._lock:
            removed = self.batches_by_id.pop(batch_id, None) is not None
            removed = self.expected_by_batch.pop(batch_id, None) is not None or removed
            removed = self.submission_refs_by_batch.pop(batch_id, None) is not None or removed
            removed = self.sequence_by_pair_by_batch.pop(batch_id, None) is not None or removed
            removed = self.pair_by_sequence_by_batch.pop(batch_id, None) is not None or removed
            removed = self.next_sequence_by_batch.pop(batch_id, None) is not None or removed
            removed = self.provider_counters_by_batch.pop(batch_id, None) is not None or removed

            session_ids = tuple(
                session_id
                for session_id, context in self.session_context_by_id.items()
                if context.batch_id == batch_id
            )
            for session_id in session_ids:
                removed = self.session_context_by_id.pop(session_id, None) is not None or removed
                removed = self.failed_provider_keys_by_session.pop(session_id, None) is not None or removed

            return self.blob_store.delete_batch(batch_id) or removed

    def prune_stale_batch_dirs_older_than(self, cutoff: datetime) -> tuple[UUID, ...]:
        with self._lock:
            return self.blob_store.prune_stale_batch_dirs(
                cutoff=cutoff,
                protected_batch_ids=frozenset(self.batches_by_id),
            )

    def _latest_sequence(self, batch_id: UUID) -> int:
        return max(0, int(self.next_sequence_by_batch.get(batch_id, 1)) - 1)

    def _provider_evidence_snapshot(
        self,
        *,
        batch_id: UUID,
        key: tuple[str, str],
    ) -> ProviderEvidenceSnapshot | None:
        provider_counters = self.provider_counters_by_batch.get(batch_id, {})
        counter = provider_counters.get(key)
        if counter is None:
            return None
        provider, model = key
        snapshot: ProviderEvidenceSnapshot = {
            "provider": provider,
            "model": model,
            "total_calls": counter.total_calls,
            "failed_calls": counter.failed_calls,
        }
        if counter.failure_reason is not None:
            snapshot["failure_reason"] = counter.failure_reason
        return snapshot

    def _record_submission(
        self,
        *,
        batch_id: UUID,
        refs: dict[tuple[UUID, UUID], RunSubmissionBlobRef],
        result: MinerTaskRunSubmission,
        sequence_by_pair: dict[tuple[UUID, UUID], int] | None = None,
        pair_by_sequence: dict[int, tuple[UUID, UUID]] | None = None,
        next_sequence: int | None = None,
        commit_next_sequence: bool = True,
    ) -> int:
        if sequence_by_pair is None:
            sequence_by_pair = self.sequence_by_pair_by_batch.setdefault(batch_id, {})
        if pair_by_sequence is None:
            pair_by_sequence = self.pair_by_sequence_by_batch.setdefault(batch_id, {})
        if next_sequence is None:
            next_sequence = int(self.next_sequence_by_batch.get(batch_id, 1))

        pair = _submission_pair(result)
        existing_ref = refs.get(pair)
        if existing_ref is not None:
            existing = self.blob_store.read(existing_ref)
            if existing != result:
                raise RuntimeError(
                    "batch already recorded a different result for artifact/task pair"
                )
            if pair not in sequence_by_pair:
                sequence_by_pair[pair] = next_sequence
                pair_by_sequence[next_sequence] = pair
                next_sequence += 1
            if commit_next_sequence:
                self.next_sequence_by_batch[batch_id] = max(
                    int(self.next_sequence_by_batch.get(batch_id, 1)),
                    next_sequence,
                )
            return next_sequence

        assigned_sequence = next_sequence
        ref = self.blob_store.append(
            batch_id=batch_id,
            sequence=assigned_sequence,
            submission=result,
        )
        refs[pair] = ref
        sequence_by_pair[pair] = assigned_sequence
        pair_by_sequence[assigned_sequence] = pair
        next_sequence = assigned_sequence + 1
        if commit_next_sequence:
            self.next_sequence_by_batch[batch_id] = next_sequence
        return next_sequence

    def _provider_evidence_unlocked(self, batch_id: UUID) -> tuple[ProviderEvidenceSnapshot, ...]:
        provider_counters = self.provider_counters_by_batch.get(batch_id, {})
        snapshots: list[ProviderEvidenceSnapshot] = []
        for provider, model in sorted(provider_counters):
            snapshot = self._provider_evidence_snapshot(batch_id=batch_id, key=(provider, model))
            if snapshot is None:
                continue
            snapshots.append(snapshot)
        return tuple(snapshots)


def _submission_pair(result: MinerTaskRunSubmission) -> tuple[UUID, UUID]:
    return (result.run.artifact_id, result.run.task_id)


def _provider_model_key(*, provider: str, model: str) -> tuple[str, str]:
    normalized_provider = provider.strip()
    normalized_model = model.strip()
    if not normalized_provider:
        raise RuntimeError("provider key must not be empty")
    if not normalized_model:
        raise RuntimeError("model key must not be empty")
    return normalized_provider, normalized_model


__all__ = [
    "FileBackedRunProgress",
    "ProviderEvidenceSnapshot",
    "RunProgressPage",
    "RunProgressSummary",
    "SequencedRun",
]
