"""Checksum-bound, answer-free DeepSearchQA forms for generation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from uuid import UUID

from harnyx_commons.domain_tweak_generation.contracts import GenerationForm
from harnyx_commons.miner_task_benchmark.deepsearchqa.loader import load_deepsearchqa_snapshot

_DATASET_VERSION = "2026-04-02-google-main"
_SCORING_VERSION = "correctness-v1"


@dataclass(slots=True)
class GenerationFormCursor:
    _forms: tuple[GenerationForm, ...]
    _batch_id: UUID
    _cycle: int = 0
    _offset: int = 0
    _permutation: tuple[GenerationForm, ...] = field(init=False)

    def __post_init__(self) -> None:
        self._permutation = self._shuffle(self._cycle)

    def next_inputs(
        self,
        count: int,
        *,
        excluding: frozenset[str] = frozenset(),
    ) -> tuple[GenerationForm, ...]:
        if count <= 0:
            raise ValueError("form count must be positive")
        eligible_count = len(self._forms) - sum(item.form_identity in excluding for item in self._forms)
        if count > eligible_count:
            raise ValueError("requested form count exceeds the eligible DeepSearchQA form pool")

        selected: list[GenerationForm] = []
        selected_identities: set[str] = set()
        while len(selected) < count:
            if self._offset == len(self._permutation):
                self._cycle += 1
                self._offset = 0
                self._permutation = self._shuffle(self._cycle)
            form = self._permutation[self._offset]
            self._offset += 1
            if form.form_identity in excluding or form.form_identity in selected_identities:
                continue
            selected.append(form)
            selected_identities.add(form.form_identity)
        return tuple(selected)

    def _shuffle(self, cycle: int) -> tuple[GenerationForm, ...]:
        seed_bytes = hashlib.sha256(f"{self._batch_id}:{cycle}".encode()).digest()
        shuffled = list(self._forms)
        random.Random(int.from_bytes(seed_bytes)).shuffle(shuffled)  # noqa: S311 - deterministic selection seed
        return tuple(shuffled)


class DeepSearchQAFormSource:
    """Loads every problem as a generation form without exposing answer fields."""

    def __init__(self) -> None:
        self._forms = _load_forms()

    @property
    def forms(self) -> tuple[GenerationForm, ...]:
        return self._forms

    def cursor(self, batch_id: UUID, *, target_count: int) -> GenerationFormCursor:
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        if target_count > len(self._forms):
            raise ValueError("target_count exceeds the eligible DeepSearchQA form pool")
        return GenerationFormCursor(self._forms, batch_id)


def _load_forms() -> tuple[GenerationForm, ...]:
    snapshot = load_deepsearchqa_snapshot(
        dataset_version=_DATASET_VERSION,
        scoring_version=_SCORING_VERSION,
    )
    return tuple(
        GenerationForm(
            form_identity=f"deepsearchqa:{snapshot.manifest.dataset_version}:{item.item_index}",
            source_index=item.item_index,
            form=item.problem,
        )
        for item in snapshot.items
    )


__all__ = ["DeepSearchQAFormSource", "GenerationFormCursor"]
