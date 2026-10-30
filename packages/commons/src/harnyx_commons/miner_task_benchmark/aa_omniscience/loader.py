"""Pinned public v1 source, separate from the automatic benchmark catalog."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import fields
from functools import lru_cache
from hashlib import sha256
from importlib.abc import Traversable
from importlib.resources import files

from harnyx_commons.miner_task_benchmark.types import (
    BenchmarkAnswerType,
    BenchmarkDatasetItem,
    BenchmarkDatasetManifest,
    BenchmarkDatasetSnapshot,
)

AA_OMNISCIENCE_SCORING_VERSION = "aa-omniscience-accuracy-v1"
_VERSION = "2026-10-29-hf-e4883edb-v1"
_SOURCE_SHA256 = "1e04603dafa3bd0d16d8151f07a5eb74c43d90c3a85d2fca120da2359174d02f"
_PROMPT_SHA256 = "d598fe204eb74f568c61a965bc277731379ace0582523db30acc1b89dc8a2e03"
_COLUMNS = ("domain", "topic", "subtopic", "question_id", "question", "answer")


def _snapshot_dir() -> Traversable:
    return files("harnyx_commons.miner_task_benchmark.aa_omniscience").joinpath("data", "versions", _VERSION)


@lru_cache(maxsize=1)
def load_aa_omniscience_snapshot() -> BenchmarkDatasetSnapshot:
    directory = _snapshot_dir()
    provenance = json.loads(directory.joinpath("manifest.json").read_text(encoding="utf-8"))
    manifest = BenchmarkDatasetManifest(
        **{field.name: provenance[field.name] for field in fields(BenchmarkDatasetManifest)
           if field.name != "scoring_version"},
        scoring_version=AA_OMNISCIENCE_SCORING_VERSION,
    )
    raw = directory.joinpath(manifest.file_name).read_bytes()
    if manifest.sha256 != _SOURCE_SHA256 or sha256(raw).hexdigest() != _SOURCE_SHA256:
        raise RuntimeError("AA-Omniscience public v1 source checksum mismatch")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != _COLUMNS:
        raise RuntimeError("AA-Omniscience source columns mismatch")
    rows = tuple(reader)
    if len(rows) != 600 or manifest.row_count != 600:
        raise RuntimeError("AA-Omniscience public v1 must contain 600 rows")
    for index, row in enumerate(rows, start=1):
        if None in row or any(not row.get(key) for key in _COLUMNS):
            raise RuntimeError(f"AA-Omniscience invalid source row {index}")
        if row["question_id"] != str(index):
            raise RuntimeError(f"AA-Omniscience question ID mismatch at row {index}")
    if dict(Counter(row["domain"] for row in rows)) != provenance["domain_row_counts"]:
        raise RuntimeError("AA-Omniscience source domain counts mismatch")
    return BenchmarkDatasetSnapshot(
        manifest=manifest,
        items=tuple(
            BenchmarkDatasetItem(
                item_index=index,
                problem=row["question"],
                problem_category=row["domain"],
                answer=row["answer"],
                answer_type=BenchmarkAnswerType.SINGLE_ANSWER,
                source_item_id=row["question_id"],
            )
            for index, row in enumerate(rows)
        ),
    )


@lru_cache(maxsize=1)
def load_aa_omniscience_grader_prompt() -> str:
    raw = _snapshot_dir().joinpath("grader_prompt.txt").read_bytes()
    if sha256(raw).hexdigest() != _PROMPT_SHA256:
        raise RuntimeError("AA-Omniscience original grader prompt checksum mismatch")
    return raw.decode("utf-8").strip()
