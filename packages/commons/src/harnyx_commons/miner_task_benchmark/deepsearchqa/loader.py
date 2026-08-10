from __future__ import annotations

import csv
import io
import json
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

DEEPSEARCHQA_SUITE_SLUG = "deepsearchqa"
DEEPSEARCHQA_SUITE_NAME = "DeepSearchQA"
_CURRENT_VERSION_FILE = "current_version.json"
_DATA_PACKAGE = "harnyx_commons.miner_task_benchmark.deepsearchqa.data"
_VERSIONS_DIR = "versions"


def load_deepsearchqa_snapshot(
    *,
    dataset_version: str | None = None,
    scoring_version: str | None = None,
) -> BenchmarkDatasetSnapshot:
    expected_version = _expected_version(
        dataset_version=dataset_version,
        scoring_version=scoring_version,
    )
    snapshots = list_deepsearchqa_snapshots()
    if expected_version is None:
        expected_version = _current_deepsearchqa_version()
    for snapshot in snapshots:
        snapshot_version = (snapshot.manifest.dataset_version, snapshot.manifest.scoring_version)
        if snapshot_version == expected_version:
            return snapshot
    raise RuntimeError(
        "unknown DeepSearchQA snapshot version: "
        f"dataset_version={expected_version[0]!r} scoring_version={expected_version[1]!r}"
    )


@lru_cache(maxsize=1)
def list_deepsearchqa_snapshots() -> tuple[BenchmarkDatasetSnapshot, ...]:
    data_dir = files(_DATA_PACKAGE)
    versions_dir = data_dir.joinpath(_VERSIONS_DIR)
    snapshots = tuple(
        _load_snapshot_from_dir(entry)
        for entry in sorted(versions_dir.iterdir(), key=lambda path: path.name)
        if entry.is_dir()
    )
    if not snapshots:
        raise RuntimeError("DeepSearchQA snapshot catalog is empty")
    _current_deepsearchqa_version()
    return snapshots


def _load_snapshot_from_dir(snapshot_dir: Traversable) -> BenchmarkDatasetSnapshot:
    manifest_payload = json.loads(snapshot_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    manifest = BenchmarkDatasetManifest(**manifest_payload)
    if manifest.suite_slug != DEEPSEARCHQA_SUITE_SLUG:
        raise RuntimeError(
            f"DeepSearchQA suite slug mismatch: expected {DEEPSEARCHQA_SUITE_SLUG} got {manifest.suite_slug}"
        )
    if manifest.suite_name != DEEPSEARCHQA_SUITE_NAME:
        raise RuntimeError(
            f"DeepSearchQA suite name mismatch: expected {DEEPSEARCHQA_SUITE_NAME} got {manifest.suite_name}"
        )
    csv_path = snapshot_dir.joinpath(manifest.file_name)
    checksum = sha256(csv_path.read_bytes()).hexdigest()
    if checksum != manifest.sha256:
        raise RuntimeError(f"DeepSearchQA checksum mismatch: expected {manifest.sha256} got {checksum}")
    with io.StringIO(csv_path.read_text(encoding="utf-8")) as handle:
        rows = tuple(
            BenchmarkDatasetItem(
                item_index=index,
                problem=row["problem"],
                problem_category=row["problem_category"],
                answer=row["answer"],
                answer_type=BenchmarkAnswerType(row["answer_type"]),
            )
            for index, row in enumerate(csv.DictReader(handle))
        )
    if len(rows) != manifest.row_count:
        raise RuntimeError(f"DeepSearchQA row count mismatch: expected {manifest.row_count} got {len(rows)}")
    return BenchmarkDatasetSnapshot(manifest=manifest, items=rows)


@lru_cache(maxsize=1)
def _current_deepsearchqa_version() -> tuple[str, str]:
    data_dir = files(_DATA_PACKAGE)
    payload = json.loads(data_dir.joinpath(_CURRENT_VERSION_FILE).read_text(encoding="utf-8"))
    version = _expected_version(
        dataset_version=payload["dataset_version"],
        scoring_version=payload["scoring_version"],
    )
    if version is None:
        raise RuntimeError("DeepSearchQA current version file must define dataset_version and scoring_version")
    return version


def _expected_version(
    *,
    dataset_version: str | None,
    scoring_version: str | None,
) -> tuple[str, str] | None:
    if dataset_version is None and scoring_version is None:
        return None
    if dataset_version is None or scoring_version is None:
        raise RuntimeError("DeepSearchQA snapshot lookup requires both dataset_version and scoring_version")
    return dataset_version, scoring_version


__all__ = [
    "DEEPSEARCHQA_SUITE_NAME",
    "DEEPSEARCHQA_SUITE_SLUG",
    "list_deepsearchqa_snapshots",
    "load_deepsearchqa_snapshot",
]
