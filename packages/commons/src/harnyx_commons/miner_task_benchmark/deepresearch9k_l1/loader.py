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

DEEPRESEARCH9K_L1_SUITE_SLUG = "deepresearch9k-l1"
DEEPRESEARCH9K_L1_SUITE_NAME = "DeepResearch-9K L1"
_CURRENT_VERSION_FILE = "current_version.json"
_DATA_PACKAGE = "harnyx_commons.miner_task_benchmark.deepresearch9k_l1.data"
_VERSIONS_DIR = "versions"


def load_deepresearch9k_l1_snapshot(
    *,
    dataset_version: str | None = None,
    scoring_version: str | None = None,
) -> BenchmarkDatasetSnapshot:
    expected_version = _expected_version(
        dataset_version=dataset_version,
        scoring_version=scoring_version,
    )
    snapshots = list_deepresearch9k_l1_snapshots()
    if expected_version is None:
        expected_version = _current_deepresearch9k_l1_version()
    for snapshot in snapshots:
        snapshot_version = (snapshot.manifest.dataset_version, snapshot.manifest.scoring_version)
        if snapshot_version == expected_version:
            return snapshot
    raise RuntimeError(
        "unknown DeepResearch-9K L1 snapshot version: "
        f"dataset_version={expected_version[0]!r} scoring_version={expected_version[1]!r}"
    )


@lru_cache(maxsize=1)
def list_deepresearch9k_l1_snapshots() -> tuple[BenchmarkDatasetSnapshot, ...]:
    data_dir = files(_DATA_PACKAGE)
    versions_dir = data_dir.joinpath(_VERSIONS_DIR)
    snapshots = tuple(
        _load_snapshot_from_dir(entry)
        for entry in sorted(versions_dir.iterdir(), key=lambda path: path.name)
        if entry.is_dir()
    )
    if not snapshots:
        raise RuntimeError("DeepResearch-9K L1 snapshot catalog is empty")
    _current_deepresearch9k_l1_version()
    return snapshots


def _load_snapshot_from_dir(snapshot_dir: Traversable) -> BenchmarkDatasetSnapshot:
    manifest_payload = json.loads(snapshot_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    manifest = BenchmarkDatasetManifest(**manifest_payload)
    if manifest.suite_slug != DEEPRESEARCH9K_L1_SUITE_SLUG:
        raise RuntimeError(
            "DeepResearch-9K L1 suite slug mismatch: "
            f"expected {DEEPRESEARCH9K_L1_SUITE_SLUG} got {manifest.suite_slug}"
        )
    if manifest.suite_name != DEEPRESEARCH9K_L1_SUITE_NAME:
        raise RuntimeError(
            "DeepResearch-9K L1 suite name mismatch: "
            f"expected {DEEPRESEARCH9K_L1_SUITE_NAME} got {manifest.suite_name}"
        )
    csv_path = snapshot_dir.joinpath(manifest.file_name)
    checksum = sha256(csv_path.read_bytes()).hexdigest()
    if checksum != manifest.sha256:
        raise RuntimeError(
            f"DeepResearch-9K L1 checksum mismatch: expected {manifest.sha256} got {checksum}"
        )
    with io.StringIO(csv_path.read_text(encoding="utf-8")) as handle:
        rows = tuple(
            BenchmarkDatasetItem(
                item_index=int(row["item_index"]),
                problem=row["problem"],
                problem_category=row["problem_category"],
                answer=row["answer"],
                answer_type=BenchmarkAnswerType(row["answer_type"]),
            )
            for row in csv.DictReader(handle)
        )
    if len(rows) != manifest.row_count:
        raise RuntimeError(
            f"DeepResearch-9K L1 row count mismatch: expected {manifest.row_count} got {len(rows)}"
        )
    return BenchmarkDatasetSnapshot(manifest=manifest, items=rows)


@lru_cache(maxsize=1)
def _current_deepresearch9k_l1_version() -> tuple[str, str]:
    data_dir = files(_DATA_PACKAGE)
    payload = json.loads(data_dir.joinpath(_CURRENT_VERSION_FILE).read_text(encoding="utf-8"))
    version = _expected_version(
        dataset_version=payload["dataset_version"],
        scoring_version=payload["scoring_version"],
    )
    if version is None:
        raise RuntimeError(
            "DeepResearch-9K L1 current version file must define dataset_version and scoring_version"
        )
    return version


def _expected_version(
    *,
    dataset_version: str | None,
    scoring_version: str | None,
) -> tuple[str, str] | None:
    if dataset_version is None and scoring_version is None:
        return None
    if dataset_version is None or scoring_version is None:
        raise RuntimeError("DeepResearch-9K L1 snapshot lookup requires both dataset_version and scoring_version")
    return dataset_version, scoring_version


__all__ = [
    "DEEPRESEARCH9K_L1_SUITE_NAME",
    "DEEPRESEARCH9K_L1_SUITE_SLUG",
    "list_deepresearch9k_l1_snapshots",
    "load_deepresearch9k_l1_snapshot",
]
