from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from importlib.abc import Traversable
from importlib.resources import files
from typing import Any, cast

from harnyx_commons.miner_task_benchmark.types import (
    BenchmarkAnswerType,
    BenchmarkDatasetItem,
    BenchmarkDatasetManifest,
    BenchmarkDatasetSnapshot,
)

WEBWALKERQA_SUITE_SLUG = "webwalkerqa"
WEBWALKERQA_SUITE_NAME = "WebWalkerQA Easy"
_CURRENT_VERSION_FILE = "current_version.json"
_DATA_PACKAGE = "harnyx_commons.miner_task_benchmark.webwalkerqa.data"
_VERSIONS_DIR = "versions"
_PROBLEM_CATEGORY = "single_source_easy"


def load_webwalkerqa_snapshot(
    *,
    dataset_version: str | None = None,
    scoring_version: str | None = None,
) -> BenchmarkDatasetSnapshot:
    expected_version = _expected_version(
        dataset_version=dataset_version,
        scoring_version=scoring_version,
    )
    snapshots = list_webwalkerqa_snapshots()
    if expected_version is None:
        expected_version = _current_webwalkerqa_version()
    for snapshot in snapshots:
        snapshot_version = (snapshot.manifest.dataset_version, snapshot.manifest.scoring_version)
        if snapshot_version == expected_version:
            return snapshot
    raise RuntimeError(
        "unknown WebWalkerQA snapshot version: "
        f"dataset_version={expected_version[0]!r} scoring_version={expected_version[1]!r}"
    )


@lru_cache(maxsize=1)
def list_webwalkerqa_snapshots() -> tuple[BenchmarkDatasetSnapshot, ...]:
    data_dir = files(_DATA_PACKAGE)
    versions_dir = data_dir.joinpath(_VERSIONS_DIR)
    snapshots = tuple(
        _load_snapshot_from_dir(entry)
        for entry in sorted(versions_dir.iterdir(), key=lambda path: path.name)
        if entry.is_dir()
    )
    if not snapshots:
        raise RuntimeError("WebWalkerQA snapshot catalog is empty")
    _current_webwalkerqa_version()
    return snapshots


def _load_snapshot_from_dir(snapshot_dir: Traversable) -> BenchmarkDatasetSnapshot:
    manifest_payload = json.loads(snapshot_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    manifest = BenchmarkDatasetManifest(**manifest_payload)
    if manifest.suite_slug != WEBWALKERQA_SUITE_SLUG:
        raise RuntimeError(
            f"WebWalkerQA suite slug mismatch: expected {WEBWALKERQA_SUITE_SLUG} got {manifest.suite_slug}"
        )
    if manifest.suite_name != WEBWALKERQA_SUITE_NAME:
        raise RuntimeError(
            f"WebWalkerQA suite name mismatch: expected {WEBWALKERQA_SUITE_NAME} got {manifest.suite_name}"
        )
    json_path = snapshot_dir.joinpath(manifest.file_name)
    raw_bytes = json_path.read_bytes()
    checksum = sha256(raw_bytes).hexdigest()
    if checksum != manifest.sha256:
        raise RuntimeError(
            f"WebWalkerQA checksum mismatch: expected {manifest.sha256} got {checksum}"
        )
    raw_rows = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(raw_rows, list):
        raise RuntimeError("WebWalkerQA raw payload must be a JSON array")
    rows = tuple(
        _item_from_row(source_index=index, row=_require_mapping(row, f"row {index}"))
        for index, row in enumerate(raw_rows)
        if _is_easy_single_source_row(row)
    )
    if len(rows) != manifest.row_count:
        raise RuntimeError(
            f"WebWalkerQA row count mismatch: expected {manifest.row_count} got {len(rows)}"
        )
    return BenchmarkDatasetSnapshot(manifest=manifest, items=rows)


def _item_from_row(*, source_index: int, row: dict[str, Any]) -> BenchmarkDatasetItem:
    question = _require_string(row.get("Question"), f"row {source_index} Question")
    answer = _require_string(row.get("answer"), f"row {source_index} answer")
    root_url = _require_string(row.get("root_url"), f"row {source_index} root_url")
    source_website = row.get("source_website")
    if not isinstance(source_website, list) or not source_website:
        raise RuntimeError(f"row {source_index} source_website must be a non-empty array")
    return BenchmarkDatasetItem(
        item_index=source_index,
        problem=f"Root URL: {root_url}\nQuestion: {question}",
        problem_category=_PROBLEM_CATEGORY,
        answer=answer,
        answer_type=BenchmarkAnswerType.SINGLE_ANSWER,
    )


def _is_easy_single_source_row(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    typed_row = cast(dict[str, Any], row)
    return typed_row.get("type") == "single_source" and typed_row.get("difficulty_level") == "easy"


@lru_cache(maxsize=1)
def _current_webwalkerqa_version() -> tuple[str, str]:
    data_dir = files(_DATA_PACKAGE)
    payload = json.loads(data_dir.joinpath(_CURRENT_VERSION_FILE).read_text(encoding="utf-8"))
    version = _expected_version(
        dataset_version=payload["dataset_version"],
        scoring_version=payload["scoring_version"],
    )
    if version is None:
        raise RuntimeError("WebWalkerQA current version file must define dataset_version and scoring_version")
    return version


def _expected_version(
    *,
    dataset_version: str | None,
    scoring_version: str | None,
) -> tuple[str, str] | None:
    if dataset_version is None and scoring_version is None:
        return None
    if dataset_version is None or scoring_version is None:
        raise RuntimeError("WebWalkerQA snapshot lookup requires both dataset_version and scoring_version")
    return dataset_version, scoring_version


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"WebWalkerQA {label} must be an object")
    return cast(dict[str, Any], value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"WebWalkerQA {label} must be a non-empty string")
    return value


__all__ = [
    "WEBWALKERQA_SUITE_NAME",
    "WEBWALKERQA_SUITE_SLUG",
    "list_webwalkerqa_snapshots",
    "load_webwalkerqa_snapshot",
]
