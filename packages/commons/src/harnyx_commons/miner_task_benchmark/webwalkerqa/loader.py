from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.abc import Traversable
from importlib.resources import files
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

from harnyx_commons.miner_task_benchmark.types import (
    BenchmarkAnswerType,
    BenchmarkDatasetItem,
    BenchmarkDatasetManifest,
    BenchmarkDatasetSnapshot,
)

WEBWALKERQA_SUITE_SLUG = "webwalkerqa"
WEBWALKERQA_SUITE_NAME = "WebWalkerQA Easy"
WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_SLUG = "webwalkerqa-single-source-medium"
WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_NAME = "WebWalkerQA Single-Source Medium"
WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_SLUG = "webwalkerqa-multi-source-medium"
WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_NAME = "WebWalkerQA Multi-Source Medium"

_CURRENT_VERSION_FILE = "current_version.json"
_DATA_PACKAGE = "harnyx_commons.miner_task_benchmark.webwalkerqa.data"
_SOURCES_DIR = "sources"
_VERSIONS_DIR = "versions"
_NonEmptyString = Annotated[str, StringConstraints(min_length=1)]


@dataclass(frozen=True, slots=True)
class _WebWalkerQASuiteSpec:
    slug: str
    name: str
    data_package: str
    source_type: str
    difficulty: str
    problem_category: str


@dataclass(frozen=True, slots=True)
class _SnapshotVersion:
    dataset_version: str
    scoring_version: str


class _WebWalkerQARowSelector(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    source_type: _NonEmptyString = Field(alias="type")
    difficulty: _NonEmptyString = Field(alias="difficulty_level")


class _WebWalkerQARow(_WebWalkerQARowSelector):
    question: _NonEmptyString = Field(alias="Question")
    answer: _NonEmptyString
    root_url: _NonEmptyString
    source_website: list[_NonEmptyString] = Field(min_length=1)


_WEBWALKERQA_EASY = _WebWalkerQASuiteSpec(
    slug=WEBWALKERQA_SUITE_SLUG,
    name=WEBWALKERQA_SUITE_NAME,
    data_package=_DATA_PACKAGE,
    source_type="single_source",
    difficulty="easy",
    problem_category="single_source_easy",
)
_WEBWALKERQA_SINGLE_SOURCE_MEDIUM = _WebWalkerQASuiteSpec(
    slug=WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_SLUG,
    name=WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_NAME,
    data_package="harnyx_commons.miner_task_benchmark.webwalkerqa.data.single_source_medium",
    source_type="single_source",
    difficulty="medium",
    problem_category="single_source_medium",
)
_WEBWALKERQA_MULTI_SOURCE_MEDIUM = _WebWalkerQASuiteSpec(
    slug=WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_SLUG,
    name=WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_NAME,
    data_package="harnyx_commons.miner_task_benchmark.webwalkerqa.data.multi_source_medium",
    source_type="multi_source",
    difficulty="medium",
    problem_category="multi_source_medium",
)
_MANIFEST_ADAPTER = TypeAdapter(BenchmarkDatasetManifest)
_RAW_ROWS_ADAPTER = TypeAdapter(list[dict[str, object]])
_ROW_SELECTOR_ADAPTER = TypeAdapter(_WebWalkerQARowSelector)
_ROW_ADAPTER = TypeAdapter(_WebWalkerQARow)
_VERSION_ADAPTER = TypeAdapter(_SnapshotVersion)


def load_webwalkerqa_snapshot(
    *,
    dataset_version: str | None = None,
    scoring_version: str | None = None,
) -> BenchmarkDatasetSnapshot:
    return _load_snapshot(
        spec=_WEBWALKERQA_EASY,
        snapshots=list_webwalkerqa_snapshots(),
        dataset_version=dataset_version,
        scoring_version=scoring_version,
    )


def load_webwalkerqa_single_source_medium_snapshot(
    *,
    dataset_version: str | None = None,
    scoring_version: str | None = None,
) -> BenchmarkDatasetSnapshot:
    return _load_snapshot(
        spec=_WEBWALKERQA_SINGLE_SOURCE_MEDIUM,
        snapshots=list_webwalkerqa_single_source_medium_snapshots(),
        dataset_version=dataset_version,
        scoring_version=scoring_version,
    )


def load_webwalkerqa_multi_source_medium_snapshot(
    *,
    dataset_version: str | None = None,
    scoring_version: str | None = None,
) -> BenchmarkDatasetSnapshot:
    return _load_snapshot(
        spec=_WEBWALKERQA_MULTI_SOURCE_MEDIUM,
        snapshots=list_webwalkerqa_multi_source_medium_snapshots(),
        dataset_version=dataset_version,
        scoring_version=scoring_version,
    )


@lru_cache(maxsize=1)
def list_webwalkerqa_snapshots() -> tuple[BenchmarkDatasetSnapshot, ...]:
    return _list_snapshots(_WEBWALKERQA_EASY)


@lru_cache(maxsize=1)
def list_webwalkerqa_single_source_medium_snapshots() -> tuple[BenchmarkDatasetSnapshot, ...]:
    return _list_snapshots(_WEBWALKERQA_SINGLE_SOURCE_MEDIUM)


@lru_cache(maxsize=1)
def list_webwalkerqa_multi_source_medium_snapshots() -> tuple[BenchmarkDatasetSnapshot, ...]:
    return _list_snapshots(_WEBWALKERQA_MULTI_SOURCE_MEDIUM)


def _load_snapshot(
    *,
    spec: _WebWalkerQASuiteSpec,
    snapshots: tuple[BenchmarkDatasetSnapshot, ...],
    dataset_version: str | None,
    scoring_version: str | None,
) -> BenchmarkDatasetSnapshot:
    expected_version = _expected_version(
        dataset_version=dataset_version,
        scoring_version=scoring_version,
        suite_name=spec.name,
    )
    if expected_version is None:
        expected_version = _current_version(spec)
    for snapshot in snapshots:
        snapshot_version = (snapshot.manifest.dataset_version, snapshot.manifest.scoring_version)
        if snapshot_version == expected_version:
            return snapshot
    raise RuntimeError(
        f"unknown {spec.name} snapshot version: "
        f"dataset_version={expected_version[0]!r} scoring_version={expected_version[1]!r}"
    )


def _list_snapshots(spec: _WebWalkerQASuiteSpec) -> tuple[BenchmarkDatasetSnapshot, ...]:
    data_dir = files(spec.data_package)
    sources_dir = files(_DATA_PACKAGE).joinpath(_SOURCES_DIR)
    versions_dir = data_dir.joinpath(_VERSIONS_DIR)
    snapshots = tuple(
        _load_snapshot_from_dir(snapshot_dir=entry, sources_dir=sources_dir, spec=spec)
        for entry in sorted(versions_dir.iterdir(), key=lambda path: path.name)
        if entry.is_dir()
    )
    if not snapshots:
        raise RuntimeError(f"{spec.name} snapshot catalog is empty")
    _current_version(spec)
    return snapshots


def _load_snapshot_from_dir(
    *,
    snapshot_dir: Traversable,
    sources_dir: Traversable,
    spec: _WebWalkerQASuiteSpec,
) -> BenchmarkDatasetSnapshot:
    manifest = _MANIFEST_ADAPTER.validate_json(
        snapshot_dir.joinpath("manifest.json").read_text(encoding="utf-8")
    )
    if manifest.suite_slug != spec.slug:
        raise RuntimeError(
            f"{spec.name} suite slug mismatch: expected {spec.slug} got {manifest.suite_slug}"
        )
    if manifest.suite_name != spec.name:
        raise RuntimeError(
            f"{spec.name} suite name mismatch: expected {spec.name} got {manifest.suite_name}"
        )
    json_path = sources_dir.joinpath(manifest.sha256, manifest.file_name)
    raw_bytes = json_path.read_bytes()
    checksum = sha256(raw_bytes).hexdigest()
    if checksum != manifest.sha256:
        raise RuntimeError(f"{spec.name} checksum mismatch: expected {manifest.sha256} got {checksum}")
    source_rows = _RAW_ROWS_ADAPTER.validate_json(raw_bytes)
    items: list[BenchmarkDatasetItem] = []
    for source_index, source_row in enumerate(source_rows):
        selector = _ROW_SELECTOR_ADAPTER.validate_python(source_row)
        if selector.source_type != spec.source_type or selector.difficulty != spec.difficulty:
            continue
        row = _ROW_ADAPTER.validate_python(source_row)
        items.append(_item_from_row(source_index=source_index, row=row, spec=spec))
    if len(items) != manifest.row_count:
        raise RuntimeError(
            f"{spec.name} row count mismatch: expected {manifest.row_count} got {len(items)}"
        )
    return BenchmarkDatasetSnapshot(manifest=manifest, items=tuple(items))


def _item_from_row(
    *,
    source_index: int,
    row: _WebWalkerQARow,
    spec: _WebWalkerQASuiteSpec,
) -> BenchmarkDatasetItem:
    return BenchmarkDatasetItem(
        item_index=source_index,
        problem=f"Root URL: {row.root_url}\nQuestion: {row.question}",
        problem_category=spec.problem_category,
        answer=row.answer,
        answer_type=BenchmarkAnswerType.SINGLE_ANSWER,
    )


@lru_cache(maxsize=3)
def _current_version(spec: _WebWalkerQASuiteSpec) -> tuple[str, str]:
    data_dir = files(spec.data_package)
    payload = data_dir.joinpath(_CURRENT_VERSION_FILE).read_text(encoding="utf-8")
    version_data = _VERSION_ADAPTER.validate_json(payload)
    version = _expected_version(
        dataset_version=version_data.dataset_version,
        scoring_version=version_data.scoring_version,
        suite_name=spec.name,
    )
    if version is None:
        raise RuntimeError(
            f"{spec.name} current version file must define dataset_version and scoring_version"
        )
    return version


def _expected_version(
    *,
    dataset_version: str | None,
    scoring_version: str | None,
    suite_name: str,
) -> tuple[str, str] | None:
    if dataset_version is None and scoring_version is None:
        return None
    if dataset_version is None or scoring_version is None:
        raise RuntimeError(f"{suite_name} snapshot lookup requires both dataset_version and scoring_version")
    return dataset_version, scoring_version


__all__ = [
    "WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_NAME",
    "WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_SLUG",
    "WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_NAME",
    "WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_SLUG",
    "WEBWALKERQA_SUITE_NAME",
    "WEBWALKERQA_SUITE_SLUG",
    "list_webwalkerqa_multi_source_medium_snapshots",
    "list_webwalkerqa_single_source_medium_snapshots",
    "list_webwalkerqa_snapshots",
    "load_webwalkerqa_multi_source_medium_snapshot",
    "load_webwalkerqa_single_source_medium_snapshot",
    "load_webwalkerqa_snapshot",
]
