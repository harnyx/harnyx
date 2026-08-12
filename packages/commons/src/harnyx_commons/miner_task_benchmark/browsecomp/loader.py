from __future__ import annotations

import base64
import binascii
import csv
import io
import json
from dataclasses import dataclass
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

BROWSECOMP_SUITE_SLUG = "browsecomp"
BROWSECOMP_SUITE_NAME = "BrowseComp"
_CURRENT_VERSION_FILE = "current_version.json"
_DATA_PACKAGE = "harnyx_commons.miner_task_benchmark.browsecomp.data"
_VERSIONS_DIR = "versions"
_EXPECTED_HEADER = ("problem", "answer", "problem_topic", "canary")
_EXPECTED_MANIFEST = BenchmarkDatasetManifest(
    suite_slug=BROWSECOMP_SUITE_SLUG,
    suite_name=BROWSECOMP_SUITE_NAME,
    dataset_version="2026-08-11-openai-browsecomp",
    scoring_version="correctness-v1",
    source_url="https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv",
    source_page_url="https://github.com/openai/simple-evals",
    license="MIT",
    sha256="7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf",
    row_count=1266,
    file_name="browse_comp_test_set.csv",
    fetched_at="2026-08-11",
)
_EXPECTED_CURRENT_VERSION = (
    _EXPECTED_MANIFEST.dataset_version,
    _EXPECTED_MANIFEST.scoring_version,
)


@dataclass(frozen=True, slots=True)
class _BrowseCompSourceRow:
    problem: str
    answer: str
    problem_topic: str
    canary: str


def load_browsecomp_snapshot(
    *,
    dataset_version: str | None = None,
    scoring_version: str | None = None,
) -> BenchmarkDatasetSnapshot:
    expected_version = _expected_version(
        dataset_version=dataset_version,
        scoring_version=scoring_version,
    )
    if expected_version is None:
        expected_version = _current_browsecomp_version()
    for snapshot in list_browsecomp_snapshots():
        snapshot_version = (snapshot.manifest.dataset_version, snapshot.manifest.scoring_version)
        if snapshot_version == expected_version:
            return snapshot
    raise RuntimeError(
        "unknown BrowseComp snapshot version: "
        f"dataset_version={expected_version[0]!r} scoring_version={expected_version[1]!r}"
    )


@lru_cache(maxsize=1)
def list_browsecomp_snapshots() -> tuple[BenchmarkDatasetSnapshot, ...]:
    data_dir = files(_DATA_PACKAGE)
    versions_dir = data_dir.joinpath(_VERSIONS_DIR)
    snapshots = tuple(
        _load_snapshot_from_dir(entry)
        for entry in sorted(versions_dir.iterdir(), key=lambda path: path.name)
        if entry.is_dir()
    )
    if not snapshots:
        raise RuntimeError("BrowseComp snapshot catalog is empty")
    _current_browsecomp_version()
    return snapshots


def _load_snapshot_from_dir(snapshot_dir: Traversable) -> BenchmarkDatasetSnapshot:
    manifest_path = snapshot_dir.joinpath("manifest.json")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = BenchmarkDatasetManifest(**manifest_payload)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("BrowseComp manifest is invalid") from exc
    if manifest != _EXPECTED_MANIFEST:
        raise RuntimeError(
            f"BrowseComp manifest mismatch: expected {_EXPECTED_MANIFEST!r} got {manifest!r}"
        )

    csv_path = snapshot_dir.joinpath(manifest.file_name)
    raw_source = csv_path.read_bytes()
    checksum = sha256(raw_source).hexdigest()
    if checksum != manifest.sha256:
        raise RuntimeError(f"BrowseComp checksum mismatch: expected {manifest.sha256} got {checksum}")

    rows = _parse_rows(raw_source)
    if len(rows) != manifest.row_count:
        raise RuntimeError(f"BrowseComp row count mismatch: expected {manifest.row_count} got {len(rows)}")
    items = tuple(
        BenchmarkDatasetItem(
            item_index=item_index,
            problem=_decrypt_required(row.problem, row.canary, field="problem"),
            problem_category=row.problem_topic,
            answer=_decrypt_required(row.answer, row.canary, field="answer"),
            answer_type=BenchmarkAnswerType.SINGLE_ANSWER,
        )
        for item_index, row in enumerate(rows)
    )
    return BenchmarkDatasetSnapshot(manifest=manifest, items=items)


def _parse_rows(raw_source: bytes) -> tuple[_BrowseCompSourceRow, ...]:
    try:
        source_text = raw_source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("BrowseComp source is not valid UTF-8") from exc

    reader = csv.DictReader(io.StringIO(source_text, newline=""))
    if tuple(reader.fieldnames or ()) != _EXPECTED_HEADER:
        raise RuntimeError(
            f"BrowseComp header mismatch: expected {_EXPECTED_HEADER!r} got {reader.fieldnames!r}"
        )

    rows: list[_BrowseCompSourceRow] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise RuntimeError(f"BrowseComp row {row_number} has an extra cell")
        missing_fields = tuple(name for name in _EXPECTED_HEADER if row.get(name) is None)
        if missing_fields:
            raise RuntimeError(
                f"BrowseComp row {row_number} has a missing cell for {missing_fields[0]}"
            )
        empty_fields = tuple(name for name in _EXPECTED_HEADER if not row[name])
        if empty_fields:
            raise RuntimeError(f"BrowseComp row {row_number} has empty {empty_fields[0]}")
        rows.append(
            _BrowseCompSourceRow(
                problem=row["problem"],
                answer=row["answer"],
                problem_topic=row["problem_topic"],
                canary=row["canary"],
            )
        )
    return tuple(rows)


def _decrypt_required(encoded: str, canary: str, *, field: str) -> str:
    try:
        ciphertext = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"BrowseComp {field} has invalid base64") from exc
    if not ciphertext:
        raise RuntimeError(f"BrowseComp {field} has empty ciphertext")

    key = sha256(canary.encode("utf-8")).digest()
    decrypted = bytes(value ^ key[index % len(key)] for index, value in enumerate(ciphertext))
    try:
        plaintext = decrypted.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"BrowseComp {field} does not decrypt to valid UTF-8") from exc
    if not plaintext:
        raise RuntimeError(f"BrowseComp {field} decrypts to empty plaintext")
    return plaintext


@lru_cache(maxsize=1)
def _current_browsecomp_version() -> tuple[str, str]:
    data_dir = files(_DATA_PACKAGE)
    try:
        payload = json.loads(
            data_dir.joinpath(_CURRENT_VERSION_FILE).read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("BrowseComp current version file is invalid") from exc
    if payload != {
        "dataset_version": _EXPECTED_CURRENT_VERSION[0],
        "scoring_version": _EXPECTED_CURRENT_VERSION[1],
    }:
        raise RuntimeError(
            "BrowseComp current version mismatch: "
            f"expected {_EXPECTED_CURRENT_VERSION!r} got {payload!r}"
        )
    return _EXPECTED_CURRENT_VERSION


def _expected_version(
    *,
    dataset_version: str | None,
    scoring_version: str | None,
) -> tuple[str, str] | None:
    if dataset_version is None and scoring_version is None:
        return None
    if dataset_version is None or scoring_version is None:
        raise RuntimeError("BrowseComp snapshot lookup requires both dataset_version and scoring_version")
    return dataset_version, scoring_version


__all__ = [
    "BROWSECOMP_SUITE_NAME",
    "BROWSECOMP_SUITE_SLUG",
    "list_browsecomp_snapshots",
    "load_browsecomp_snapshot",
]
