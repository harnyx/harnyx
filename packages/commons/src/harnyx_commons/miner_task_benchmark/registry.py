from __future__ import annotations

from collections.abc import Callable

from harnyx_commons.miner_task_benchmark.deepresearch9k_l1.loader import (
    DEEPRESEARCH9K_L1_SUITE_SLUG,
    list_deepresearch9k_l1_snapshots,
    load_deepresearch9k_l1_snapshot,
)
from harnyx_commons.miner_task_benchmark.deepsearchqa.loader import (
    DEEPSEARCHQA_SUITE_SLUG,
    list_deepsearchqa_snapshots,
    load_deepsearchqa_snapshot,
)
from harnyx_commons.miner_task_benchmark.types import BenchmarkDatasetSnapshot

BenchmarkSnapshotCatalogLoader = Callable[[], tuple[BenchmarkDatasetSnapshot, ...]]
BenchmarkActiveSnapshotLoader = Callable[[], BenchmarkDatasetSnapshot]
BenchmarkSnapshotVersionKey = tuple[str, str]

_BENCHMARK_SNAPSHOT_CATALOG_LOADERS: dict[str, BenchmarkSnapshotCatalogLoader] = {
    DEEPSEARCHQA_SUITE_SLUG: list_deepsearchqa_snapshots,
    DEEPRESEARCH9K_L1_SUITE_SLUG: list_deepresearch9k_l1_snapshots,
}
_BENCHMARK_ACTIVE_SNAPSHOT_LOADERS: dict[str, BenchmarkActiveSnapshotLoader] = {
    DEEPSEARCHQA_SUITE_SLUG: load_deepsearchqa_snapshot,
    DEEPRESEARCH9K_L1_SUITE_SLUG: load_deepresearch9k_l1_snapshot,
}


def load_benchmark_snapshot(
    suite_slug: str,
    *,
    dataset_version: str | None = None,
    scoring_version: str | None = None,
) -> BenchmarkDatasetSnapshot:
    expected_version = _expected_snapshot_version(
        dataset_version=dataset_version,
        scoring_version=scoring_version,
    )
    if expected_version is None:
        active_loader = _BENCHMARK_ACTIVE_SNAPSHOT_LOADERS.get(suite_slug)
        if active_loader is None:
            raise RuntimeError(f"unknown benchmark suite_slug: {suite_slug}")
        return active_loader()

    catalog_loader = _BENCHMARK_SNAPSHOT_CATALOG_LOADERS.get(suite_slug)
    if catalog_loader is None:
        raise RuntimeError(f"unknown benchmark suite_slug: {suite_slug}")
    snapshots = catalog_loader()
    for snapshot in snapshots:
        if snapshot.manifest.suite_slug != suite_slug:
            raise RuntimeError(
                "benchmark suite registry mismatch: "
                f"requested {suite_slug} got {snapshot.manifest.suite_slug}"
            )
        snapshot_version = (snapshot.manifest.dataset_version, snapshot.manifest.scoring_version)
        if snapshot_version == expected_version:
            return snapshot
    raise RuntimeError(
        "unknown benchmark snapshot version: "
        f"{suite_slug} dataset_version={dataset_version!r} scoring_version={scoring_version!r}"
    )


def load_active_benchmark_snapshot() -> BenchmarkDatasetSnapshot:
    if len(_BENCHMARK_ACTIVE_SNAPSHOT_LOADERS) != 1:
        raise RuntimeError("active benchmark suite is ambiguous; resolve an explicit suite_slug")
    _, loader = next(iter(_BENCHMARK_ACTIVE_SNAPSHOT_LOADERS.items()))
    return loader()


def list_current_benchmark_snapshots() -> tuple[BenchmarkDatasetSnapshot, ...]:
    return tuple(
        loader()
        for _, loader in sorted(_BENCHMARK_ACTIVE_SNAPSHOT_LOADERS.items(), key=lambda entry: entry[0])
    )


def load_current_benchmark_snapshot(suite_slug: str) -> BenchmarkDatasetSnapshot:
    active_loader = _BENCHMARK_ACTIVE_SNAPSHOT_LOADERS.get(suite_slug)
    if active_loader is None:
        raise RuntimeError(f"unknown current benchmark suite_slug: {suite_slug}")
    return active_loader()


def _expected_snapshot_version(
    *,
    dataset_version: str | None,
    scoring_version: str | None,
) -> BenchmarkSnapshotVersionKey | None:
    if dataset_version is None and scoring_version is None:
        return None
    if dataset_version is None or scoring_version is None:
        raise RuntimeError("benchmark snapshot lookup requires both dataset_version and scoring_version")
    return dataset_version, scoring_version


__all__ = [
    "list_current_benchmark_snapshots",
    "load_active_benchmark_snapshot",
    "load_benchmark_snapshot",
    "load_current_benchmark_snapshot",
]
