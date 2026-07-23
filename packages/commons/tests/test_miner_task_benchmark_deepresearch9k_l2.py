from __future__ import annotations

from hashlib import sha256
from importlib.resources import files
from uuid import UUID

from harnyx_commons.miner_task_benchmark import (
    DEEPRESEARCH9K_L2_SUITE_NAME,
    DEEPRESEARCH9K_L2_SUITE_SLUG,
    BenchmarkDatasetSnapshot,
    benchmark_task_id_for_item,
    list_deepresearch9k_l2_snapshots,
    load_benchmark_snapshot,
    load_deepresearch9k_l2_snapshot,
    sample_benchmark_items,
)

_DATASET_VERSION = "2026-07-22-hf-9eaf02da-l2"
_SCORING_VERSION = "correctness-v1"


def test_load_deepresearch9k_l2_snapshot_preserves_source_item_indices() -> None:
    snapshot = load_deepresearch9k_l2_snapshot()

    assert isinstance(snapshot, BenchmarkDatasetSnapshot)
    assert snapshot.manifest.suite_slug == DEEPRESEARCH9K_L2_SUITE_SLUG
    assert snapshot.manifest.suite_name == DEEPRESEARCH9K_L2_SUITE_NAME
    assert snapshot.manifest.dataset_version == _DATASET_VERSION
    assert snapshot.manifest.scoring_version == _SCORING_VERSION
    assert snapshot.manifest.row_count == 3000
    assert len(snapshot.items) == 3000
    assert [item.item_index for item in snapshot.items] == list(range(1, 9000, 3))
    assert {item.problem_category for item in snapshot.items} == {"difficulty-2"}


def test_deepresearch9k_l2_manifest_checksum_matches_versioned_packaged_csv() -> None:
    snapshot = load_deepresearch9k_l2_snapshot()
    version_dir = files("harnyx_commons.miner_task_benchmark.deepresearch9k_l2.data").joinpath(
        "versions",
        f"{snapshot.manifest.dataset_version}__{snapshot.manifest.scoring_version}",
    )
    checksum = sha256(version_dir.joinpath(snapshot.manifest.file_name).read_bytes()).hexdigest()

    assert checksum == snapshot.manifest.sha256


def test_deepresearch9k_l2_loads_current_and_explicit_versions_through_registry() -> None:
    snapshot = load_deepresearch9k_l2_snapshot()

    assert list_deepresearch9k_l2_snapshots() == (snapshot,)
    assert load_benchmark_snapshot(DEEPRESEARCH9K_L2_SUITE_SLUG) == snapshot
    assert (
        load_benchmark_snapshot(
            DEEPRESEARCH9K_L2_SUITE_SLUG,
            dataset_version=_DATASET_VERSION,
            scoring_version=_SCORING_VERSION,
        )
        == snapshot
    )


def test_deepresearch9k_l2_sampling_uses_preserved_source_item_indices() -> None:
    snapshot = load_deepresearch9k_l2_snapshot()
    run_id = UUID("00000000-0000-4000-8000-00000000d902")

    sampled = sample_benchmark_items(
        items=snapshot.items,
        run_id=run_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=20,
    )

    assert [item.item_index for item in sampled] == [
        1054,
        1081,
        1483,
        1636,
        2233,
        2368,
        3148,
        3169,
        3445,
        5380,
        5647,
        5761,
        6283,
        6754,
        7894,
        7963,
        8221,
        8593,
        8737,
        8953,
    ]
    assert str(
        benchmark_task_id_for_item(
            suite_slug=snapshot.manifest.suite_slug,
            run_id=run_id,
            item_index=sampled[0].item_index,
        )
    ) == "054d2331-688d-541a-9f6d-4b902b651603"
