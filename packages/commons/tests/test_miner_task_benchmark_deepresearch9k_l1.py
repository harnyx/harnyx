from __future__ import annotations

from hashlib import sha256
from importlib.resources import files
from uuid import UUID

from harnyx_commons.miner_task_benchmark import (
    DEEPRESEARCH9K_L1_SUITE_NAME,
    DEEPRESEARCH9K_L1_SUITE_SLUG,
    BenchmarkDatasetSnapshot,
    benchmark_task_id_for_item,
    load_benchmark_snapshot,
    load_deepresearch9k_l1_snapshot,
    sample_benchmark_items,
)


def test_load_deepresearch9k_l1_snapshot_preserves_source_item_indices() -> None:
    snapshot = load_deepresearch9k_l1_snapshot()

    assert isinstance(snapshot, BenchmarkDatasetSnapshot)
    assert snapshot.manifest.suite_slug == DEEPRESEARCH9K_L1_SUITE_SLUG
    assert snapshot.manifest.suite_name == DEEPRESEARCH9K_L1_SUITE_NAME
    assert snapshot.manifest.dataset_version == "2026-05-14-hf-9eaf02da-l1"
    assert snapshot.manifest.scoring_version == "correctness-v1"
    assert len(snapshot.items) == 3000
    assert snapshot.manifest.row_count == 3000
    assert [item.item_index for item in snapshot.items[:5]] == [0, 3, 6, 9, 12]
    assert snapshot.items[-1].item_index == 8997


def test_deepresearch9k_l1_manifest_checksum_matches_versioned_packaged_csv() -> None:
    snapshot = load_deepresearch9k_l1_snapshot()
    version_dir = files("harnyx_commons.miner_task_benchmark.deepresearch9k_l1.data").joinpath(
        "versions",
        f"{snapshot.manifest.dataset_version}__{snapshot.manifest.scoring_version}",
    )
    checksum = sha256(version_dir.joinpath(snapshot.manifest.file_name).read_bytes()).hexdigest()

    assert checksum == snapshot.manifest.sha256


def test_deepresearch9k_l1_loads_through_generic_registry() -> None:
    snapshot = load_deepresearch9k_l1_snapshot()

    assert load_benchmark_snapshot("deepresearch9k-l1") == snapshot
    assert (
        load_benchmark_snapshot(
            "deepresearch9k-l1",
            dataset_version=snapshot.manifest.dataset_version,
            scoring_version=snapshot.manifest.scoring_version,
        )
        == snapshot
    )


def test_deepresearch9k_l1_sampling_uses_preserved_source_item_indices() -> None:
    snapshot = load_deepresearch9k_l1_snapshot()
    run_id = UUID("00000000-0000-4000-8000-00000000d901")

    sampled = sample_benchmark_items(
        items=snapshot.items,
        run_id=run_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=20,
    )

    assert len(sampled) == 20
    assert [item.item_index for item in sampled] == [
        933,
        1029,
        1278,
        1965,
        2850,
        3147,
        4686,
        4974,
        4977,
        5319,
        6207,
        6387,
        6564,
        7011,
        7119,
        7647,
        7764,
        8922,
        8973,
        8979,
    ]
    assert all(item.item_index % 3 == 0 for item in sampled)
    assert [item.item_index for item in sampled] == sorted(item.item_index for item in sampled)
    assert str(
        benchmark_task_id_for_item(
            suite_slug=snapshot.manifest.suite_slug,
            run_id=run_id,
            item_index=sampled[0].item_index,
        )
    ) == "9c2e7286-0ed3-5d82-9fcc-58474d26f780"
