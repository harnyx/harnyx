from __future__ import annotations

import json
from hashlib import sha256
from importlib.resources import files
from uuid import UUID

import harnyx_commons.miner_task_benchmark.registry as registry_mod
from harnyx_commons.miner_task_benchmark import (
    DEEPRESEARCH9K_L1_SUITE_SLUG,
    BenchmarkDatasetSnapshot,
    benchmark_backing_batch_id_for_run,
    benchmark_run_id_for_source_batch,
    benchmark_task_id_for_item,
    list_current_benchmark_snapshots,
    list_current_benchmark_suite_slugs,
    list_deepsearchqa_snapshots,
    load_active_benchmark_snapshot,
    load_benchmark_snapshot,
    load_current_benchmark_snapshot,
    load_deepsearchqa_snapshot,
    sample_benchmark_items,
)


def test_load_deepsearchqa_snapshot_reads_packaged_manifest_and_data() -> None:
    snapshot = load_deepsearchqa_snapshot()

    assert isinstance(snapshot, BenchmarkDatasetSnapshot)
    assert snapshot.manifest.suite_slug == "deepsearchqa"
    assert snapshot.manifest.dataset_version
    assert snapshot.manifest.scoring_version
    assert len(snapshot.items) == snapshot.manifest.row_count
    assert snapshot.items[0].item_index == 0
    assert snapshot.items[-1].item_index == snapshot.manifest.row_count - 1


def test_benchmark_registry_loads_deepsearchqa_snapshot_generically() -> None:
    snapshot = load_deepsearchqa_snapshot()

    assert load_benchmark_snapshot("deepsearchqa") == snapshot
    assert (
        load_benchmark_snapshot(
            "deepsearchqa",
            dataset_version=snapshot.manifest.dataset_version,
            scoring_version=snapshot.manifest.scoring_version,
        )
        == snapshot
    )
    assert "deepsearchqa" in list_current_benchmark_suite_slugs()
    assert snapshot in list_current_benchmark_snapshots()
    assert load_current_benchmark_snapshot("deepsearchqa") == snapshot


def test_benchmark_registry_has_three_current_suites_without_single_active_default() -> None:
    snapshots = list_current_benchmark_snapshots()

    assert tuple(snapshot.manifest.suite_slug for snapshot in snapshots) == (
        "deepresearch9k-l1",
        "deepsearchqa",
        "webwalkerqa",
    )
    assert list_current_benchmark_suite_slugs() == (
        "deepresearch9k-l1",
        "deepsearchqa",
        "webwalkerqa",
    )
    assert load_current_benchmark_snapshot(DEEPRESEARCH9K_L1_SUITE_SLUG).manifest.row_count == 3000
    try:
        load_active_benchmark_snapshot()
    except RuntimeError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("load_active_benchmark_snapshot must not resolve when multiple suites are current")


def test_deepsearchqa_loader_retains_current_snapshot_in_version_catalog() -> None:
    snapshot = load_deepsearchqa_snapshot()
    snapshots = list_deepsearchqa_snapshots()

    assert snapshots == (snapshot,)


def test_load_deepsearchqa_snapshot_manifest_checksum_matches_versioned_packaged_csv() -> None:
    snapshot = load_deepsearchqa_snapshot()
    version_dir = files("harnyx_commons.miner_task_benchmark.deepsearchqa.data").joinpath(
        "versions",
        f"{snapshot.manifest.dataset_version}__{snapshot.manifest.scoring_version}",
    )
    checksum = sha256(version_dir.joinpath(snapshot.manifest.file_name).read_bytes()).hexdigest()

    assert checksum == snapshot.manifest.sha256


def test_load_deepsearchqa_snapshot_current_version_points_at_versioned_payload() -> None:
    snapshot = load_deepsearchqa_snapshot()
    data_dir = files("harnyx_commons.miner_task_benchmark.deepsearchqa.data")
    current_version = json.loads(
        data_dir.joinpath("current_version.json").read_text(encoding="utf-8")
    )

    assert current_version == {
        "dataset_version": snapshot.manifest.dataset_version,
        "scoring_version": snapshot.manifest.scoring_version,
    }


def test_benchmark_registry_resolves_explicit_active_version_only(monkeypatch) -> None:
    current = load_deepsearchqa_snapshot()
    monkeypatch.setitem(
        registry_mod._BENCHMARK_SNAPSHOT_CATALOG_LOADERS,
        "deepsearchqa",
        lambda: (current,),
    )
    monkeypatch.setitem(
        registry_mod._BENCHMARK_CURRENT_SNAPSHOT_LOADERS,
        "deepsearchqa",
        lambda: current,
    )

    assert (
        load_benchmark_snapshot(
            "deepsearchqa",
            dataset_version=current.manifest.dataset_version,
            scoring_version=current.manifest.scoring_version,
        )
        == current
    )
    assert load_benchmark_snapshot("deepsearchqa") == current


def test_sample_benchmark_items_is_deterministic_for_run_id() -> None:
    snapshot = load_deepsearchqa_snapshot()
    run_id = UUID("00000000-0000-4000-8000-00000000b501")

    first = sample_benchmark_items(
        items=snapshot.items,
        run_id=run_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=20,
    )
    second = sample_benchmark_items(
        items=snapshot.items,
        run_id=run_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=20,
    )

    assert first == second
    assert len(first) == 20
    assert [item.item_index for item in first] == sorted(item.item_index for item in first)


def test_benchmark_identity_helpers_match_existing_public_values() -> None:
    source_batch_id = UUID("00000000-0000-4000-8000-00000000b501")
    run_id = benchmark_run_id_for_source_batch(
        suite_slug="deepsearchqa",
        source_batch_id=source_batch_id,
        dataset_version="2026-04-02-google-main",
        scoring_version="correctness-v1",
    )

    assert str(run_id) == "d40019ec-5d16-5ba1-b30d-545c8c5d252d"
    assert str(benchmark_backing_batch_id_for_run(suite_slug="deepsearchqa", run_id=run_id)) == (
        "d4ca3d15-ca41-5af1-a692-1f150d0a8463"
    )
    assert str(benchmark_task_id_for_item(suite_slug="deepsearchqa", run_id=run_id, item_index=0)) == (
        "b46064ba-ed49-5552-a61a-8c9dbc7913e6"
    )
    assert str(benchmark_task_id_for_item(suite_slug="deepsearchqa", run_id=run_id, item_index=17)) == (
        "8b511d85-6c81-58c4-a101-7feef9999c73"
    )
