from __future__ import annotations

import json
from hashlib import sha256
from importlib.resources import files
from uuid import UUID

from harnyx_commons.miner_task_benchmark import (
    BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    DRACO_SUITE_NAME,
    DRACO_SUITE_SLUG,
    BenchmarkAnswerType,
    benchmark_task_id_for_item,
    list_current_benchmark_suite_slugs,
    list_draco_snapshots,
    load_benchmark_snapshot,
    load_current_benchmark_snapshot,
    load_draco_snapshot,
    parse_weighted_rubric,
    sample_benchmark_items,
)

DRACO_DATASET_VERSION = "2026-06-16-hf-ce076749"
DRACO_CHECKSUM = "e35bfe78cd827fa1d541b79fbc7bc7b91966d3227d8742c83e99d26d4ac4679a"


def test_load_draco_snapshot_reads_pinned_manifest_and_rubric_items() -> None:
    snapshot = load_draco_snapshot(
        dataset_version=DRACO_DATASET_VERSION,
        scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    )

    assert snapshot.manifest.suite_slug == DRACO_SUITE_SLUG
    assert snapshot.manifest.suite_name == DRACO_SUITE_NAME
    assert snapshot.manifest.dataset_version == DRACO_DATASET_VERSION
    assert snapshot.manifest.scoring_version == BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION
    assert snapshot.manifest.sha256 == DRACO_CHECKSUM
    assert snapshot.manifest.row_count == 100
    assert len(snapshot.items) == 100
    assert [item.item_index for item in snapshot.items] == list(range(100))
    assert {item.answer_type for item in snapshot.items} == {BenchmarkAnswerType.SINGLE_ANSWER}
    assert snapshot.items[0].source_item_id == "0c2c668a-c3bf-41af-93c9-b5614ff63508"
    assert snapshot.items[-1].source_item_id == "91408757-a874-44b5-ad5a-66a22b39141d"
    assert snapshot.items[0].problem_category == "Academic"
    assert snapshot.items[-1].problem_category == "Technology"
    assert parse_weighted_rubric(snapshot.items[0].answer).rubric_id == (
        "staggered-did-methodology-evaluation"
    )


def test_draco_manifest_checksum_matches_packaged_jsonl() -> None:
    snapshot = load_draco_snapshot(
        dataset_version=DRACO_DATASET_VERSION,
        scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    )
    version_dir = files("harnyx_commons.miner_task_benchmark.draco.data").joinpath(
        "versions",
        f"{DRACO_DATASET_VERSION}__{BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION}",
    )
    payload = version_dir.joinpath(snapshot.manifest.file_name).read_bytes()

    assert sha256(payload).hexdigest() == DRACO_CHECKSUM
    assert len(payload.splitlines()) == 100


def test_draco_packaged_rubric_statistics_match_verified_source() -> None:
    snapshot = load_draco_snapshot(
        dataset_version=DRACO_DATASET_VERSION,
        scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    )
    criteria: list[dict[str, object]] = []
    section_orders: set[tuple[str, ...]] = set()

    for item in snapshot.items:
        row = json.loads(item.answer)
        section_orders.add(tuple(section["id"] for section in row["sections"]))
        criteria.extend(criterion for section in row["sections"] for criterion in section["criteria"])

    weights = tuple(int(criterion["weight"]) for criterion in criteria)
    assert section_orders == {
        (
            "factual-accuracy",
            "breadth-and-depth-of-analysis",
            "presentation-quality",
            "citation-quality",
        )
    }
    assert len(criteria) == 3934
    assert sum(1 for weight in weights if weight < 0) == 415
    assert min(weights) == -500
    assert max(weights) == 20


def test_draco_registry_is_current_and_explicit_version_loadable() -> None:
    snapshot = load_draco_snapshot(
        dataset_version=DRACO_DATASET_VERSION,
        scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    )

    assert load_draco_snapshot() == snapshot
    assert (
        load_benchmark_snapshot(
            "draco",
            dataset_version=DRACO_DATASET_VERSION,
            scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
        )
        == snapshot
    )
    assert "draco" in list_current_benchmark_suite_slugs()
    assert load_benchmark_snapshot("draco") == snapshot
    assert load_current_benchmark_snapshot("draco") == snapshot


def test_draco_current_version_points_at_versioned_payload() -> None:
    snapshot = load_draco_snapshot(
        dataset_version=DRACO_DATASET_VERSION,
        scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    )
    data_dir = files("harnyx_commons.miner_task_benchmark.draco.data")
    current_version = json.loads(
        data_dir.joinpath("current_version.json").read_text(encoding="utf-8")
    )

    assert current_version == {
        "dataset_version": snapshot.manifest.dataset_version,
        "scoring_version": snapshot.manifest.scoring_version,
    }


def test_draco_snapshot_catalog_contains_only_pinned_snapshot() -> None:
    assert list_draco_snapshots() == (
        load_draco_snapshot(
            dataset_version=DRACO_DATASET_VERSION,
            scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
        ),
    )


def test_draco_sampling_uses_fixed_snapshot_panel() -> None:
    snapshot = load_draco_snapshot(
        dataset_version=DRACO_DATASET_VERSION,
        scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    )
    run_id = UUID("00000000-0000-4000-8000-00000000d905")

    sampled = sample_benchmark_items(
        items=snapshot.items,
        run_id=run_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=20,
    )

    assert [item.item_index for item in sampled] == [
        7,
        17,
        29,
        30,
        31,
        35,
        38,
        45,
        52,
        54,
        63,
        68,
        69,
        76,
        79,
        83,
        87,
        91,
        94,
        95,
    ]
    assert str(
        benchmark_task_id_for_item(
            suite_slug=snapshot.manifest.suite_slug,
            run_id=run_id,
            item_index=sampled[0].item_index,
        )
    ) == "ef30afcd-a058-5500-8640-a9dc88a1eb3e"
