from __future__ import annotations

import json
from hashlib import sha256
from importlib.resources import files
from uuid import UUID

from harnyx_commons.miner_task_benchmark import (
    BenchmarkAnswerType,
    benchmark_backing_batch_id_for_run,
    benchmark_run_id_for_source_batch,
    benchmark_task_id_for_item,
    list_current_benchmark_snapshots,
    list_current_benchmark_suite_slugs,
    list_webwalkerqa_snapshots,
    load_benchmark_snapshot,
    load_webwalkerqa_snapshot,
    sample_benchmark_items,
)


def test_load_webwalkerqa_snapshot_reads_packaged_manifest_and_filters_easy_single_source_rows() -> None:
    snapshot = load_webwalkerqa_snapshot()

    assert snapshot.manifest.suite_slug == "webwalkerqa"
    assert snapshot.manifest.suite_name == "WebWalkerQA Easy"
    assert snapshot.manifest.dataset_version == "2026-05-14-webwalkerqa-test-single-source-easy"
    assert snapshot.manifest.scoring_version == "correctness-v1"
    assert snapshot.manifest.row_count == 80
    assert len(snapshot.items) == 80
    assert snapshot.items[0].item_index == 40
    assert snapshot.items[-1].item_index == 647
    assert {item.problem_category for item in snapshot.items} == {"single_source_easy"}
    assert {item.answer_type for item in snapshot.items} == {BenchmarkAnswerType.SINGLE_ANSWER}


def test_webwalkerqa_problem_uses_root_url_and_question_only() -> None:
    item = load_webwalkerqa_snapshot().items[0]

    assert item.problem == (
        "Root URL: https://computer.hut.edu.cn/\n"
        "Question: 湖南工业大学计算机学院“工作动态”部分中，记录的最早日期是什么？"
    )
    assert item.answer == "2021-11-02"


def test_webwalkerqa_manifest_checksum_matches_upstream_raw_test_json() -> None:
    snapshot = load_webwalkerqa_snapshot()
    version_dir = files("harnyx_commons.miner_task_benchmark.webwalkerqa.data").joinpath(
        "versions",
        f"{snapshot.manifest.dataset_version}__{snapshot.manifest.scoring_version}",
    )
    payload = version_dir.joinpath(snapshot.manifest.file_name).read_bytes()
    raw_rows = json.loads(payload.decode("utf-8"))

    assert len(raw_rows) == 680
    assert sha256(payload).hexdigest() == snapshot.manifest.sha256
    assert snapshot.manifest.sha256 == "26743935e573cca30571793bc28f3798d2a7ce73c6c0981e1bd54a5fe476fe46"


def test_webwalkerqa_current_version_points_at_versioned_payload() -> None:
    snapshot = load_webwalkerqa_snapshot()
    data_dir = files("harnyx_commons.miner_task_benchmark.webwalkerqa.data")
    current_version = json.loads(
        data_dir.joinpath("current_version.json").read_text(encoding="utf-8")
    )

    assert current_version == {
        "dataset_version": snapshot.manifest.dataset_version,
        "scoring_version": snapshot.manifest.scoring_version,
    }
    assert list_webwalkerqa_snapshots() == (snapshot,)


def test_benchmark_registry_loads_webwalkerqa_current_and_explicit_snapshot() -> None:
    snapshot = load_webwalkerqa_snapshot()

    assert load_benchmark_snapshot("webwalkerqa") == snapshot
    assert (
        load_benchmark_snapshot(
            "webwalkerqa",
            dataset_version=snapshot.manifest.dataset_version,
            scoring_version=snapshot.manifest.scoring_version,
        )
        == snapshot
    )
    assert list_current_benchmark_suite_slugs() == (
        "deepresearch9k-l1",
        "deepsearchqa",
        "webwalkerqa",
    )
    assert {item.manifest.suite_slug for item in list_current_benchmark_snapshots()} == {
        "deepresearch9k-l1",
        "deepsearchqa",
        "webwalkerqa",
    }


def test_webwalkerqa_identity_and_sampling_match_recorded_local_run_anchor() -> None:
    snapshot = load_webwalkerqa_snapshot()
    source_batch_id = UUID("855ad3da-c8f2-4114-abab-50c0463c4814")
    run_id = benchmark_run_id_for_source_batch(
        suite_slug=snapshot.manifest.suite_slug,
        source_batch_id=source_batch_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
    )

    sampled_items = sample_benchmark_items(
        items=snapshot.items,
        run_id=run_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=20,
    )

    assert str(run_id) == "5dba4369-0152-5553-876e-61afc2066201"
    assert str(benchmark_backing_batch_id_for_run(suite_slug="webwalkerqa", run_id=run_id)) == (
        "837e691c-0dec-5a1e-8eb5-f936dd9ac2bb"
    )
    assert str(benchmark_task_id_for_item(suite_slug="webwalkerqa", run_id=run_id, item_index=40)) == (
        "3bcb5b02-2003-581e-8512-15a6640fdaf6"
    )
    assert str(benchmark_task_id_for_item(suite_slug="webwalkerqa", run_id=run_id, item_index=647)) == (
        "31e0c534-8de3-5563-9184-86f8f8293629"
    )
    assert [item.item_index for item in sampled_items] == [
        41,
        78,
        120,
        133,
        136,
        140,
        141,
        144,
        147,
        148,
        150,
        153,
        154,
        157,
        168,
        173,
        177,
        182,
        187,
        188,
    ]
