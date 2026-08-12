from __future__ import annotations

import json
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from uuid import UUID

import harnyx_commons.miner_task_benchmark.webwalkerqa.loader as webwalkerqa_loader
from harnyx_commons.miner_task_benchmark import (
    WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_NAME,
    WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_SLUG,
    WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_NAME,
    WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_SLUG,
    BenchmarkAnswerType,
    benchmark_backing_batch_id_for_run,
    benchmark_run_id_for_source_batch,
    benchmark_task_id_for_item,
    list_current_benchmark_snapshots,
    list_current_benchmark_suite_slugs,
    list_webwalkerqa_multi_source_medium_snapshots,
    list_webwalkerqa_single_source_medium_snapshots,
    list_webwalkerqa_snapshots,
    load_benchmark_snapshot,
    load_webwalkerqa_multi_source_medium_snapshot,
    load_webwalkerqa_single_source_medium_snapshot,
    load_webwalkerqa_snapshot,
    sample_benchmark_items,
)

_SINGLE_SOURCE_MEDIUM_VERSION = "2026-07-22-webwalkerqa-test-single-source-medium"
_MULTI_SOURCE_MEDIUM_VERSION = "2026-07-22-webwalkerqa-test-multi-source-medium"


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


def test_webwalkerqa_easy_validates_task_fields_only_for_selected_rows(tmp_path: Path) -> None:
    rows = [
        {
            "Question": "Selected question",
            "answer": "Selected answer",
            "root_url": "https://example.com/",
            "source_website": ["https://example.com/answer"],
            "type": "single_source",
            "difficulty_level": "easy",
        },
        {
            "type": "multi_source",
            "difficulty_level": "medium",
        },
    ]
    payload = json.dumps(rows).encode()
    payload_hash = sha256(payload).hexdigest()
    sources_dir = tmp_path.joinpath("sources")
    source_dir = sources_dir.joinpath(payload_hash)
    source_dir.mkdir(parents=True)
    source_dir.joinpath("test.json").write_bytes(payload)
    tmp_path.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "suite_slug": "webwalkerqa",
                "suite_name": "WebWalkerQA Easy",
                "dataset_version": "test-version",
                "scoring_version": "correctness-v1",
                "source_url": "https://example.com/test.json",
                "source_page_url": "https://example.com/",
                "license": "test",
                "sha256": payload_hash,
                "row_count": 1,
                "file_name": "test.json",
                "fetched_at": "2026-07-22",
            }
        ),
        encoding="utf-8",
    )

    snapshot = webwalkerqa_loader._load_snapshot_from_dir(
        snapshot_dir=tmp_path,
        sources_dir=sources_dir,
        spec=webwalkerqa_loader._WEBWALKERQA_EASY,
    )

    assert [item.problem for item in snapshot.items] == [
        "Root URL: https://example.com/\nQuestion: Selected question"
    ]


def test_webwalkerqa_problem_uses_root_url_and_question_only() -> None:
    item = load_webwalkerqa_snapshot().items[0]

    assert item.problem == (
        "Root URL: https://computer.hut.edu.cn/\n"
        "Question: 湖南工业大学计算机学院“工作动态”部分中，记录的最早日期是什么？"
    )
    assert item.answer == "2021-11-02"


def test_webwalkerqa_manifest_checksum_matches_upstream_raw_test_json() -> None:
    snapshot = load_webwalkerqa_snapshot()
    payload = files("harnyx_commons.miner_task_benchmark.webwalkerqa.data").joinpath(
        "sources",
        snapshot.manifest.sha256,
        snapshot.manifest.file_name,
    )
    raw_bytes = payload.read_bytes()
    raw_rows = json.loads(raw_bytes.decode("utf-8"))

    assert len(raw_rows) == 680
    assert sha256(raw_bytes).hexdigest() == snapshot.manifest.sha256
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
        "browsecomp",
        "deepresearch9k-l1",
        "deepresearch9k-l2",
        "deepsearchqa",
        "draco",
        "webwalkerqa",
        "webwalkerqa-multi-source-medium",
        "webwalkerqa-single-source-medium",
    )
    assert {item.manifest.suite_slug for item in list_current_benchmark_snapshots()} == {
        "browsecomp",
        "deepresearch9k-l1",
        "deepresearch9k-l2",
        "deepsearchqa",
        "draco",
        "webwalkerqa",
        "webwalkerqa-multi-source-medium",
        "webwalkerqa-single-source-medium",
    }


def test_webwalkerqa_identity_and_sampling_use_fixed_snapshot_panel() -> None:
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
        40,
        122,
        126,
        128,
        131,
        132,
        137,
        139,
        140,
        141,
        142,
        144,
        145,
        147,
        148,
        150,
        155,
        172,
        184,
        188,
    ]


def test_webwalkerqa_medium_snapshots_keep_populations_separate() -> None:
    single = load_webwalkerqa_single_source_medium_snapshot()
    multi = load_webwalkerqa_multi_source_medium_snapshot()

    assert single.manifest.suite_slug == WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_SLUG
    assert single.manifest.suite_name == WEBWALKERQA_SINGLE_SOURCE_MEDIUM_SUITE_NAME
    assert single.manifest.dataset_version == _SINGLE_SOURCE_MEDIUM_VERSION
    assert single.manifest.row_count == 140
    assert len(single.items) == 140
    assert {item.problem_category for item in single.items} == {"single_source_medium"}
    assert [item.item_index for item in single.items[:5]] == [0, 1, 2, 19, 20]
    assert single.items[-1].item_index == 653

    assert multi.manifest.suite_slug == WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_SLUG
    assert multi.manifest.suite_name == WEBWALKERQA_MULTI_SOURCE_MEDIUM_SUITE_NAME
    assert multi.manifest.dataset_version == _MULTI_SOURCE_MEDIUM_VERSION
    assert multi.manifest.row_count == 140
    assert len(multi.items) == 140
    assert {item.problem_category for item in multi.items} == {"multi_source_medium"}
    assert [item.item_index for item in multi.items[:5]] == [5, 6, 7, 8, 9]
    assert multi.items[-1].item_index == 678


def test_webwalkerqa_medium_snapshots_load_current_and_explicit_versions() -> None:
    single = load_webwalkerqa_single_source_medium_snapshot()
    multi = load_webwalkerqa_multi_source_medium_snapshot()

    assert list_webwalkerqa_single_source_medium_snapshots() == (single,)
    assert list_webwalkerqa_multi_source_medium_snapshots() == (multi,)
    for snapshot in (single, multi):
        assert load_benchmark_snapshot(snapshot.manifest.suite_slug) == snapshot
        assert (
            load_benchmark_snapshot(
                snapshot.manifest.suite_slug,
                dataset_version=snapshot.manifest.dataset_version,
                scoring_version=snapshot.manifest.scoring_version,
            )
            == snapshot
        )


def test_webwalkerqa_medium_sampling_uses_fixed_snapshot_panels() -> None:
    expectations = (
        (
            load_webwalkerqa_single_source_medium_snapshot(),
            UUID("00000000-0000-4000-8000-00000000d903"),
            [21, 38, 80, 81, 82, 201, 207, 219, 227, 233, 249, 252, 253, 257, 269, 284, 291, 295, 637, 648],
            "e5eb6fcf-8577-59f6-bfde-715ae840f3ec",
        ),
        (
            load_webwalkerqa_multi_source_medium_snapshot(),
            UUID("00000000-0000-4000-8000-00000000d904"),
            [8, 11, 26, 32, 53, 56, 93, 106, 112, 484, 487, 492, 497, 503, 506, 515, 638, 660, 666, 675],
            "78786263-1ab5-5dbb-9028-6ebac61dfe5a",
        ),
    )

    for snapshot, run_id, expected_indices, expected_first_task_id in expectations:
        sampled = sample_benchmark_items(
            items=snapshot.items,
            run_id=run_id,
            dataset_version=snapshot.manifest.dataset_version,
            scoring_version=snapshot.manifest.scoring_version,
            sample_size=20,
        )

        assert [item.item_index for item in sampled] == expected_indices
        assert str(
            benchmark_task_id_for_item(
                suite_slug=snapshot.manifest.suite_slug,
                run_id=run_id,
                item_index=sampled[0].item_index,
            )
        ) == expected_first_task_id
