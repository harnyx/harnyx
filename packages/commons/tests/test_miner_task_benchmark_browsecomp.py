from __future__ import annotations

import base64
import json
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from shutil import copyfile
from uuid import UUID

import pytest

from harnyx_commons.miner_task_benchmark import (
    BenchmarkAnswerType,
    BenchmarkDatasetSnapshot,
    list_browsecomp_snapshots,
    list_current_benchmark_suite_slugs,
    load_benchmark_snapshot,
    load_browsecomp_snapshot,
    load_current_benchmark_snapshot,
    sample_benchmark_items,
)
from harnyx_commons.miner_task_benchmark.browsecomp.loader import (
    _decrypt_required,
    _load_snapshot_from_dir,
    _parse_rows,
)

_DATASET_VERSION = "2026-08-11-openai-browsecomp"
_SCORING_VERSION = "correctness-v1"
_SOURCE_SHA256 = "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
_LICENSE_SHA256 = "d831db55645e47ca8e491c5a0e37f1ee744d7b10bf5aa8d50146c795ac0176c0"


def test_load_browsecomp_snapshot_reads_strict_pinned_source() -> None:
    snapshot = load_browsecomp_snapshot()

    assert isinstance(snapshot, BenchmarkDatasetSnapshot)
    assert snapshot.manifest == snapshot.manifest.__class__(
        suite_slug="browsecomp",
        suite_name="BrowseComp",
        dataset_version=_DATASET_VERSION,
        scoring_version=_SCORING_VERSION,
        source_url="https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv",
        source_page_url="https://github.com/openai/simple-evals",
        license="MIT",
        sha256=_SOURCE_SHA256,
        row_count=1266,
        file_name="browse_comp_test_set.csv",
        fetched_at="2026-08-11",
    )
    assert len(snapshot.items) == 1266
    assert snapshot.items[0].item_index == 0
    assert snapshot.items[-1].item_index == 1265
    assert {item.answer_type for item in snapshot.items} == {BenchmarkAnswerType.SINGLE_ANSWER}
    assert all(item.problem and item.answer and item.problem_category for item in snapshot.items)


def test_browsecomp_snapshot_decodes_stable_rows_without_exposing_plaintext() -> None:
    snapshot = load_browsecomp_snapshot()

    decoded_hashes = tuple(
        (
            item.item_index,
            sha256(item.problem.encode()).hexdigest(),
            sha256(item.answer.encode()).hexdigest(),
            sha256(item.problem_category.encode()).hexdigest(),
        )
        for item in (snapshot.items[0], snapshot.items[632], snapshot.items[-1])
    )
    assert decoded_hashes == (
        (
            0,
            "6662dea134c10c3e9e1b19eff2ee5d56be2217f1c7b0d8e309eddbf775a44978",
            "e755d3d9f41b38b2bbe0e90ca9d917d568ebc9a7fe186ba75cb6fb7cde6f8419",
            "75df3579c73089e317e5bf917f6b6cf1cb8d19f21ad70c15e70e431c3aa2a62a",
        ),
        (
            632,
            "18e0b41ff4946d40537f54d3ad10d02b89587d45f7f430ad3291567b01aab5d5",
            "99d50e032223964c51e69e3bd6ea4e639521aeccb24774bc3aef057440d5ade0",
            "0e769600933790607b2a13b33ddfade0fa17810eb62c3b28ee23e59516516491",
        ),
        (
            1265,
            "25af3f67e7d8a59682799b56e1a62a055746381b3b300436ce46bb70e6abca15",
            "f2b5bb06179819f12ff841d525942581abd43408161e827f7e0c122afa3fa6a2",
            "b5c0710b0aa85969518954ca2d08c86b2685a455814240613c716f2f95be2d96",
        ),
    )


def test_browsecomp_registry_and_catalog_resolve_current_and_exact_snapshot() -> None:
    snapshot = load_browsecomp_snapshot()

    assert list_browsecomp_snapshots() == (snapshot,)
    assert load_current_benchmark_snapshot("browsecomp") == snapshot
    assert load_benchmark_snapshot("browsecomp") == snapshot
    assert (
        load_benchmark_snapshot(
            "browsecomp",
            dataset_version=_DATASET_VERSION,
            scoring_version=_SCORING_VERSION,
        )
        == snapshot
    )
    assert "browsecomp" in list_current_benchmark_suite_slugs()


def test_browsecomp_sampling_uses_fixed_snapshot_panel() -> None:
    snapshot = load_browsecomp_snapshot()

    sampled = sample_benchmark_items(
        items=snapshot.items,
        run_id=UUID("00000000-0000-4000-8000-00000000bc01"),
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=20,
    )

    assert [item.item_index for item in sampled] == [
        65,
        155,
        292,
        317,
        318,
        348,
        367,
        399,
        431,
        432,
        477,
        517,
        554,
        779,
        835,
        1035,
        1043,
        1066,
        1115,
        1231,
    ]


def test_browsecomp_packaged_source_and_license_match_pinned_bytes() -> None:
    version_dir = _version_dir()

    assert sha256(version_dir.joinpath("browse_comp_test_set.csv").read_bytes()).hexdigest() == (
        _SOURCE_SHA256
    )
    license_bytes = version_dir.joinpath("LICENSE.openai-simple-evals").read_bytes()
    assert len(license_bytes) == 1063
    assert sha256(license_bytes).hexdigest() == _LICENSE_SHA256


def test_browsecomp_current_pointer_matches_pinned_version() -> None:
    data_dir = files("harnyx_commons.miner_task_benchmark.browsecomp.data")

    assert json.loads(data_dir.joinpath("current_version.json").read_text(encoding="utf-8")) == {
        "dataset_version": _DATASET_VERSION,
        "scoring_version": _SCORING_VERSION,
    }


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"answer,problem,problem_topic,canary\nA,P,T,C\n", "header"),
        (b"problem,answer,problem_topic,problem_topic\nP,A,T,C\n", "header"),
        (b"problem,answer,problem_topic,canary\nP,A,T\n", "missing cell"),
        (b"problem,answer,problem_topic,canary\nP,A,T,C,extra\n", "extra cell"),
        (b"problem,answer,problem_topic,canary\nP,,T,C\n", "empty answer"),
    ],
)
def test_browsecomp_parser_rejects_malformed_rows(raw: bytes, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        _parse_rows(raw)


def test_browsecomp_decryption_rejects_invalid_base64_empty_ciphertext_and_utf8() -> None:
    with pytest.raises(RuntimeError, match="problem.*base64"):
        _decrypt_required("not base64!", "canary", field="problem")
    with pytest.raises(RuntimeError, match="problem.*empty ciphertext"):
        _decrypt_required("", "canary", field="problem")

    key_byte = sha256(b"canary").digest()[0]
    invalid_utf8_ciphertext = base64.b64encode(bytes([0xFF ^ key_byte])).decode()
    with pytest.raises(RuntimeError, match="problem.*UTF-8"):
        _decrypt_required(invalid_utf8_ciphertext, "canary", field="problem")


def test_browsecomp_loader_rejects_manifest_and_checksum_drift(tmp_path: Path) -> None:
    snapshot_dir = _copy_version_dir(tmp_path)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 1265
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest mismatch"):
        _load_snapshot_from_dir(snapshot_dir)

    snapshot_dir = _copy_version_dir(tmp_path / "checksum")
    source_path = snapshot_dir / "browse_comp_test_set.csv"
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _load_snapshot_from_dir(snapshot_dir)


def _version_dir():
    return files("harnyx_commons.miner_task_benchmark.browsecomp.data").joinpath(
        "versions",
        f"{_DATASET_VERSION}__{_SCORING_VERSION}",
    )


def _copy_version_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    version_dir = _version_dir()
    for name in ("manifest.json", "browse_comp_test_set.csv", "LICENSE.openai-simple-evals"):
        copyfile(version_dir.joinpath(name), root / name)
    return root
