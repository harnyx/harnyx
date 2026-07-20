from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from harnyx_miner.agent_source import validate_agent_query_entrypoint
from harnyx_miner.platform_monitoring import platform_base_url_from_env

MINER_ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = MINER_ROOT / "prepare.py"
TRAIN_PATH = MINER_ROOT / "train.py"
PROGRAM_PATH = MINER_ROOT / "program.md"
AUTO_RESEARCH_PATH = MINER_ROOT / "AUTO-RESEARCH.md"
README_PATH = MINER_ROOT / "README.md"


def _load_prepare_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("harnyx_autoresearch_prepare", PREPARE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "identifiers": {"batch_id": "00000000-0000-0000-0000-000000000001"},
                "local_result_summary": {
                    "leaderboard": [
                        {
                            "label": "target",
                            "total_score": 0.67,
                            "error_count": 0,
                            "cost_totals": {"total_cost_usd": 0.02},
                        },
                        {
                            "label": "champion",
                            "total_score": 0.60,
                            "error_count": 0,
                            "cost_totals": {"total_cost_usd": 0.018},
                        },
                    ]
                },
                "tasks": [
                    {"target": {"elapsed_ms": 1200.0}},
                    {"target": {"elapsed_ms": 800.0}},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_benchmark_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark_metadata": {
                    "manifest": {
                        "suite_slug": "deepsearchqa",
                        "dataset_version": "2026-04-30",
                        "scoring_version": "correctness-v1",
                    }
                },
                "identifiers": {
                    "source_batch_id": "00000000-0000-0000-0000-000000000001",
                },
                "summary": {
                    "mean_total_score": 0.55,
                    "total_seconds": 12.3,
                    "error_count": 1,
                    "cost_totals": {"total_cost_usd": 0.01},
                },
            }
        ),
        encoding="utf-8",
    )


def test_train_py_is_valid_agent_source() -> None:
    validate_agent_query_entrypoint(TRAIN_PATH)


def test_prepare_parses_local_eval_report_summary(tmp_path: Path) -> None:
    prepare = _load_prepare_module()
    report_path = tmp_path / "local-eval-report-batch-vs-champion.json"
    _write_report(report_path)

    summary = prepare.parse_report_summary(report_path)

    assert summary.batch_id == "00000000-0000-0000-0000-000000000001"
    assert summary.score_a == 0.67
    assert summary.champion_score == 0.60
    assert round(summary.delta_vs_champion, 6) == 0.07
    assert summary.cost_usd == 0.02
    assert summary.error_count == 0
    assert summary.total_seconds == 2.0


def test_prepare_rejects_malformed_numeric_report_values(tmp_path: Path) -> None:
    prepare = _load_prepare_module()
    report_path = tmp_path / "local-eval-report-batch-vs-champion.json"
    _write_report(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["local_result_summary"]["leaderboard"][0]["total_score"] = math.nan
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="finite number"):
        prepare.parse_report_summary(report_path)


def test_prepare_builds_local_eval_command_for_pinned_batch(tmp_path: Path) -> None:
    prepare = _load_prepare_module()

    command = prepare.build_local_eval_command(
        agent_path=Path("train.py"),
        batch_id="00000000-0000-0000-0000-000000000001",
        output_dir=tmp_path,
    )

    assert command == [
        "uv",
        "run",
        "harnyx-miner-local-eval",
        "--agent-path",
        "train.py",
        "--batch-id",
        "00000000-0000-0000-0000-000000000001",
        "--mode",
        "vs-champion",
        "--output-dir",
        str(tmp_path),
    ]


def test_prepare_builds_local_benchmark_command(tmp_path: Path) -> None:
    prepare = _load_prepare_module()

    command = prepare.build_local_benchmark_command(
        agent_path=Path("train.py"),
        batch_id="00000000-0000-0000-0000-000000000001",
        benchmark=prepare.BenchmarkPin(
            suite_slug="deepsearchqa",
            dataset_version="2026-04-30",
            scoring_version="correctness-v1",
            sample_size=20,
        ),
        output_dir=tmp_path,
    )

    assert command == [
        "uv",
        "run",
        "harnyx-miner-local-benchmark",
        "--agent-path",
        "train.py",
        "--source-batch-id",
        "00000000-0000-0000-0000-000000000001",
        "--suite",
        "deepsearchqa",
        "--dataset-version",
        "2026-04-30",
        "--scoring-version",
        "correctness-v1",
        "--sample-size",
        "20",
        "--output-dir",
        str(tmp_path),
    ]


def test_prepare_requires_explicit_benchmark_suite(capsys: pytest.CaptureFixture[str]) -> None:
    prepare = _load_prepare_module()

    with pytest.raises(SystemExit) as exc_info:
        prepare.main(())

    assert exc_info.value.code == 2
    assert "--benchmark-suite" in capsys.readouterr().err


def test_prepare_resolves_explicit_benchmark_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare = _load_prepare_module()
    captured: dict[str, object] = {}

    class _Manifest:
        suite_slug = "webwalkerqa"
        dataset_version = "2026-05-14-webwalkerqa-test-single-source-easy"
        scoring_version = "correctness-v1"

    class _Snapshot:
        manifest = _Manifest()

    def fake_load_current_benchmark_snapshot(suite_slug: str):
        captured["suite_slug"] = suite_slug
        return _Snapshot()

    monkeypatch.setattr(prepare, "load_current_benchmark_snapshot", fake_load_current_benchmark_snapshot)

    pin = prepare._resolve_benchmark(benchmark_suite="webwalkerqa", sample_size=20)

    assert captured["suite_slug"] == "webwalkerqa"
    assert pin.suite_slug == "webwalkerqa"
    assert pin.dataset_version == "2026-05-14-webwalkerqa-test-single-source-easy"
    assert pin.scoring_version == "correctness-v1"
    assert pin.sample_size == 20


def test_prepare_resolves_draco_current_benchmark_suite() -> None:
    prepare = _load_prepare_module()

    pin = prepare._resolve_benchmark(benchmark_suite="draco", sample_size=1)

    assert pin.suite_slug == "draco"
    assert pin.dataset_version == "2026-06-16-hf-ce076749"
    assert pin.scoring_version == "weighted-rubric-v1"
    assert pin.sample_size == 1


def test_prepare_parses_local_benchmark_report_summary(tmp_path: Path) -> None:
    prepare = _load_prepare_module()
    report_path = tmp_path / "local-benchmark-report.json"
    _write_benchmark_report(report_path)

    summary = prepare.parse_benchmark_report_summary(report_path)

    assert summary.source_batch_id == "00000000-0000-0000-0000-000000000001"
    assert summary.suite_slug == "deepsearchqa"
    assert summary.dataset_version == "2026-04-30"
    assert summary.scoring_version == "correctness-v1"
    assert summary.score_b == 0.55
    assert summary.cost_usd == 0.01
    assert summary.error_count == 1
    assert summary.total_seconds == 12.3


def test_public_env_loader_finds_parent_env_from_miner_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_root = tmp_path / "public"
    miner_root = public_root / "miner"
    miner_root.mkdir(parents=True)
    (public_root / ".env").write_text("PLATFORM_BASE_URL=https://platform.example.com\n", encoding="utf-8")
    monkeypatch.chdir(miner_root)
    monkeypatch.delenv("PLATFORM_BASE_URL", raising=False)

    assert platform_base_url_from_env() == "https://platform.example.com"


def test_public_program_matches_single_file_contract() -> None:
    program = PROGRAM_PATH.read_text(encoding="utf-8")

    assert "Modify only `train.py`" in program
    assert "Do not modify `prepare.py`" in program
    assert "Do not commit `results.tsv`" in program
    assert "score_a" in program
    assert "score_b" in program
    assert "hardcode benchmark item IDs" in program
    assert "LOG_LEVEL=DEBUG uv run train.py > run.log 2>&1" in program
    assert ".autoresearch/reports/<timestamp>/local-eval.stderr" in program


def test_public_program_requires_failure_first_research_cycle() -> None:
    program = PROGRAM_PATH.read_text(encoding="utf-8")

    assert "The agent must behave like a rigorous research engineer" in program
    assert "### Start from failures" in program
    assert "### Build a failure taxonomy" in program
    assert "### Pick one bottleneck per cycle" in program
    assert "### Write the hypothesis before editing" in program
    assert "### Use focused diagnostic cases before full evaluation" in program
    assert "Full evaluation with `LOG_LEVEL=DEBUG uv run train.py > run.log 2>&1` is allowed only when:" in program
    assert "### Inspect intermediate artifacts, not just score" in program
    assert "### Do not abandon a hypothesis after one failed attempt" in program
    assert "### Use the intervention ladder" in program
    assert "## Research ledger" in program
    assert ".autoresearch/experiment-ledger.md" in program

    assert program.index("### Start from failures") < program.index("### Build a failure taxonomy")
    assert program.index("### Build a failure taxonomy") < program.index("### Pick one bottleneck per cycle")
    assert program.index("### Pick one bottleneck per cycle") < program.index(
        "### Write the hypothesis before editing"
    )
    assert program.index("### Write the hypothesis before editing") < program.index(
        "Only then edit `train.py`"
    )
    assert program.index("Only then edit `train.py`") < program.index(
        "### Use focused diagnostic cases before full evaluation"
    )


def test_auto_research_runbook_documents_operator_startup_contract() -> None:
    runbook = AUTO_RESEARCH_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "## What To Tell The Agent" in runbook
    assert "Read README.md first, then read program.md" in runbook
    assert "Only edit train.py" in runbook
    assert "uv run prepare.py --benchmark-suite" in runbook
    assert "results.tsv" in runbook
    assert ".autoresearch/experiment-ledger.md" in runbook
    assert "PLATFORM_BASE_URL" in runbook
    assert "CHUTES_API_KEY" in runbook
    assert "SEARCH_PROVIDER" in runbook
    assert "DESEARCH_API_KEY" in runbook
    assert "BENCHMARK_LLM_PROVIDER" in runbook
    assert "BENCHMARK_LLM_MODEL" in runbook
    assert "LOG_LEVEL=DEBUG uv run train.py > run.log 2>&1" in runbook
    assert "local-eval.stderr" in runbook
    assert "AutoResearch does not upload automatically" in runbook
    assert "cd miner" in runbook
    assert "public/miner" not in runbook
    assert "public/.env" not in runbook

    assert "[`AUTO-RESEARCH.md`](AUTO-RESEARCH.md)" in readme
    assert "The agent-facing research policy lives in [`program.md`](program.md)" in readme
