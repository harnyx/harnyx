"""
Fixed support for Harnyx miner autoresearch experiments.

Usage:
    uv run prepare.py --benchmark-suite <suite-slug>

Experiment agents should read this file for context, but should not edit it.
The editable experiment file is train.py.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from harnyx_commons.miner_task_benchmark import load_current_benchmark_snapshot
from harnyx_miner.env import load_public_env
from harnyx_miner.platform_monitoring import PlatformMonitoringClient

CACHE_DIR = Path(".autoresearch")
BATCH_ID_PATH = CACHE_DIR / "batch_id"
BENCHMARK_PATH = CACHE_DIR / "benchmark.json"
REPORTS_DIR = CACHE_DIR / "reports"
RESULTS_HEADER = "commit\tscore_a\tscore_b\tcost_usd\tstatus\tdescription"
DEFAULT_BENCHMARK_SAMPLE_SIZE = 20


@dataclass(frozen=True, slots=True)
class BenchmarkPin:
    suite_slug: str
    dataset_version: str
    scoring_version: str
    sample_size: int


@dataclass(frozen=True, slots=True)
class LocalEvalSummary:
    batch_id: str
    score_a: float
    champion_score: float | None
    delta_vs_champion: float | None
    total_seconds: float
    cost_usd: float
    error_count: int
    json_report: Path


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    suite_slug: str
    dataset_version: str
    scoring_version: str
    source_batch_id: str
    score_b: float
    total_seconds: float
    cost_usd: float
    error_count: int
    json_report: Path


def prepare(
    batch_id: str | None = None,
    *,
    benchmark_suite: str,
    benchmark_sample_size: int = DEFAULT_BENCHMARK_SAMPLE_SIZE,
) -> str:
    """Pin the local-eval batch and local benchmark snapshot used by this autoresearch run."""
    load_public_env()
    resolved_batch_id = _normalize_batch_id(batch_id) if batch_id else _resolve_latest_completed_batch_id()
    benchmark = _resolve_benchmark(benchmark_suite=benchmark_suite, sample_size=benchmark_sample_size)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_ID_PATH.write_text(f"{resolved_batch_id}\n", encoding="utf-8")
    BENCHMARK_PATH.write_text(
        json.dumps(
            {
                "suite_slug": benchmark.suite_slug,
                "dataset_version": benchmark.dataset_version,
                "scoring_version": benchmark.scoring_version,
                "sample_size": benchmark.sample_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"batch_id: {resolved_batch_id}")
    print(
        "benchmark: "
        f"{benchmark.suite_slug} "
        f"dataset_version={benchmark.dataset_version} "
        f"scoring_version={benchmark.scoring_version} "
        f"sample_size={benchmark.sample_size}"
    )
    print(f"cache_dir: {CACHE_DIR}")
    print("next: uv run train.py")
    return resolved_batch_id


def run_experiment(agent_path: Path = Path("train.py"), *, mode: str = "vs-champion") -> int:
    """Run the fixed evaluators for the editable candidate and print a stable summary."""
    load_public_env()
    batch_id = _read_pinned_batch_id()
    benchmark = _read_pinned_benchmark()
    output_dir = REPORTS_DIR / time.strftime("%Y%m%d-%H%M%S")
    local_eval_dir = output_dir / "local-eval"
    benchmark_dir = output_dir / "benchmark"
    local_eval_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    local_eval_command = build_local_eval_command(
        agent_path=agent_path,
        batch_id=batch_id,
        output_dir=local_eval_dir,
        mode=mode,
    )
    local_eval_started = time.monotonic()
    result = subprocess.run(  # noqa: S603 - fixed local-eval command assembled by this support script
        local_eval_command,
        check=False,
        capture_output=True,
        text=True,
    )
    local_eval_elapsed_seconds = time.monotonic() - local_eval_started
    (output_dir / "local-eval.stdout").write_text(result.stdout, encoding="utf-8")
    (output_dir / "local-eval.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    local_eval_report_path = _report_path_from_stdout(
        result.stdout,
        output_dir=local_eval_dir,
        report_glob="local-eval-report-*-*.json",
    )
    local_eval_summary = parse_report_summary(
        local_eval_report_path,
        elapsed_seconds=local_eval_elapsed_seconds,
    )

    benchmark_command = build_local_benchmark_command(
        agent_path=agent_path,
        batch_id=batch_id,
        benchmark=benchmark,
        output_dir=benchmark_dir,
    )
    benchmark_started = time.monotonic()
    benchmark_result = subprocess.run(  # noqa: S603 - fixed benchmark command assembled by this support script
        benchmark_command,
        check=False,
        capture_output=True,
        text=True,
    )
    benchmark_elapsed_seconds = time.monotonic() - benchmark_started
    (output_dir / "local-benchmark.stdout").write_text(benchmark_result.stdout, encoding="utf-8")
    (output_dir / "local-benchmark.stderr").write_text(benchmark_result.stderr, encoding="utf-8")
    if benchmark_result.returncode != 0:
        sys.stdout.write(benchmark_result.stdout)
        sys.stderr.write(benchmark_result.stderr)
        return benchmark_result.returncode
    benchmark_report_path = _report_path_from_stdout(
        benchmark_result.stdout,
        output_dir=benchmark_dir,
        report_glob="local-benchmark-report-*-*.json",
    )
    benchmark_summary = parse_benchmark_report_summary(
        benchmark_report_path,
        elapsed_seconds=benchmark_elapsed_seconds,
    )
    print_summary(local_eval_summary, benchmark_summary)
    return 0


def build_local_eval_command(
    *,
    agent_path: Path,
    batch_id: str,
    output_dir: Path,
    mode: str = "vs-champion",
) -> list[str]:
    return [
        "uv",
        "run",
        "harnyx-miner-local-eval",
        "--agent-path",
        str(agent_path),
        "--batch-id",
        batch_id,
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
    ]


def build_local_benchmark_command(
    *,
    agent_path: Path,
    batch_id: str,
    benchmark: BenchmarkPin,
    output_dir: Path,
) -> list[str]:
    return [
        "uv",
        "run",
        "harnyx-miner-local-benchmark",
        "--agent-path",
        str(agent_path),
        "--source-batch-id",
        batch_id,
        "--suite",
        benchmark.suite_slug,
        "--dataset-version",
        benchmark.dataset_version,
        "--scoring-version",
        benchmark.scoring_version,
        "--sample-size",
        str(benchmark.sample_size),
        "--output-dir",
        str(output_dir),
    ]


def parse_report_summary(report_path: Path, *, elapsed_seconds: float | None = None) -> LocalEvalSummary:
    report = _read_json_object(report_path)
    batch_id = _string(_mapping(report.get("identifiers"), "identifiers").get("batch_id"), "batch_id")
    leaderboard = _sequence(
        _mapping(report.get("local_result_summary"), "local_result_summary").get("leaderboard"),
        "leaderboard",
    )
    target = _leaderboard_entry(leaderboard, "target")
    champion = _optional_leaderboard_entry(leaderboard, "champion")
    score_a = _float(target.get("total_score"), "target total_score")
    champion_score = _float(champion.get("total_score"), "champion total_score") if champion else None
    delta = score_a - champion_score if champion_score is not None else None
    costs = _mapping(target.get("cost_totals"), "target cost_totals")
    total_seconds = elapsed_seconds if elapsed_seconds is not None else _target_elapsed_seconds(report)
    return LocalEvalSummary(
        batch_id=batch_id,
        score_a=score_a,
        champion_score=champion_score,
        delta_vs_champion=delta,
        total_seconds=total_seconds,
        cost_usd=_float(costs.get("total_cost_usd", 0.0), "target total_cost_usd"),
        error_count=_non_negative_int(target.get("error_count", 0), "target error_count"),
        json_report=report_path,
    )


def parse_benchmark_report_summary(
    report_path: Path,
    *,
    elapsed_seconds: float | None = None,
) -> BenchmarkSummary:
    report = _read_json_object(report_path)
    identifiers = _mapping(report.get("identifiers"), "identifiers")
    metadata = _mapping(report.get("benchmark_metadata"), "benchmark_metadata")
    manifest = _mapping(metadata.get("manifest"), "benchmark_metadata manifest")
    summary = _mapping(report.get("summary"), "summary")
    costs = _mapping(summary.get("cost_totals"), "benchmark cost_totals")
    source_batch_id = _string(identifiers.get("source_batch_id"), "source_batch_id")
    total_seconds = elapsed_seconds if elapsed_seconds is not None else _float(
        summary.get("total_seconds"),
        "benchmark total_seconds",
    )
    return BenchmarkSummary(
        suite_slug=_string(manifest.get("suite_slug"), "benchmark suite_slug"),
        dataset_version=_string(manifest.get("dataset_version"), "benchmark dataset_version"),
        scoring_version=_string(manifest.get("scoring_version"), "benchmark scoring_version"),
        source_batch_id=_normalize_batch_id(source_batch_id),
        score_b=_float(summary.get("mean_total_score"), "benchmark mean_total_score"),
        total_seconds=total_seconds,
        cost_usd=_float(costs.get("total_cost_usd", 0.0), "benchmark total_cost_usd"),
        error_count=_non_negative_int(summary.get("error_count", 0), "benchmark error_count"),
        json_report=report_path,
    )


def print_summary(local_eval: LocalEvalSummary, benchmark: BenchmarkSummary) -> None:
    total_seconds = local_eval.total_seconds + benchmark.total_seconds
    total_cost_usd = local_eval.cost_usd + benchmark.cost_usd
    total_error_count = local_eval.error_count + benchmark.error_count
    print("---")
    print(f"score_a:              {local_eval.score_a:.6f}")
    print(f"score_b:              {benchmark.score_b:.6f}")
    print(f"champion_score_a:     {_format_optional_float(local_eval.champion_score)}")
    print(f"delta_vs_champion_a:  {_format_optional_float(local_eval.delta_vs_champion)}")
    print(f"total_seconds:        {total_seconds:.1f}")
    print(f"cost_usd:             {total_cost_usd:.6f}")
    print(f"local_eval_cost_usd:  {local_eval.cost_usd:.6f}")
    print(f"benchmark_cost_usd:   {benchmark.cost_usd:.6f}")
    print(f"error_count:          {total_error_count}")
    print(f"batch_id:             {local_eval.batch_id}")
    print(
        "benchmark:            "
        f"{benchmark.suite_slug} "
        f"dataset_version={benchmark.dataset_version} "
        f"scoring_version={benchmark.scoring_version}"
    )
    print(f"local_eval_json_report:  {local_eval.json_report}")
    print(f"benchmark_json_report:   {benchmark.json_report}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a Harnyx miner autoresearch run.")
    parser.add_argument("--batch-id", help="Specific completed batch id to pin. Defaults to latest completed batch.")
    parser.add_argument(
        "--benchmark-suite",
        required=True,
        help="Benchmark suite slug to pin for local benchmark runs, for example deepsearchqa or deepresearch9k-l1.",
    )
    parser.add_argument(
        "--benchmark-sample-size",
        type=int,
        default=DEFAULT_BENCHMARK_SAMPLE_SIZE,
        help=f"Number of benchmark items to pin. Default: {DEFAULT_BENCHMARK_SAMPLE_SIZE}.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    prepare(
        batch_id=args.batch_id,
        benchmark_suite=args.benchmark_suite,
        benchmark_sample_size=args.benchmark_sample_size,
    )
    return 0


def _resolve_latest_completed_batch_id() -> str:
    client = PlatformMonitoringClient.from_env()
    try:
        return str(client.resolve_batch_context(None).batch_id)
    finally:
        client.close()


def _resolve_benchmark(*, benchmark_suite: str, sample_size: int) -> BenchmarkPin:
    if sample_size <= 0:
        raise ValueError("benchmark sample size must be positive")

    snapshot = load_current_benchmark_snapshot(benchmark_suite)
    return BenchmarkPin(
        suite_slug=snapshot.manifest.suite_slug,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=sample_size,
    )


def _normalize_batch_id(raw: str) -> str:
    return str(UUID(raw))


def _read_pinned_batch_id() -> str:
    if not BATCH_ID_PATH.exists():
        raise RuntimeError(
            "missing .autoresearch/batch_id; run `uv run prepare.py --benchmark-suite <suite-slug>` first"
        )
    return _normalize_batch_id(BATCH_ID_PATH.read_text(encoding="utf-8").strip())


def _read_pinned_benchmark() -> BenchmarkPin:
    if not BENCHMARK_PATH.exists():
        raise RuntimeError(
            "missing .autoresearch/benchmark.json; run `uv run prepare.py --benchmark-suite <suite-slug>` first"
        )
    payload = _read_json_object(BENCHMARK_PATH)
    return BenchmarkPin(
        suite_slug=_string(payload.get("suite_slug"), "benchmark suite_slug"),
        dataset_version=_string(payload.get("dataset_version"), "benchmark dataset_version"),
        scoring_version=_string(payload.get("scoring_version"), "benchmark scoring_version"),
        sample_size=_positive_int(payload.get("sample_size"), "benchmark sample_size"),
    )


def _report_path_from_stdout(stdout: str, *, output_dir: Path, report_glob: str) -> Path:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and isinstance(payload.get("json_report"), str):
            return Path(payload["json_report"])
    candidates = sorted(output_dir.glob(report_glob))
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError("evaluator did not report exactly one structured JSON report")


def _read_json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, f"report {path}")


def _leaderboard_entry(entries: Sequence[object], label: str) -> Mapping[str, object]:
    entry = _optional_leaderboard_entry(entries, label)
    if entry is None:
        raise RuntimeError(f"local eval report is missing {label} leaderboard entry")
    return entry


def _optional_leaderboard_entry(entries: Sequence[object], label: str) -> Mapping[str, object] | None:
    for raw_entry in entries:
        entry = _mapping(raw_entry, "leaderboard entry")
        if entry.get("label") == label:
            return entry
    return None


def _target_elapsed_seconds(report: Mapping[str, object]) -> float:
    total_ms = 0.0
    for raw_task in _sequence(report.get("tasks", ()), "tasks"):
        task = _mapping(raw_task, "task")
        target = task.get("target")
        if target is None:
            continue
        elapsed_ms = _mapping(target, "target").get("elapsed_ms")
        if elapsed_ms is not None:
            total_ms += _float(elapsed_ms, "target elapsed_ms")
    return total_ms / 1000


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimeError(f"{label} must be a finite number")
    return parsed


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def _non_negative_int(value: object, label: str) -> int:
    parsed = _int(value, label)
    if parsed < 0:
        raise RuntimeError(f"{label} must be non-negative")
    return parsed


def _positive_int(value: object, label: str) -> int:
    parsed = _int(value, label)
    if parsed <= 0:
        raise RuntimeError(f"{label} must be positive")
    return parsed


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
