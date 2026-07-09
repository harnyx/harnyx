from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.benchmark_rubric_judge import BenchmarkRubricJudgeLlmSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.domain.miner_task import (
    DEFAULT_MINER_TASK_BUDGET_USD,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.json_types import JsonObject
from harnyx_commons.llm.provider_factory import (
    build_cached_llm_provider_registry,
)
from harnyx_commons.miner_task_benchmark import (
    BENCHMARK_CORRECTNESS_SCORING_VERSION,
    BENCHMARK_SAMPLE_SIZE,
    BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    BenchmarkCorrectnessScoringConfig,
    BenchmarkCorrectnessScoringService,
    BenchmarkDatasetItem,
    BenchmarkDatasetSnapshot,
    BenchmarkItemOutcome,
    BenchmarkItemState,
    BenchmarkRunMetrics,
    BenchmarkWeightedRubricScoringConfig,
    BenchmarkWeightedRubricScoringService,
    aggregate_benchmark_metrics,
    benchmark_backing_batch_id_for_run,
    benchmark_run_id_for_source_batch,
    benchmark_task_id_for_item,
    is_supported_benchmark_scoring_version,
    list_current_benchmark_snapshots,
    load_benchmark_snapshot,
    sample_benchmark_items,
    unsupported_benchmark_scoring_version_error,
)
from harnyx_commons.miner_task_scoring import EvaluationScoringConfig, EvaluationScoringService
from harnyx_miner.agent_source import (
    agent_sha256,
    load_agent_bytes,
    require_existing_agent_path,
)
from harnyx_miner.env import load_public_env
from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission, ScriptArtifactSpec
from harnyx_validator.version import VALIDATOR_RELEASE_VERSION

if TYPE_CHECKING:
    from harnyx_miner.local_eval import LocalEvaluationRuntime

_DEFAULT_OUTPUT_PREFIX = "local-benchmark-report"
_DEFAULT_SAMPLE_SIZE = BENCHMARK_SAMPLE_SIZE
_DEFAULT_PARALLELISM = 1
_LOCAL_BENCHMARK_UID = 1
_INVOCATION_ONLY_SCORING_CONFIG = EvaluationScoringConfig(
    provider="chutes",
    model="benchmark-invocation-only",
    max_output_tokens=0,
    scoring_version="benchmark-invocation-only",
)


def _emit_progress(message: str) -> None:
    print(f"[local-benchmark] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True, slots=True)
class _BenchmarkItemScore:
    is_correct: bool | None
    score: float | None
    score_reason: str | None
    score_detail: JsonObject | None


class _LocalBenchmarkScoringService(Protocol):
    async def score(
        self,
        *,
        item: BenchmarkDatasetItem,
        generated_answer: str,
    ) -> _BenchmarkItemScore: ...


class _BenchmarkScoringRegistry(Protocol):
    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _CorrectnessLocalBenchmarkScoringService:
    delegate: BenchmarkCorrectnessScoringService

    async def score(
        self,
        *,
        item: BenchmarkDatasetItem,
        generated_answer: str,
    ) -> _BenchmarkItemScore:
        score = await self.delegate.score(
            question=item.problem,
            reference_answer=item.answer,
            generated_answer=generated_answer,
        )
        return _BenchmarkItemScore(
            is_correct=score.is_correct,
            score=1.0 if score.is_correct else 0.0,
            score_reason=score.reason,
            score_detail=None,
        )


@dataclass(frozen=True, slots=True)
class _WeightedRubricLocalBenchmarkScoringService:
    delegate: BenchmarkWeightedRubricScoringService

    async def score(
        self,
        *,
        item: BenchmarkDatasetItem,
        generated_answer: str,
    ) -> _BenchmarkItemScore:
        score = await self.delegate.score(
            question=item.problem,
            rubric_answer=item.answer,
            generated_answer=generated_answer,
        )
        return _BenchmarkItemScore(
            is_correct=None,
            score=score.normalized_score,
            score_reason=None,
            score_detail=score.score_detail,
        )


@dataclass(frozen=True, slots=True)
class _BenchmarkScoringBundle:
    service: _LocalBenchmarkScoringService
    registry: _BenchmarkScoringRegistry
    config: BenchmarkCorrectnessScoringConfig | BenchmarkWeightedRubricScoringConfig
    scoring_boundary: str
    scoring_method: str
    uses_numeric_scores: bool
    failed_item_report_score: float | None

    async def aclose(self) -> None:
        await self.registry.aclose()


@dataclass(frozen=True, slots=True)
class _BenchmarkItemResult:
    item: BenchmarkDatasetItem
    task: MinerTask
    submission: MinerTaskRunSubmission | None
    score: _BenchmarkItemScore | None
    error_code: str | None
    error_message: str | None

    @property
    def is_correct(self) -> bool | None:
        if self.score is None:
            return None
        return self.score.is_correct

    @property
    def state(self) -> BenchmarkItemState:
        return BenchmarkItemState.COMPLETED if self.score is not None else BenchmarkItemState.FAILED


class _InvocationOnlyScoringService(EvaluationScoringService):
    def __init__(self) -> None:
        self._config = _INVOCATION_ONLY_SCORING_CONFIG

    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        del task, response
        return ScoreBreakdown(
            comparison_score=0.0,
            total_score=0.0,
            scoring_version="benchmark-invocation-only",
        )


def _create_invocation_only_runtime(
    *,
    scoring_service: EvaluationScoringService,
    scoring_config: EvaluationScoringConfig,
    run_progress_root: Path,
) -> LocalEvaluationRuntime:
    from harnyx_miner.local_eval import LocalEvaluationRuntime

    return LocalEvaluationRuntime.create_invocation_only(
        scoring_service=scoring_service,
        scoring_config=scoring_config,
        run_progress_root=run_progress_root,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local miner artifact against an explicit benchmark suite.",
    )
    parser.add_argument("--agent-path", help="Path to the local miner agent file.")
    parser.add_argument(
        "--source-batch-id",
        help="Pinned source batch id used to derive the deterministic benchmark sample.",
    )
    parser.add_argument(
        "--suite",
        help=(
            "Benchmark suite slug. Required unless --list-suites is used, "
            "for example draco, webwalkerqa, deepsearchqa, or deepresearch9k-l1."
        ),
    )
    parser.add_argument("--dataset-version", help="Pinned benchmark dataset version.")
    parser.add_argument("--scoring-version", help="Pinned benchmark scoring version.")
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List current benchmark suites and exit.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help=f"Number of benchmark items to run. Default: {_DEFAULT_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=_DEFAULT_PARALLELISM,
        help=f"Concurrent benchmark item sandboxes. Default: {_DEFAULT_PARALLELISM}.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd()),
        help="Directory for the JSON report. Default: current working directory.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


async def _amain(argv: Sequence[str] | None) -> None:
    args = _parse_args(argv)
    load_public_env()
    if args.list_suites:
        _print_current_suites()
        return
    if not args.agent_path:
        raise ValueError("--agent-path is required")
    if not args.source_batch_id:
        raise ValueError("--source-batch-id is required")
    if not args.suite:
        raise ValueError("--suite is required")
    target_path = require_existing_agent_path(args.agent_path)
    source_batch_id = UUID(args.source_batch_id)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.parallelism <= 0:
        raise ValueError("--parallelism must be positive")

    snapshot = load_benchmark_snapshot(
        args.suite,
        dataset_version=args.dataset_version,
        scoring_version=args.scoring_version,
    )
    if not is_supported_benchmark_scoring_version(snapshot.manifest.scoring_version):
        raise unsupported_benchmark_scoring_version_error(snapshot.manifest.scoring_version)
    run_id = benchmark_run_id_for_source_batch(
        suite_slug=snapshot.manifest.suite_slug,
        source_batch_id=source_batch_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
    )
    run_progress_root = output_dir / ".harnyx-local-benchmark-progress" / str(run_id)
    backing_batch_id = benchmark_backing_batch_id_for_run(
        suite_slug=snapshot.manifest.suite_slug,
        run_id=run_id,
    )
    sampled_items = sample_benchmark_items(
        items=snapshot.items,
        run_id=run_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
        sample_size=args.sample_size,
    )
    tasks = _build_tasks(run_id=run_id, snapshot=snapshot, items=sampled_items)
    target_bytes = load_agent_bytes(target_path)
    target_artifact = _build_target_artifact_spec(
        run_id=run_id,
        target_bytes=target_bytes,
    )

    _emit_progress(
        "selected benchmark: "
        f"suite={snapshot.manifest.suite_slug} "
        f"dataset_version={snapshot.manifest.dataset_version} "
        f"scoring_version={snapshot.manifest.scoring_version} "
        f"items={len(sampled_items)}"
    )
    _emit_progress(
        "loaded target artifact: "
        f"path={target_path} size_bytes={len(target_bytes)} sha256={target_artifact.content_hash}"
    )

    runtime: LocalEvaluationRuntime | None = None
    scoring: _BenchmarkScoringBundle | None = None
    started = time.monotonic()
    try:
        invocation_scoring = _InvocationOnlyScoringService()
        runtime = _create_invocation_only_runtime(
            scoring_service=invocation_scoring,
            scoring_config=_INVOCATION_ONLY_SCORING_CONFIG,
            run_progress_root=run_progress_root,
        )
        scoring = _build_benchmark_scoring_bundle(snapshot.manifest.scoring_version)
        _emit_progress(
            "running candidate against benchmark items: "
            f"parallelism={args.parallelism}"
        )
        results = await _evaluate_and_score_items(
            runtime=runtime,
            target_bytes=target_bytes,
            target_artifact=target_artifact,
            backing_batch_id=backing_batch_id,
            items=sampled_items,
            tasks=tasks,
            invocation_scoring=invocation_scoring,
            scoring_service=scoring.service,
            parallelism=args.parallelism,
        )
    finally:
        if runtime is not None:
            await runtime.aclose()
        if scoring is not None:
            await scoring.aclose()

    elapsed_seconds = time.monotonic() - started
    if scoring is None:
        raise RuntimeError("benchmark scoring was not initialized")
    report = _build_report(
        snapshot=snapshot,
        source_batch_id=source_batch_id,
        run_id=run_id,
        backing_batch_id=backing_batch_id,
        target_path=target_path,
        target_bytes=target_bytes,
        target_artifact=target_artifact,
        results=results,
        scoring=scoring,
        output_dir=output_dir,
        elapsed_seconds=elapsed_seconds,
        parallelism=args.parallelism,
    )
    json_path = _write_report(
        report=report,
        output_dir=output_dir,
        source_batch_id=source_batch_id,
        snapshot=snapshot,
    )
    _emit_progress(f"report written: json={json_path}")
    summary = _require_mapping(report["summary"], label="summary")
    print(
        json.dumps(
            {
                "suite_slug": snapshot.manifest.suite_slug,
                "dataset_version": snapshot.manifest.dataset_version,
                "scoring_version": snapshot.manifest.scoring_version,
                "source_batch_id": str(source_batch_id),
                "run_id": str(run_id),
                "json_report": str(json_path),
                "mean_total_score": summary["mean_total_score"],
                "item_count": summary["item_count"],
                "error_count": summary["error_count"],
                "total_cost_usd": _require_mapping(summary["cost_totals"], label="summary cost_totals")[
                    "total_cost_usd"
                ],
            },
            sort_keys=True,
        )
    )


def _build_tasks(
    *,
    run_id: UUID,
    snapshot: BenchmarkDatasetSnapshot,
    items: Sequence[BenchmarkDatasetItem],
) -> tuple[MinerTask, ...]:
    return tuple(
        MinerTask(
            task_id=benchmark_task_id_for_item(
                suite_slug=snapshot.manifest.suite_slug,
                run_id=run_id,
                item_index=item.item_index,
            ),
            query=Query(text=item.problem),
            reference_answer=ReferenceAnswer(text=item.answer),
            budget_usd=DEFAULT_MINER_TASK_BUDGET_USD,
        )
        for item in items
    )


def _build_target_artifact_spec(
    *,
    run_id: UUID,
    target_bytes: bytes,
) -> ScriptArtifactSpec:
    content_hash = agent_sha256(target_bytes)
    return ScriptArtifactSpec(
        uid=_LOCAL_BENCHMARK_UID,
        artifact_id=uuid5(NAMESPACE_URL, f"harnyx-local-benchmark:{run_id}:{content_hash}"),
        content_hash=content_hash,
        size_bytes=len(target_bytes),
    )


def _build_benchmark_scoring_bundle(scoring_version: str) -> _BenchmarkScoringBundle:
    if scoring_version == BENCHMARK_CORRECTNESS_SCORING_VERSION:
        return _build_correctness_scoring_bundle()
    if scoring_version == BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION:
        return _build_weighted_rubric_scoring_bundle()
    raise unsupported_benchmark_scoring_version_error(scoring_version)


def _build_correctness_scoring_bundle() -> _BenchmarkScoringBundle:
    load_public_env()
    llm_settings = LlmSettings()
    model = llm_settings.benchmark_llm_model.strip()
    if not model:
        raise RuntimeError("BENCHMARK_LLM_MODEL must be configured to run local benchmark scoring")
    registry = build_cached_llm_provider_registry(
        llm_settings=llm_settings,
        bedrock_settings=BedrockSettings(),
        vertex_settings=VertexSettings(),
    )
    provider_name = llm_settings.benchmark_llm_provider
    provider = registry.resolve(provider_name)
    config = BenchmarkCorrectnessScoringConfig(
        provider=provider_name,
        model=model,
        temperature=llm_settings.benchmark_llm_temperature,
        max_output_tokens=llm_settings.benchmark_llm_max_output_tokens,
        reasoning_effort=llm_settings.benchmark_llm_reasoning_effort,
        timeout_seconds=llm_settings.benchmark_llm_timeout_seconds or llm_settings.llm_timeout_seconds,
    )
    return _BenchmarkScoringBundle(
        service=_CorrectnessLocalBenchmarkScoringService(
            BenchmarkCorrectnessScoringService(
                llm_provider=provider,
                config=config,
            )
        ),
        registry=registry,
        config=config,
        scoring_boundary="benchmark-correctness-judge",
        scoring_method="binary correctness against the canonical benchmark answer",
        uses_numeric_scores=False,
        failed_item_report_score=0.0,
    )


def _build_weighted_rubric_scoring_bundle() -> _BenchmarkScoringBundle:
    load_public_env()
    llm_settings = LlmSettings()
    rubric_settings = BenchmarkRubricJudgeLlmSettings()
    provider_name = rubric_settings.provider
    model = rubric_settings.model.strip()
    if provider_name is None or not model:
        raise RuntimeError(
            "BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER and BENCHMARK_RUBRIC_JUDGE_LLM_MODEL "
            "must be configured to run weighted rubric local benchmark scoring"
        )
    registry = build_cached_llm_provider_registry(
        llm_settings=llm_settings,
        bedrock_settings=BedrockSettings(),
        vertex_settings=VertexSettings(),
    )
    provider = registry.resolve(provider_name)
    config = BenchmarkWeightedRubricScoringConfig(
        provider=provider_name,
        model=model,
        temperature=rubric_settings.temperature,
        max_output_tokens=None,
        reasoning_effort=rubric_settings.reasoning_effort,
        timeout_seconds=rubric_settings.timeout_seconds or llm_settings.llm_timeout_seconds,
    )
    return _BenchmarkScoringBundle(
        service=_WeightedRubricLocalBenchmarkScoringService(
            BenchmarkWeightedRubricScoringService(
                llm_provider=provider,
                config=config,
            )
        ),
        registry=registry,
        config=config,
        scoring_boundary="benchmark-weighted-rubric-judge",
        scoring_method="weighted rubric criteria judged independently",
        uses_numeric_scores=True,
        failed_item_report_score=None,
    )


def _print_current_suites() -> None:
    suites = [
        {
            "suite_slug": snapshot.manifest.suite_slug,
            "suite_name": snapshot.manifest.suite_name,
            "dataset_version": snapshot.manifest.dataset_version,
            "scoring_version": snapshot.manifest.scoring_version,
            "row_count": snapshot.manifest.row_count,
        }
        for snapshot in list_current_benchmark_snapshots()
    ]
    print(json.dumps({"suites": suites}, sort_keys=True))


async def _evaluate_and_score_items(
    *,
    runtime: LocalEvaluationRuntime,
    target_bytes: bytes,
    target_artifact: ScriptArtifactSpec,
    backing_batch_id: UUID,
    items: Sequence[BenchmarkDatasetItem],
    tasks: Sequence[MinerTask],
    invocation_scoring: EvaluationScoringService,
    scoring_service: _LocalBenchmarkScoringService,
    parallelism: int,
) -> tuple[_BenchmarkItemResult, ...]:
    semaphore = asyncio.Semaphore(parallelism)

    async def _run_item(item: BenchmarkDatasetItem, task: MinerTask) -> _BenchmarkItemResult:
        async with semaphore:
            return await _evaluate_and_score_item(
                runtime=runtime,
                target_bytes=target_bytes,
                target_artifact=target_artifact,
                backing_batch_id=backing_batch_id,
                item=item,
                task=task,
                invocation_scoring=invocation_scoring,
                scoring_service=scoring_service,
            )

    return tuple(
        await asyncio.gather(
            *(_run_item(item, task) for item, task in zip(items, tasks, strict=True))
        )
    )


async def _evaluate_and_score_item(
    *,
    runtime: LocalEvaluationRuntime,
    target_bytes: bytes,
    target_artifact: ScriptArtifactSpec,
    backing_batch_id: UUID,
    item: BenchmarkDatasetItem,
    task: MinerTask,
    invocation_scoring: EvaluationScoringService,
    scoring_service: _LocalBenchmarkScoringService,
) -> _BenchmarkItemResult:
    try:
        outcome = await runtime.evaluate_artifact(
            artifact_label=f"benchmark-target-{item.item_index}",
            agent_source=target_bytes,
            artifact=target_artifact,
            batch_id=backing_batch_id,
            tasks=(task,),
            scoring_service=invocation_scoring,
        )
    except Exception as exc:
        return _BenchmarkItemResult(
            item=item,
            task=task,
            submission=None,
            score=None,
            error_code="runner_exception",
            error_message=str(exc),
        )
    if outcome.artifact_failure is not None:
        failure = outcome.artifact_failure
        return _BenchmarkItemResult(
            item=item,
            task=task,
            submission=None,
            score=None,
            error_code=str(failure.error_code),
            error_message=failure.message,
        )
    submission_by_task = {submission.run.task_id: submission for submission in outcome.submissions}
    return await _score_item_submission(
        item=item,
        task=task,
        submission=submission_by_task.get(task.task_id),
        scoring_service=scoring_service,
    )


async def _score_item_submission(
    *,
    item: BenchmarkDatasetItem,
    task: MinerTask,
    submission: MinerTaskRunSubmission | None,
    scoring_service: _LocalBenchmarkScoringService,
) -> _BenchmarkItemResult:
    if submission is None:
        return _BenchmarkItemResult(
            item=item,
            task=task,
            submission=None,
            score=None,
            error_code="missing_submission",
            error_message="candidate did not produce a benchmark submission",
        )
    run_error = submission.run.details.error
    if run_error is not None:
        return _BenchmarkItemResult(
            item=item,
            task=task,
            submission=submission,
            score=None,
            error_code=str(run_error.code),
            error_message=run_error.message,
        )
    response = submission.run.response
    if response is None:
        return _BenchmarkItemResult(
            item=item,
            task=task,
            submission=submission,
            score=None,
            error_code="missing_response",
            error_message="candidate run completed without a response",
        )
    try:
        score = await scoring_service.score(
            item=item,
            generated_answer=response.text,
        )
    except Exception as exc:
        return _BenchmarkItemResult(
            item=item,
            task=task,
            submission=submission,
            score=None,
            error_code="benchmark_scoring_failed",
            error_message=str(exc),
        )
    return _BenchmarkItemResult(
        item=item,
        task=task,
        submission=submission,
        score=score,
        error_code=None,
        error_message=None,
    )


def _build_report(
    *,
    snapshot: BenchmarkDatasetSnapshot,
    source_batch_id: UUID,
    run_id: UUID,
    backing_batch_id: UUID,
    target_path: Path,
    target_bytes: bytes,
    target_artifact: ScriptArtifactSpec,
    results: Sequence[_BenchmarkItemResult],
    scoring: _BenchmarkScoringBundle,
    output_dir: Path,
    elapsed_seconds: float,
    parallelism: int,
) -> dict[str, object]:
    metrics = _aggregate_local_report_metrics(results=results, scoring=scoring)
    cost_totals = _aggregate_cost_totals(
        tuple(result.submission for result in results if result.submission is not None)
    )
    return {
        "benchmark_metadata": {
            "manifest": {
                "suite_slug": snapshot.manifest.suite_slug,
                "suite_name": snapshot.manifest.suite_name,
                "dataset_version": snapshot.manifest.dataset_version,
                "scoring_version": snapshot.manifest.scoring_version,
                "source_url": snapshot.manifest.source_url,
                "source_page_url": snapshot.manifest.source_page_url,
                "license": snapshot.manifest.license,
                "sha256": snapshot.manifest.sha256,
                "row_count": snapshot.manifest.row_count,
                "file_name": snapshot.manifest.file_name,
                "fetched_at": snapshot.manifest.fetched_at,
            },
        },
        "identifiers": {
            "source_batch_id": str(source_batch_id),
            "run_id": str(run_id),
            "backing_batch_id": str(backing_batch_id),
            "target_artifact_id": str(target_artifact.artifact_id),
            "target_uid": target_artifact.uid,
        },
        "evaluation_config": {
            "agent_path": str(target_path),
            "output_dir": str(output_dir),
            "validator_version": VALIDATOR_RELEASE_VERSION,
            "sample_size": len(results),
            "execution_boundary": "docker-sandbox",
            "sandbox_item_parallelism": 1,
            "benchmark_item_parallelism": parallelism,
            "scoring_boundary": scoring.scoring_boundary,
        },
        "scoring_context": {
            "provider": scoring.config.provider,
            "model": scoring.config.model,
            "temperature": scoring.config.temperature,
            "max_output_tokens": scoring.config.max_output_tokens,
            "reasoning_effort": scoring.config.reasoning_effort,
            "timeout_seconds": scoring.config.timeout_seconds,
            "scoring_version": snapshot.manifest.scoring_version,
            "method": scoring.scoring_method,
        },
        "artifacts": {
            "target": {
                "artifact_id": str(target_artifact.artifact_id),
                "uid": target_artifact.uid,
                "content_hash": target_artifact.content_hash,
                "size_bytes": target_artifact.size_bytes,
                "path": str(target_path),
                "sha256": agent_sha256(target_bytes),
            },
        },
        "summary": {
            "item_count": len(results),
            "completed_item_count": metrics.completed_item_count,
            "failed_item_count": metrics.failed_item_count,
            "correct_item_count": metrics.correct_item_count,
            "mean_total_score": metrics.mean_total_score,
            "error_count": metrics.failed_item_count,
            "total_seconds": round(elapsed_seconds, 3),
            "cost_totals": cost_totals,
        },
        "items": [_item_report(result, scoring=scoring) for result in results],
    }


def _aggregate_local_report_metrics(
    *,
    results: Sequence[_BenchmarkItemResult],
    scoring: _BenchmarkScoringBundle,
) -> BenchmarkRunMetrics:
    metrics = aggregate_benchmark_metrics(tuple(_item_outcome(result) for result in results))
    if not scoring.uses_numeric_scores:
        return metrics
    return replace(metrics, correct_item_count=None)


def _item_outcome(result: _BenchmarkItemResult) -> BenchmarkItemOutcome:
    numeric_score = None
    if result.score is not None and result.score.is_correct is None:
        numeric_score = result.score.score
    return BenchmarkItemOutcome(
        state=result.state,
        is_correct=result.is_correct,
        score=numeric_score,
    )


def _item_report(result: _BenchmarkItemResult, *, scoring: _BenchmarkScoringBundle) -> dict[str, object]:
    return {
        "item_index": result.item.item_index,
        "task_id": str(result.task.task_id),
        "problem": result.item.problem,
        "problem_category": result.item.problem_category,
        "reference_answer": result.item.answer,
        "answer_type": result.item.answer_type.value,
        "generated_answer": (
            _serialize_answer(result.submission.run.response)
            if result.submission is not None and result.submission.run.response is not None
            else None
        ),
        "is_correct": result.score.is_correct if result.score is not None else None,
        "score": result.score.score if result.score is not None else scoring.failed_item_report_score,
        "score_reason": result.score.score_reason if result.score is not None else None,
        "score_detail": result.score.score_detail if result.score is not None else None,
        "invocation": _submission_detail(result.submission),
        "error": (
            {
                "code": result.error_code,
                "message": result.error_message,
            }
            if result.error_code is not None
            else None
        ),
    }


def _submission_detail(submission: MinerTaskRunSubmission | None) -> dict[str, object] | None:
    if submission is None:
        return None
    run = submission.run
    return {
        "artifact_id": str(run.artifact_id),
        "uid": run.uid,
        "score": submission.score,
        "elapsed_ms": run.details.elapsed_ms,
        "attempt_count": submission.session.active_attempt,
        "session_status": submission.session.status.value,
        "error": run.details.error.model_dump(mode="json") if run.details.error is not None else None,
        "cost_totals": _cost_totals_from_submission(submission),
        "token_usage": submission.usage.model_dump(mode="json"),
    }


def _serialize_answer(answer: object) -> dict[str, object]:
    model_dump = getattr(answer, "model_dump", None)
    if not callable(model_dump):
        raise RuntimeError("answer payload must support model_dump()")
    return model_dump(mode="json", exclude_none=True)


def _aggregate_cost_totals(submissions: Sequence[MinerTaskRunSubmission]) -> dict[str, object]:
    total_llm_cost = 0.0
    total_search_cost = 0.0
    total_embedding_cost = 0.0
    total_reference_cost = 0.0
    total_llm_tokens = 0
    total_llm_calls = 0
    total_search_calls = 0
    total_embedding_calls = 0
    for submission in submissions:
        details = submission.run.details.total_tool_usage
        total_llm_cost += details.llm_cost
        total_search_cost += details.search_tool_cost
        total_embedding_cost += details.embedding_cost
        total_reference_cost += details.reference_total_cost_usd
        total_llm_tokens += submission.usage.total_tokens
        total_llm_calls += submission.usage.call_count
        total_search_calls += details.search_tool.call_count
        total_embedding_calls += details.embedding.call_count
    return {
        "llm_cost_usd": round(total_llm_cost, 6),
        "search_tool_cost_usd": round(total_search_cost, 6),
        "embedding_cost_usd": round(total_embedding_cost, 6),
        "total_cost_usd": round(total_reference_cost, 6),
        "llm_total_tokens": total_llm_tokens,
        "llm_call_count": total_llm_calls,
        "search_tool_call_count": total_search_calls,
        "embedding_call_count": total_embedding_calls,
    }


def _cost_totals_from_submission(submission: MinerTaskRunSubmission) -> dict[str, object]:
    return _aggregate_cost_totals((submission,))


def _write_report(
    *,
    report: Mapping[str, object],
    output_dir: Path,
    source_batch_id: UUID,
    snapshot: BenchmarkDatasetSnapshot,
) -> Path:
    json_path = output_dir / (
        f"{_DEFAULT_OUTPUT_PREFIX}-{source_batch_id}-"
        f"{snapshot.manifest.dataset_version}-{snapshot.manifest.scoring_version}.json"
    )
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return json_path


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


async def _run(argv: Sequence[str] | None) -> int:
    await _amain(argv)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"local benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
