from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from harnyx_commons import miner_task_benchmark
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    MinerTask,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_usage import (
    EmbeddingToolUsageSummary,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from harnyx_commons.miner_task_benchmark import (
    BENCHMARK_CORRECTNESS_SCORING_VERSION,
    BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
    BenchmarkAnswerType,
    BenchmarkDatasetItem,
    BenchmarkDatasetManifest,
    BenchmarkDatasetSnapshot,
    benchmark_backing_batch_id_for_run,
    benchmark_run_id_for_source_batch,
    benchmark_task_id_for_item,
    load_current_benchmark_snapshot,
)
from harnyx_miner import local_benchmark
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.application.services.evaluation_runner import ArtifactEvaluationOutcome
from harnyx_validator.domain.evaluation import MinerTaskRun


def _snapshot() -> BenchmarkDatasetSnapshot:
    return BenchmarkDatasetSnapshot(
        manifest=BenchmarkDatasetManifest(
            suite_slug="deepsearchqa",
            suite_name="DeepSearchQA",
            dataset_version="2026-04-30",
            scoring_version="correctness-v1",
            source_url="https://example.com/deepsearchqa.csv",
            source_page_url="https://example.com/deepsearchqa",
            license="open",
            sha256="0" * 64,
            row_count=2,
            file_name="data.csv",
            fetched_at="2026-04-30T00:00:00Z",
        ),
        items=(
            BenchmarkDatasetItem(
                item_index=0,
                problem="Who wrote the benchmark?",
                problem_category="factoid",
                answer="The benchmark authors.",
                answer_type=BenchmarkAnswerType.SINGLE_ANSWER,
            ),
            BenchmarkDatasetItem(
                item_index=1,
                problem="Name the set.",
                problem_category="set",
                answer="Alpha, Beta",
                answer_type=BenchmarkAnswerType.SET_ANSWER,
            ),
        ),
    )


def _weighted_rubric_answer() -> str:
    return json.dumps(
        {
            "id": "rubric-1",
            "sections": [
                {
                    "id": "evidence",
                    "title": "Evidence",
                    "criteria": [
                        {
                            "id": "cites-supported-answer",
                            "weight": 2.0,
                            "requirement": "The response answers with support from the provided evidence.",
                        },
                        {
                            "id": "unsupported-claim",
                            "weight": -1.0,
                            "requirement": "The response includes an unsupported claim.",
                        },
                    ],
                }
            ],
        },
        sort_keys=True,
    )


def _weighted_rubric_snapshot() -> BenchmarkDatasetSnapshot:
    return BenchmarkDatasetSnapshot(
        manifest=BenchmarkDatasetManifest(
            suite_slug="draco",
            suite_name="DRACO",
            dataset_version="2026-06-16-test",
            scoring_version=BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
            source_url="https://example.com/draco.jsonl",
            source_page_url="https://example.com/draco",
            license="open",
            sha256="1" * 64,
            row_count=2,
            file_name="data.jsonl",
            fetched_at="2026-06-16T00:00:00Z",
        ),
        items=(
            BenchmarkDatasetItem(
                item_index=0,
                problem="Use evidence to answer the query.",
                problem_category="rubric",
                answer=_weighted_rubric_answer(),
                answer_type=BenchmarkAnswerType.SINGLE_ANSWER,
            ),
            BenchmarkDatasetItem(
                item_index=1,
                problem="Use evidence to answer another query.",
                problem_category="rubric",
                answer=_weighted_rubric_answer(),
                answer_type=BenchmarkAnswerType.SINGLE_ANSWER,
            ),
        ),
    )


def _submission(
    *,
    batch_id: UUID,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    answer: str,
    total_tool_usage: ToolUsageSummary | None = None,
) -> MinerTaskRunSubmission:
    completed_at = datetime(2026, 4, 30, 10, 0, tzinfo=UTC)
    return MinerTaskRunSubmission(
        batch_id=batch_id,
        run=MinerTaskRun(
            session_id=uuid4(),
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=Response(text=answer),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=0.0,
                    total_score=0.0,
                    scoring_version="benchmark-invocation-only",
                ),
                total_tool_usage=total_tool_usage or ToolUsageSummary.zero(),
                elapsed_ms=125.0,
            ),
            completed_at=completed_at,
        ),
        score=0.0,
        usage=TokenUsageSummary.empty(),
        session=Session(
            session_id=uuid4(),
            uid=artifact.uid,
            task_id=task.task_id,
            issued_at=completed_at - timedelta(seconds=5),
            expires_at=completed_at + timedelta(minutes=5),
            budget_usd=task.budget_usd,
            usage=SessionUsage(),
            status=SessionStatus.COMPLETED,
            active_attempt=1,
        ),
    )


def _tool_usage_with_embedding() -> ToolUsageSummary:
    return ToolUsageSummary(
        llm=LlmUsageSummary(
            call_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.011,
            reference_cost=0.011,
        ),
        llm_cost=0.011,
        search_tool=SearchToolUsageSummary(call_count=2, cost=0.004, reference_cost=0.004),
        search_tool_cost=0.004,
        embedding=EmbeddingToolUsageSummary(call_count=3, cost=0.006, reference_cost=0.006),
        embedding_cost=0.006,
        reference_total_cost_usd=0.021,
    )


def test_local_benchmark_builds_platform_compatible_tasks() -> None:
    snapshot = _snapshot()
    source_batch_id = UUID("00000000-0000-4000-8000-00000000b501")
    run_id = benchmark_run_id_for_source_batch(
        suite_slug=snapshot.manifest.suite_slug,
        source_batch_id=source_batch_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
    )

    tasks = local_benchmark._build_tasks(run_id=run_id, snapshot=snapshot, items=snapshot.items)

    assert [task.query.text for task in tasks] == [
        "Who wrote the benchmark?",
        "Name the set.",
    ]
    assert [task.reference_answer.text for task in tasks] == [
        "The benchmark authors.",
        "Alpha, Beta",
    ]
    assert tasks[0].task_id != tasks[1].task_id


def test_local_benchmark_uses_existing_commons_benchmark_boundaries() -> None:
    assert local_benchmark._DEFAULT_SAMPLE_SIZE == miner_task_benchmark.BENCHMARK_SAMPLE_SIZE
    assert local_benchmark.aggregate_benchmark_metrics is miner_task_benchmark.aggregate_benchmark_metrics
    assert local_benchmark.sample_benchmark_items is miner_task_benchmark.sample_benchmark_items
    assert (
        local_benchmark.is_supported_benchmark_scoring_version
        is miner_task_benchmark.is_supported_benchmark_scoring_version
    )
    assert local_benchmark.is_supported_benchmark_scoring_version(BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION)


def test_local_benchmark_help_uses_benchmark_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        local_benchmark._parse_args(("--help",))

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--agent-path" in help_text
    assert "--source-batch-id" in help_text
    assert "--logging.debug" not in help_text


def test_local_benchmark_requires_explicit_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_path = tmp_path / "agent.py"
    agent_path.write_text("print('candidate')\n", encoding="utf-8")
    monkeypatch.setattr(local_benchmark, "load_public_env", lambda: None)

    with pytest.raises(ValueError, match="--suite is required"):
        asyncio.run(
            local_benchmark._amain(
                (
                    "--agent-path",
                    str(agent_path),
                    "--source-batch-id",
                    "00000000-0000-4000-8000-00000000b501",
                )
            )
        )


def test_local_benchmark_lists_current_suites(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(local_benchmark, "load_public_env", lambda: None)

    asyncio.run(local_benchmark._amain(("--list-suites",)))

    payload = json.loads(capsys.readouterr().out)
    assert [suite["suite_slug"] for suite in payload["suites"]] == [
        "deepresearch9k-l1",
        "deepsearchqa",
        "draco",
        "webwalkerqa",
    ]


def test_local_benchmark_uses_invocation_only_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = _snapshot()
    agent_path = tmp_path / "agent.py"
    agent_path.write_text("print('candidate')\n", encoding="utf-8")
    captured: dict[str, object] = {}
    events: list[str] = []

    class _FakeRuntime:
        async def evaluate_artifact(
            self,
            *,
            artifact_label,
            agent_source,
            artifact,
            batch_id,
            tasks,
            scoring_service,
        ):
            del artifact_label, agent_source
            assert scoring_service is captured["invocation_scoring"]
            assert isinstance(scoring_service, local_benchmark._InvocationOnlyScoringService)
            return ArtifactEvaluationOutcome(
                submissions=tuple(
                    _submission(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        answer="The benchmark authors.",
                    )
                    for task in tasks
                ),
            )

        async def aclose(self) -> None:
            captured["runtime_closed"] = True

    class _FakeBenchmarkScoringService:
        async def score(
            self,
            *,
            item: BenchmarkDatasetItem,
            generated_answer: str,
        ) -> local_benchmark._BenchmarkItemScore:
            del item, generated_answer
            return local_benchmark._BenchmarkItemScore(
                is_correct=True,
                score=1.0,
                score_reason="Matches the reference.",
                score_detail=None,
            )

    class _FakeScoringBundle:
        service = _FakeBenchmarkScoringService()
        config = local_benchmark.BenchmarkCorrectnessScoringConfig(
            provider="chutes",
            model="benchmark-model",
        )
        scoring_boundary = "benchmark-correctness-judge"
        scoring_method = "binary correctness against the canonical benchmark answer"
        uses_numeric_scores = False
        failed_item_report_score = 0.0

        async def aclose(self) -> None:
            captured["scoring_closed"] = True

    def _build_benchmark_scoring_bundle(scoring_version: str) -> _FakeScoringBundle:
        assert scoring_version == BENCHMARK_CORRECTNESS_SCORING_VERSION
        return _FakeScoringBundle()

    def _create_invocation_only_runtime(*, scoring_service, scoring_config, run_progress_root):
        events.append("create_invocation_only")
        assert events == ["load_public_env", "create_invocation_only"]
        captured["invocation_scoring"] = scoring_service
        captured["invocation_config"] = scoring_config
        captured["run_progress_root"] = run_progress_root
        return _FakeRuntime()

    monkeypatch.setattr(local_benchmark, "load_public_env", lambda: events.append("load_public_env"))
    monkeypatch.setattr(local_benchmark, "_create_invocation_only_runtime", _create_invocation_only_runtime)
    monkeypatch.setattr(local_benchmark, "_build_benchmark_scoring_bundle", _build_benchmark_scoring_bundle)
    monkeypatch.setattr(local_benchmark, "load_benchmark_snapshot", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(
        local_benchmark,
        "sample_benchmark_items",
        lambda *, items, run_id, dataset_version, scoring_version, sample_size: tuple(items)[:sample_size],
    )

    asyncio.run(
        local_benchmark._amain(
            (
                "--agent-path",
                str(agent_path),
                "--source-batch-id",
                "00000000-0000-4000-8000-00000000b501",
                "--suite",
                "deepsearchqa",
                "--sample-size",
                "1",
                "--output-dir",
                str(tmp_path),
            )
        )
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["mean_total_score"] == 1.0
    assert payload["item_count"] == 1
    assert captured["invocation_config"] is local_benchmark._INVOCATION_ONLY_SCORING_CONFIG
    assert captured["run_progress_root"] == tmp_path / ".harnyx-local-benchmark-progress" / payload["run_id"]
    assert captured["runtime_closed"] is True
    assert captured["scoring_closed"] is True


def test_miner_local_benchmark_ids_match_current_platform_values() -> None:
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


def test_miner_local_deepsearchqa_loader_loads_packaged_snapshot() -> None:
    snapshot = load_current_benchmark_snapshot("deepsearchqa")

    assert snapshot.manifest.suite_slug == "deepsearchqa"
    assert snapshot.manifest.dataset_version == "2026-04-02-google-main"
    assert snapshot.manifest.scoring_version == "correctness-v1"
    assert snapshot.manifest.row_count == 900
    assert len(snapshot.items) == 900
    assert snapshot.items[0].item_index == 0
    assert snapshot.items[0].problem
    assert snapshot.items[0].answer
    assert snapshot.items[0].answer_type in {
        BenchmarkAnswerType.SINGLE_ANSWER,
        BenchmarkAnswerType.SET_ANSWER,
    }


def test_miner_local_draco_loader_loads_current_weighted_rubric_snapshot() -> None:
    snapshot = load_current_benchmark_snapshot("draco")

    assert snapshot.manifest.suite_slug == "draco"
    assert snapshot.manifest.dataset_version == "2026-06-16-hf-ce076749"
    assert snapshot.manifest.scoring_version == BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION
    assert snapshot.manifest.row_count == 100
    assert len(snapshot.items) == 100
    assert snapshot.items[0].answer


class _FakeLocalBenchmarkScoringService:
    async def score(
        self,
        *,
        item: BenchmarkDatasetItem,
        generated_answer: str,
    ) -> local_benchmark._BenchmarkItemScore:
        del item, generated_answer
        raise AssertionError("report construction must not call the scoring service")


class _CapturingLocalBenchmarkScoringService:
    def __init__(self) -> None:
        self.generated_answers: list[str] = []

    async def score(
        self,
        *,
        item: BenchmarkDatasetItem,
        generated_answer: str,
    ) -> local_benchmark._BenchmarkItemScore:
        del item
        self.generated_answers.append(generated_answer)
        return local_benchmark._BenchmarkItemScore(
            is_correct=True,
            score=1.0,
            score_reason="Captured structured answer.",
            score_detail=None,
        )


class _FakeBenchmarkScoringRegistry:
    async def aclose(self) -> None:
        return None


def _correctness_scoring_bundle() -> local_benchmark._BenchmarkScoringBundle:
    return local_benchmark._BenchmarkScoringBundle(
        service=_FakeLocalBenchmarkScoringService(),
        registry=_FakeBenchmarkScoringRegistry(),
        config=local_benchmark.BenchmarkCorrectnessScoringConfig(
            provider="chutes",
            model="benchmark-model",
        ),
        scoring_boundary="benchmark-correctness-judge",
        scoring_method="binary correctness against the canonical benchmark answer",
        uses_numeric_scores=False,
        failed_item_report_score=0.0,
    )


def _weighted_rubric_scoring_bundle() -> local_benchmark._BenchmarkScoringBundle:
    return local_benchmark._BenchmarkScoringBundle(
        service=_FakeLocalBenchmarkScoringService(),
        registry=_FakeBenchmarkScoringRegistry(),
        config=local_benchmark.BenchmarkWeightedRubricScoringConfig(
            provider="vertex",
            model="rubric-model",
        ),
        scoring_boundary="benchmark-weighted-rubric-judge",
        scoring_method="weighted rubric criteria judged independently",
        uses_numeric_scores=True,
        failed_item_report_score=None,
    )


async def test_local_benchmark_uses_canonical_structured_answer() -> None:
    snapshot = _snapshot()
    task = local_benchmark._build_tasks(
        run_id=uuid4(),
        snapshot=snapshot,
        items=snapshot.items[:1],
    )[0]
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="structured-output",
        size_bytes=128,
    )
    submission = _submission(
        batch_id=uuid4(),
        artifact=artifact,
        task=task,
        answer="unused legacy answer",
    )
    submission = submission.model_copy(
        update={
            "run": submission.run.model_copy(
                update={"response": Response(output={"z": [1, None], "a": True})},
            )
        }
    )
    scoring_service = _CapturingLocalBenchmarkScoringService()

    result = await local_benchmark._score_item_submission(
        item=snapshot.items[0],
        task=task,
        submission=submission,
        scoring_service=scoring_service,
    )

    assert result.error_code is None
    assert scoring_service.generated_answers == ['{"a":true,"z":[1,null]}']


def test_local_benchmark_report_includes_answers_and_summary(tmp_path: Path) -> None:
    snapshot = _snapshot()
    source_batch_id = UUID("00000000-0000-4000-8000-00000000b501")
    run_id = benchmark_run_id_for_source_batch(
        suite_slug=snapshot.manifest.suite_slug,
        source_batch_id=source_batch_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
    )
    backing_batch_id = uuid4()
    target_bytes = b"print('agent')\n"
    target_artifact = local_benchmark._build_target_artifact_spec(
        run_id=run_id,
        target_bytes=target_bytes,
    )
    tasks = local_benchmark._build_tasks(run_id=run_id, snapshot=snapshot, items=snapshot.items)
    first_submission = _submission(
        batch_id=backing_batch_id,
        artifact=target_artifact,
        task=tasks[0],
        answer="The benchmark authors.",
    )
    results = (
        local_benchmark._BenchmarkItemResult(
            item=snapshot.items[0],
            task=tasks[0],
            submission=first_submission,
            score=local_benchmark._BenchmarkItemScore(
                is_correct=True,
                score=1.0,
                score_reason="Matches the reference.",
                score_detail=None,
            ),
            error_code=None,
            error_message=None,
        ),
        local_benchmark._BenchmarkItemResult(
            item=snapshot.items[1],
            task=tasks[1],
            submission=None,
            score=None,
            error_code="missing_submission",
            error_message="candidate did not produce a benchmark submission",
        ),
    )

    report = local_benchmark._build_report(
        snapshot=snapshot,
        source_batch_id=source_batch_id,
        run_id=run_id,
        backing_batch_id=backing_batch_id,
        target_path=Path("train.py"),
        target_bytes=target_bytes,
        target_artifact=target_artifact,
        results=results,
        scoring=_correctness_scoring_bundle(),
        output_dir=tmp_path,
        elapsed_seconds=12.5,
        parallelism=1,
    )

    assert report["summary"]["item_count"] == 2
    assert report["summary"]["completed_item_count"] == 1
    assert report["summary"]["failed_item_count"] == 1
    assert report["summary"]["correct_item_count"] == 1
    assert report["summary"]["mean_total_score"] == 0.5
    assert report["items"][0]["problem"] == "Who wrote the benchmark?"
    assert report["items"][0]["reference_answer"] == "The benchmark authors."
    assert report["items"][0]["generated_answer"]["text"] == "The benchmark authors."
    assert report["items"][0]["is_correct"] is True
    assert report["items"][0]["score"] == 1.0
    assert report["items"][0]["score_reason"] == "Matches the reference."
    assert report["items"][0]["score_detail"] is None
    assert report["items"][1]["error"]["code"] == "missing_submission"
    assert report["items"][1]["score"] == 0.0


def test_local_benchmark_cost_totals_preserve_embedding_breakdown(tmp_path: Path) -> None:
    snapshot = _snapshot()
    source_batch_id = UUID("00000000-0000-4000-8000-00000000b501")
    run_id = benchmark_run_id_for_source_batch(
        suite_slug=snapshot.manifest.suite_slug,
        source_batch_id=source_batch_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
    )
    backing_batch_id = uuid4()
    target_bytes = b"print('agent')\n"
    target_artifact = local_benchmark._build_target_artifact_spec(
        run_id=run_id,
        target_bytes=target_bytes,
    )
    tasks = local_benchmark._build_tasks(run_id=run_id, snapshot=snapshot, items=snapshot.items)
    submission = _submission(
        batch_id=backing_batch_id,
        artifact=target_artifact,
        task=tasks[0],
        answer="The benchmark authors.",
        total_tool_usage=_tool_usage_with_embedding(),
    )

    report = local_benchmark._build_report(
        snapshot=snapshot,
        source_batch_id=source_batch_id,
        run_id=run_id,
        backing_batch_id=backing_batch_id,
        target_path=Path("train.py"),
        target_bytes=target_bytes,
        target_artifact=target_artifact,
        results=(
            local_benchmark._BenchmarkItemResult(
                item=snapshot.items[0],
                task=tasks[0],
                submission=submission,
                score=local_benchmark._BenchmarkItemScore(
                    is_correct=True,
                    score=1.0,
                    score_reason="Matches the reference.",
                    score_detail=None,
                ),
                error_code=None,
                error_message=None,
            ),
        ),
        scoring=_correctness_scoring_bundle(),
        output_dir=tmp_path,
        elapsed_seconds=12.5,
        parallelism=1,
    )

    summary_totals = report["summary"]["cost_totals"]
    item_totals = report["items"][0]["invocation"]["cost_totals"]
    assert summary_totals["llm_cost_usd"] == pytest.approx(0.011)
    assert summary_totals["search_tool_cost_usd"] == pytest.approx(0.004)
    assert summary_totals["embedding_cost_usd"] == pytest.approx(0.006)
    assert summary_totals["total_cost_usd"] == pytest.approx(0.021)
    assert summary_totals["embedding_call_count"] == 3
    assert item_totals == summary_totals


def test_weighted_rubric_local_benchmark_report_uses_numeric_score_detail(tmp_path: Path) -> None:
    snapshot = _weighted_rubric_snapshot()
    source_batch_id = UUID("00000000-0000-4000-8000-00000000b501")
    run_id = benchmark_run_id_for_source_batch(
        suite_slug=snapshot.manifest.suite_slug,
        source_batch_id=source_batch_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
    )
    backing_batch_id = uuid4()
    target_bytes = b"print('agent')\n"
    target_artifact = local_benchmark._build_target_artifact_spec(
        run_id=run_id,
        target_bytes=target_bytes,
    )
    tasks = local_benchmark._build_tasks(run_id=run_id, snapshot=snapshot, items=snapshot.items)
    first_submission = _submission(
        batch_id=backing_batch_id,
        artifact=target_artifact,
        task=tasks[0],
        answer="Supported answer.",
    )
    score_detail = {
        "scoring_version": BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION,
        "rubric_id": "rubric-1",
        "normalized_score": 0.75,
        "criteria": [
            {
                "criterion_id": "cites-supported-answer",
                "verdict": "MET",
                "met": True,
                "weight": 2.0,
                "contribution": 2.0,
                "justification": "The answer cites support.",
            }
        ],
    }
    results = (
        local_benchmark._BenchmarkItemResult(
            item=snapshot.items[0],
            task=tasks[0],
            submission=first_submission,
            score=local_benchmark._BenchmarkItemScore(
                is_correct=None,
                score=0.75,
                score_reason=None,
                score_detail=score_detail,
            ),
            error_code=None,
            error_message=None,
        ),
        local_benchmark._BenchmarkItemResult(
            item=snapshot.items[1],
            task=tasks[1],
            submission=None,
            score=None,
            error_code="benchmark_scoring_failed",
            error_message="weighted rubric judge failed",
        ),
    )

    report = local_benchmark._build_report(
        snapshot=snapshot,
        source_batch_id=source_batch_id,
        run_id=run_id,
        backing_batch_id=backing_batch_id,
        target_path=Path("train.py"),
        target_bytes=target_bytes,
        target_artifact=target_artifact,
        results=results,
        scoring=_weighted_rubric_scoring_bundle(),
        output_dir=tmp_path,
        elapsed_seconds=12.5,
        parallelism=1,
    )

    assert report["summary"]["completed_item_count"] == 1
    assert report["summary"]["failed_item_count"] == 1
    assert report["summary"]["correct_item_count"] is None
    assert report["summary"]["mean_total_score"] == pytest.approx(0.375)
    assert report["evaluation_config"]["scoring_boundary"] == "benchmark-weighted-rubric-judge"
    assert report["scoring_context"]["method"] == "weighted rubric criteria judged independently"
    assert report["items"][0]["is_correct"] is None
    assert report["items"][0]["score"] == 0.75
    assert report["items"][0]["score_reason"] is None
    assert report["items"][0]["score_detail"] == score_detail
    assert report["items"][1]["score"] is None
    assert report["items"][1]["score_detail"] is None
    assert report["items"][1]["error"]["code"] == "benchmark_scoring_failed"


def test_weighted_rubric_report_keeps_correct_item_count_na_when_all_items_fail(tmp_path: Path) -> None:
    snapshot = _weighted_rubric_snapshot()
    source_batch_id = UUID("00000000-0000-4000-8000-00000000b501")
    run_id = benchmark_run_id_for_source_batch(
        suite_slug=snapshot.manifest.suite_slug,
        source_batch_id=source_batch_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
    )
    backing_batch_id = uuid4()
    target_bytes = b"print('agent')\n"
    target_artifact = local_benchmark._build_target_artifact_spec(
        run_id=run_id,
        target_bytes=target_bytes,
    )
    tasks = local_benchmark._build_tasks(run_id=run_id, snapshot=snapshot, items=snapshot.items)
    results = (
        local_benchmark._BenchmarkItemResult(
            item=snapshot.items[0],
            task=tasks[0],
            submission=None,
            score=None,
            error_code="benchmark_scoring_failed",
            error_message="weighted rubric judge failed",
        ),
        local_benchmark._BenchmarkItemResult(
            item=snapshot.items[1],
            task=tasks[1],
            submission=None,
            score=None,
            error_code="benchmark_scoring_failed",
            error_message="weighted rubric judge failed",
        ),
    )

    report = local_benchmark._build_report(
        snapshot=snapshot,
        source_batch_id=source_batch_id,
        run_id=run_id,
        backing_batch_id=backing_batch_id,
        target_path=Path("train.py"),
        target_bytes=target_bytes,
        target_artifact=target_artifact,
        results=results,
        scoring=_weighted_rubric_scoring_bundle(),
        output_dir=tmp_path,
        elapsed_seconds=12.5,
        parallelism=1,
    )

    assert report["summary"]["completed_item_count"] == 0
    assert report["summary"]["failed_item_count"] == 2
    assert report["summary"]["correct_item_count"] is None
    assert report["summary"]["mean_total_score"] == 0.0
    assert [item["score"] for item in report["items"]] == [None, None]


def test_weighted_rubric_judge_exception_fails_item_without_partial_score(tmp_path: Path) -> None:
    snapshot = _weighted_rubric_snapshot()
    source_batch_id = UUID("00000000-0000-4000-8000-00000000b501")
    run_id = benchmark_run_id_for_source_batch(
        suite_slug=snapshot.manifest.suite_slug,
        source_batch_id=source_batch_id,
        dataset_version=snapshot.manifest.dataset_version,
        scoring_version=snapshot.manifest.scoring_version,
    )
    backing_batch_id = uuid4()
    target_bytes = b"print('agent')\n"
    target_artifact = local_benchmark._build_target_artifact_spec(
        run_id=run_id,
        target_bytes=target_bytes,
    )
    tasks = local_benchmark._build_tasks(run_id=run_id, snapshot=snapshot, items=snapshot.items)
    submission = _submission(
        batch_id=backing_batch_id,
        artifact=target_artifact,
        task=tasks[0],
        answer="Supported answer.",
    )

    class _FailingWeightedRubricScoringService:
        async def score(
            self,
            *,
            item: BenchmarkDatasetItem,
            generated_answer: str,
        ) -> local_benchmark._BenchmarkItemScore:
            del item, generated_answer
            raise RuntimeError("judge failed")

    result = asyncio.run(
        local_benchmark._score_item_submission(
            item=snapshot.items[0],
            task=tasks[0],
            submission=submission,
            scoring_service=_FailingWeightedRubricScoringService(),
        )
    )

    report = local_benchmark._build_report(
        snapshot=snapshot,
        source_batch_id=source_batch_id,
        run_id=run_id,
        backing_batch_id=backing_batch_id,
        target_path=Path("train.py"),
        target_bytes=target_bytes,
        target_artifact=target_artifact,
        results=(result,),
        scoring=_weighted_rubric_scoring_bundle(),
        output_dir=tmp_path,
        elapsed_seconds=12.5,
        parallelism=1,
    )

    assert result.score is None
    assert result.error_code == "benchmark_scoring_failed"
    assert result.error_message == "judge failed"
    assert report["summary"]["completed_item_count"] == 0
    assert report["summary"]["failed_item_count"] == 1
    assert report["summary"]["correct_item_count"] is None
    assert report["summary"]["mean_total_score"] == 0.0
    assert report["items"][0]["score"] is None
    assert report["items"][0]["score_detail"] is None
    assert report["items"][0]["error"]["code"] == "benchmark_scoring_failed"


def test_weighted_rubric_local_benchmark_bundle_uses_dedicated_judge_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_provider_names: list[str] = []

    class _FakeRegistry:
        def resolve(self, provider_name: str) -> object:
            resolved_provider_names.append(provider_name)
            return object()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(local_benchmark, "load_public_env", lambda: None)
    monkeypatch.setenv("BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER", "vertex")
    monkeypatch.setenv("BENCHMARK_RUBRIC_JUDGE_LLM_MODEL", "rubric-model")
    monkeypatch.setenv("BENCHMARK_RUBRIC_JUDGE_LLM_TEMPERATURE", "0")
    monkeypatch.setenv("BENCHMARK_RUBRIC_JUDGE_LLM_REASONING_EFFORT", "high")
    monkeypatch.setenv("BENCHMARK_RUBRIC_JUDGE_LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.delenv("BENCHMARK_LLM_MODEL", raising=False)
    monkeypatch.setattr(
        local_benchmark,
        "build_cached_llm_provider_registry",
        lambda **_kwargs: _FakeRegistry(),
    )

    bundle = local_benchmark._build_benchmark_scoring_bundle(BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION)

    assert resolved_provider_names == ["vertex"]
    assert isinstance(bundle.service, local_benchmark._WeightedRubricLocalBenchmarkScoringService)
    assert bundle.config == local_benchmark.BenchmarkWeightedRubricScoringConfig(
        provider="vertex",
        model="rubric-model",
        temperature=0.0,
        max_output_tokens=None,
        reasoning_effort="high",
        timeout_seconds=45.0,
    )
    assert bundle.scoring_boundary == "benchmark-weighted-rubric-judge"
    assert bundle.uses_numeric_scores is True
    assert bundle.failed_item_report_score is None


def test_weighted_rubric_local_benchmark_bundle_rejects_missing_dedicated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_benchmark, "load_public_env", lambda: None)
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "vertex")
    monkeypatch.setenv("BENCHMARK_LLM_MODEL", "benchmark-model")
    monkeypatch.delenv("BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("BENCHMARK_RUBRIC_JUDGE_LLM_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER"):
        local_benchmark._build_benchmark_scoring_bundle(BENCHMARK_WEIGHTED_RUBRIC_SCORING_VERSION)
