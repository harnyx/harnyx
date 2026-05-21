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
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.miner_task_benchmark import (
    BenchmarkAnswerType,
    BenchmarkCorrectnessScore,
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


def _submission(
    *,
    batch_id: UUID,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    answer: str,
) -> MinerTaskRunSubmission:
    completed_at = datetime(2026, 4, 30, 10, 0, tzinfo=UTC)
    return MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=0,
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
                total_tool_usage=ToolUsageSummary.zero(),
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
                unresolved_tasks=(),
                timeout_observations_by_pair={},
            )

        async def aclose(self) -> None:
            captured["runtime_closed"] = True

    class _FakeBenchmarkScoringService:
        async def score(self, *, question: str, reference_answer: str, generated_answer: str):
            del question, reference_answer, generated_answer
            return BenchmarkCorrectnessScore(is_correct=True, reason="Matches the reference.")

    class _FakeScoringBundle:
        service = _FakeBenchmarkScoringService()
        config = local_benchmark.BenchmarkCorrectnessScoringConfig(
            provider="chutes",
            model="benchmark-model",
        )

        async def aclose(self) -> None:
            captured["scoring_closed"] = True

    def _create_invocation_only_runtime(*, scoring_service, scoring_config, run_progress_root):
        events.append("create_invocation_only")
        assert events == ["load_public_env", "create_invocation_only"]
        captured["invocation_scoring"] = scoring_service
        captured["invocation_config"] = scoring_config
        captured["run_progress_root"] = run_progress_root
        return _FakeRuntime()

    monkeypatch.setattr(local_benchmark, "load_public_env", lambda: events.append("load_public_env"))
    monkeypatch.setattr(local_benchmark, "_create_invocation_only_runtime", _create_invocation_only_runtime)
    monkeypatch.setattr(local_benchmark, "_build_benchmark_scoring_bundle", _FakeScoringBundle)
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
            score=BenchmarkCorrectnessScore(is_correct=True, reason="Matches the reference."),
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
        scoring_config=local_benchmark.BenchmarkCorrectnessScoringConfig(
            provider="chutes",
            model="benchmark-model",
        ),
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
    assert report["items"][1]["error"]["code"] == "missing_submission"
