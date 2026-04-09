from __future__ import annotations

import asyncio
import base64
import json
import runpy
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest

from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import LlmUsageTotals, Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_usage import (
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.manager import SandboxDeployment
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.state import DEFAULT_STATE_DIR
from harnyx_miner import local_eval
from harnyx_miner.platform_monitoring import PlatformMonitoringClient, SelectedBatchContext
from harnyx_miner_sdk.json_types import JsonValue
from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission, ScriptArtifactSpec, TokenUsageSummary
from harnyx_validator.application.services.evaluation_runner import (
    ArtifactEvaluationOutcome,
    ArtifactFailure,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.application.services.evaluation_scoring import EvaluationScoringConfig
from harnyx_validator.domain.evaluation import MinerTaskRun


def _write_agent(path: Path, *, answer: str = "local answer") -> None:
    path.write_text(
        "\n".join(
            (
                "from harnyx_miner_sdk.decorators import entrypoint",
                "from harnyx_miner_sdk.query import Query, Response",
                "",
                '@entrypoint("query")',
                "async def query(query: Query) -> Response:",
                f'    return Response(text="{answer}")',
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_sleeping_agent(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "import asyncio",
                "from harnyx_miner_sdk.decorators import entrypoint",
                "from harnyx_miner_sdk.query import Query, Response",
                "",
                '@entrypoint("query")',
                "async def query(query: Query) -> Response:",
                "    await asyncio.sleep(60)",
                '    return Response(text="never")',
                "",
            )
        ),
        encoding="utf-8",
    )


def _task(task_id, text: str) -> MinerTask:
    return MinerTask(
        task_id=task_id,
        query=Query(text=text),
        reference_answer=ReferenceAnswer(text=f"reference for {text}"),
        budget_usd=0.5,
    )




def _usage_totals() -> dict[str, dict[str, LlmUsageTotals]]:
    return {
        "openai": {
            "openai/gpt-oss-120b-TEE": LlmUsageTotals(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                call_count=1,
            )
        }
    }


def _tool_usage(*, total_cost: float) -> ToolUsageSummary:
    usage = _usage_totals()["openai"]["openai/gpt-oss-120b-TEE"]
    return ToolUsageSummary(
        search_tool=SearchToolUsageSummary(call_count=1, cost=0.001),
        search_tool_cost=0.001,
        llm=LlmUsageSummary(
            call_count=usage.call_count,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=round(total_cost - 0.001, 6),
            providers={
                "openai": {
                    "openai/gpt-oss-120b-TEE": LlmModelUsageCost(
                        usage=usage,
                        cost=round(total_cost - 0.001, 6),
                    )
                }
            },
        ),
        llm_cost=round(total_cost - 0.001, 6),
    )


def _submission(
    *,
    batch_id,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    score: float,
    answer_text: str,
    citations: tuple[dict[str, str], ...] | None = None,
    attempt_count: int = 1,
) -> MinerTaskRunSubmission:
    usage_totals = _usage_totals()
    completed_at = datetime(2026, 3, 27, 6, 0, tzinfo=UTC)
    return MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=0,
        run=MinerTaskRun(
            session_id=uuid4(),
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=Response(text=answer_text, citations=citations),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=score,
                    similarity_score=score,
                    total_score=score,
                    scoring_version="v1",
                ),
                total_tool_usage=_tool_usage(total_cost=0.011),
                elapsed_ms=125.0,
            ),
            completed_at=completed_at,
        ),
        score=score,
        usage=TokenUsageSummary.from_totals(usage_totals),
        session=Session(
            session_id=uuid4(),
            uid=artifact.uid,
            task_id=task.task_id,
            issued_at=completed_at - timedelta(seconds=5),
            expires_at=completed_at + timedelta(minutes=5),
            budget_usd=task.budget_usd,
            usage=SessionUsage(llm_usage_totals=usage_totals),
            status=SessionStatus.COMPLETED,
            active_attempt=attempt_count,
        ),
    )


def _batch_detail(*, batch_id, champion_artifact_id, tasks: tuple[MinerTask, ...]) -> dict[str, object]:
    return {
        "summary": {
            "batch_id": str(batch_id),
            "status": "completed",
            "created_at": "2026-03-27T06:00:00Z",
            "cutoff_at": "2026-03-27T05:55:00Z",
            "completed_at": "2026-03-27T06:02:00Z",
            "failed_at": None,
            "artifact_count": 2,
            "task_count": len(tasks),
            "champion_artifact_id": str(champion_artifact_id),
        },
        "batch": {
            "batch_id": str(batch_id),
            "cutoff_at": "2026-03-27T05:55:00Z",
            "created_at": "2026-03-27T06:00:00Z",
            "completed_at": "2026-03-27T06:02:00Z",
            "failed_at": None,
            "champion_artifact_id": str(champion_artifact_id),
            "tasks": tuple(task.model_dump(mode="json") for task in tasks),
            "artifacts": (
                {
                    "uid": 2,
                    "artifact_id": str(champion_artifact_id),
                    "content_hash": "champion-hash",
                    "size_bytes": 128,
                },
                {
                    "uid": 3,
                    "artifact_id": str(uuid4()),
                    "content_hash": "challenger-hash",
                    "size_bytes": 128,
                },
            ),
        },
        "artifact_aggregates": (),
        "observed_artifact_aggregates": (),
        "deliveries": (),
        "cost_totals": {
            "llm_cost_usd": 0.1,
            "search_tool_cost_usd": 0.02,
            "total_cost_usd": 0.12,
            "llm_total_tokens": 100,
            "llm_call_count": 6,
            "search_tool_call_count": 6,
        },
        "observed_cost_totals": {
            "llm_cost_usd": 0.1,
            "search_tool_cost_usd": 0.02,
            "total_cost_usd": 0.12,
            "llm_total_tokens": 100,
            "llm_call_count": 6,
            "search_tool_call_count": 6,
        },
    }


def _recorded_rows(*, batch_id, champion_artifact_id, tasks: tuple[MinerTask, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        rows.append(
            {
                "batch_id": str(batch_id),
                "validator_hotkey": "validator-1",
                "validator_uid": 10,
                "miner_uid": 2,
                "artifact_id": str(champion_artifact_id),
                "task_id": str(task.task_id),
                "score": 0.61,
                "received_at": "2026-03-27T06:02:00Z",
                "response": {"text": f"recorded champion {task.query.text}"},
                "specifics": {
                    "score_breakdown": {
                        "comparison_score": 0.61,
                        "similarity_score": 0.61,
                        "total_score": 0.61,
                        "scoring_version": "v1",
                    },
                    "total_tool_usage": {
                        "search_tool": {"call_count": 1, "cost": 0.001},
                        "search_tool_cost": 0.001,
                        "llm": {
                            "call_count": 1,
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                            "cost": 0.01,
                            "providers": {},
                        },
                        "llm_cost": 0.01,
                    },
                    "elapsed_ms": 100.0,
                    "error": None,
                },
                "cost_totals": {
                    "llm_cost_usd": 0.01,
                    "search_tool_cost_usd": 0.001,
                    "total_cost_usd": 0.011,
                    "llm_total_tokens": 15,
                    "llm_call_count": 1,
                    "search_tool_call_count": 1,
                },
                "llm_models": (),
                "payload_json": {"source": "platform"},
            }
        )
    return tuple(rows)


class _FakeMonitoringClient:
    def __init__(self, *, batch_context: SelectedBatchContext, champion_script: dict[str, object]) -> None:
        self.batch_context = batch_context
        self.champion_script = champion_script
        self.resolve_calls: list[object] = []
        self.script_calls = 0
        self.closed = False

    def resolve_batch_context(self, batch_id) -> SelectedBatchContext:
        self.resolve_calls.append(batch_id)
        return self.batch_context

    def get_script(self, artifact_id) -> dict[str, object]:
        self.script_calls += 1
        assert str(artifact_id) == str(self.champion_script["artifact_id"])
        return self.champion_script

    def close(self) -> None:
        self.closed = True


class _FakeRuntime:
    def __init__(
        self,
        *,
        batch_id,
        champion_artifact_id,
        tasks: tuple[MinerTask, ...],
        target_scores: tuple[float, ...] | None = None,
        champion_scores: tuple[float, ...] | None = None,
        delay_seconds: float = 0.0,
        target_outcome: ArtifactEvaluationOutcome | None = None,
        champion_outcome: ArtifactEvaluationOutcome | None = None,
    ) -> None:
        self.scoring_config = EvaluationScoringConfig(
            provider="chutes",
            model="openai/gpt-oss-120b-TEE",
            timeout_seconds=30.0,
        )
        self.settings = SimpleNamespace(
            sandbox=SimpleNamespace(
                sandbox_image="local/harnyx-sandbox:0.1.0-dev",
                sandbox_pull_policy="missing",
            )
        )
        self.calls: list[tuple[str, str]] = []
        self._target_scores = target_scores or tuple(0.9 - (index * 0.2) for index in range(len(tasks)))
        self._champion_scores = champion_scores or tuple(0.6 for _ in tasks)
        self._batch_id = batch_id
        self._champion_artifact_id = champion_artifact_id
        self._tasks = tasks
        self._delay_seconds = delay_seconds
        self._target_outcome = target_outcome
        self._champion_outcome = champion_outcome
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False
        self.progress_reporter: Any | None = None

    async def evaluate_artifact(
        self,
        *,
        artifact_label: str,
        agent_source: bytes,
        artifact: ScriptArtifactSpec,
        batch_id,
        tasks: Sequence[MinerTask],
    ) -> ArtifactEvaluationOutcome:
        self.calls.append((agent_source.decode("utf-8"), str(artifact.artifact_id)))
        assert batch_id == self._batch_id
        assert tuple(tasks) == self._tasks
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.progress_reporter is not None:
            self.progress_reporter.begin_artifact(
                label=artifact_label,
                artifact=artifact,
                task_count=len(tasks),
            )
        try:
            if self._delay_seconds > 0:
                await asyncio.sleep(self._delay_seconds)
            if artifact.artifact_id == self._champion_artifact_id:
                override = self._champion_outcome
                scores = self._champion_scores
                prefix = "champion"
            else:
                override = self._target_outcome
                scores = self._target_scores
                prefix = "target"

            if override is None:
                outcome = ArtifactEvaluationOutcome(
                    submissions=tuple(
                        _submission(
                            batch_id=batch_id,
                            artifact=artifact,
                            task=task,
                            score=score,
                            answer_text=f"{prefix} answer {index}",
                            citations=(
                                {
                                    "url": f"https://example.com/{prefix}/{index}",
                                    "title": f"{prefix.title()} source {index}",
                                    "note": f"{prefix.title()} note {index}",
                                },
                            ),
                            attempt_count=2 if prefix == "target" and index == 0 else 1,
                        )
                        for index, (task, score) in enumerate(zip(tasks, scores, strict=True))
                    ),
                    unresolved_tasks=(),
                    timeout_observations_by_pair={},
                )
            else:
                outcome = override
            if self.progress_reporter is not None:
                for submission in outcome.submissions:
                    self.progress_reporter.record(submission)
                if outcome.artifact_failure is None:
                    self.progress_reporter.finish_artifact(
                        label=artifact_label,
                        artifact=artifact,
                        submissions=outcome.submissions,
                    )
            return outcome
        finally:
            self.in_flight -= 1

    async def aclose(self) -> None:
        self.closed = True


class _UnusedToolExecutor:
    async def execute(self, request) -> object:  # pragma: no cover - defensive
        raise AssertionError(f"tool execution should not be reached: {request}")


class _FakeAsyncResource:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeSandboxClient(SandboxClient):
    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        token: str,
        session_id: UUID,
    ) -> Mapping[str, JsonValue]:
        del payload, context, token, session_id
        raise AssertionError(
            f"sandbox client invoke should not be reached in this unit test: entrypoint={entrypoint}"
        )

    def close(self) -> None:
        return None


class _FakeSandboxManager:
    def __init__(self) -> None:
        self.started_options: list[SandboxOptions] = []
        self.stopped_deployments: list[SandboxDeployment] = []
        self.clients: list[_FakeSandboxClient] = []
        self.mount_paths_exist: list[bool] = []

    def start(self, options: SandboxOptions) -> SandboxDeployment:
        self.started_options.append(options)
        self.mount_paths_exist.append(Path(options.volumes[0][0]).exists())
        client = _FakeSandboxClient()
        self.clients.append(client)
        return SandboxDeployment(
            client=client,
            identifier=f"sandbox-{len(self.started_options)}",
            base_url="http://127.0.0.1:38000",
        )

    def stop(self, deployment: SandboxDeployment) -> None:
        self.stopped_deployments.append(deployment)


class _BlockingSandboxManager(_FakeSandboxManager):
    def __init__(self) -> None:
        super().__init__()
        self.start_entered = threading.Event()
        self.release_start = threading.Event()
        self.returned_deployment: SandboxDeployment | None = None

    def start(self, options: SandboxOptions) -> SandboxDeployment:
        self.started_options.append(options)
        self.mount_paths_exist.append(Path(options.volumes[0][0]).exists())
        self.start_entered.set()
        self.release_start.wait(timeout=1.0)
        client = _FakeSandboxClient()
        self.clients.append(client)
        deployment = SandboxDeployment(
            client=client,
            identifier=f"sandbox-{len(self.started_options)}",
            base_url="http://127.0.0.1:38000",
        )
        self.returned_deployment = deployment
        return deployment


class _FakeToolHost:
    def __init__(self, *, port: int = 39100, host_container_url: str = "http://host.docker.internal:39100") -> None:
        self.port = port
        self.host_container_url = host_container_url
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _CapturingRunner:
    def __init__(self, results: Sequence[ArtifactEvaluationOutcome]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    async def evaluate_artifact(
        self,
        *,
        batch_id,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        orchestrator,
    ) -> ArtifactEvaluationOutcome:
        sandbox_client = orchestrator._invoker._sandbox
        self.calls.append(
            {
                "batch_id": batch_id,
                "artifact_id": artifact.artifact_id,
                "tasks": tuple(tasks),
                "sandbox_client": sandbox_client,
            }
        )
        return self._results.pop(0)


def test_local_eval_writes_default_reports_for_latest_completed_vs_champion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (
        MinerTask(
            task_id=uuid4(),
            query=Query(text="task one"),
            reference_answer=ReferenceAnswer(
                text="reference for task one",
                citations=(
                    {
                        "url": "https://example.com/reference",
                        "title": "Reference source",
                        "note": "Reference support",
                    },
                ),
            ),
            budget_usd=0.5,
        ),
        _task(uuid4(), "task two"),
    )
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": base64.b64encode(

                    b"from harnyx_miner_sdk.decorators import entrypoint\n"
                    b"from harnyx_miner_sdk.query import Query, Response\n"
                    b'@entrypoint("query")\n'
                    b"async def query(query: Query) -> Response:\n"
                    b'    return Response(text="champion")\n'

            ).decode("ascii"),
        },
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(lambda *, progress_reporter=None: _bind_progress(runtime, progress_reporter)),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    json_path = tmp_path / f"local-eval-report-{batch_id}-vs-champion.json"
    markdown_path = tmp_path / f"local-eval-report-{batch_id}-vs-champion.md"
    assert json_path.exists()
    assert markdown_path.exists()
    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert monitoring.resolve_calls == [None]
    assert monitoring.script_calls == 1
    assert report["mode"] == "vs-champion"
    assert report["batch_metadata"]["selection_source"] == "latest-completed"
    assert report["evaluation_config"]["artifact_task_parallelism"] == 5
    assert report["evaluation_config"]["artifact_evaluation_parallelism"] == 2
    assert report["local_result_summary"]["local_champion_selection"]["selected_label"] == "target"
    assert report["local_result_summary"]["head_to_head"]["winner_by_total_score"] == "target"
    assert len(report["local_result_summary"]["leaderboard"]) == 2
    assert len(report["tasks"]) == 2
    assert report["tasks"][0]["reference_answer"]["citations"] == [
        {
            "url": "https://example.com/reference",
            "title": "Reference source",
            "note": "Reference support",
        }
    ]
    assert report["tasks"][0]["target"]["answer"]["text"] == "target answer 0"
    assert report["tasks"][0]["opponent"]["answer"]["text"] == "champion answer 0"
    assert report["tasks"][0]["target"]["attempt_count"] == 2
    assert report["recorded_platform_context"]["results"][0]["payload_json"] == {"source": "platform"}
    assert runtime.closed is True
    assert monitoring.closed is True
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "- Reference citations:" in markdown
    assert "Reference source - https://example.com/reference - Reference support" in markdown
    assert "- Target citations:" in markdown
    assert "Target source 0 - https://example.com/target/0 - Target note 0" in markdown
    assert "- Opponent citations:" in markdown
    assert "Champion source 0 - https://example.com/champion/0 - Champion note 0" in markdown


def test_render_answer_markdown_uses_shared_models_and_ignores_empty_optional_citation_fields() -> None:
    lines = local_eval._render_answer_markdown(
        "Target",
        {
            "text": "answer",
            "citations": [
                {
                    "url": "https://example.com/source",
                    "title": "",
                    "note": "",
                }
            ],
        },
        model_type=Response,
    )

    assert lines == [
        "- Target answer: answer",
        "- Target citations:",
        "  - https://example.com/source",
    ]


def test_local_eval_target_only_skips_champion_fetch_and_keeps_recorded_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (_task(uuid4(), "solo task"),)
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="explicit",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": "",
        },
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path, answer="target only")

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(lambda *, progress_reporter=None: _bind_progress(runtime, progress_reporter)),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    local_eval.main(
        [
            "--agent-path",
            str(agent_path),
            "--batch-id",
            str(batch_id),
            "--mode",
            "target-only",
            "--output-dir",
            str(tmp_path),
        ]
    )

    json_path = tmp_path / f"local-eval-report-{batch_id}-target-only.json"
    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert monitoring.resolve_calls == [batch_id]
    assert monitoring.script_calls == 0
    assert report["mode"] == "target-only"
    assert report["local_result_summary"]["head_to_head"] is None
    assert len(report["local_result_summary"]["leaderboard"]) == 1
    assert report["tasks"][0]["opponent"] is None
    assert len(report["recorded_platform_context"]["results"]) == 1
    assert len(runtime.calls) == 1


def test_local_eval_vs_champion_uses_platform_cascade_not_raw_total_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (
        _task(uuid4(), "task one"),
        _task(uuid4(), "task two"),
    )
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": base64.b64encode(

                    b"from harnyx_miner_sdk.decorators import entrypoint\n"
                    b"from harnyx_miner_sdk.query import Query, Response\n"
                    b'@entrypoint("query")\n'
                    b"async def query(query: Query) -> Response:\n"
                    b'    return Response(text="champion")\n'

            ).decode("ascii"),
        },
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
        target_scores=(0.65, 0.61),
        champion_scores=(0.6, 0.6),
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(lambda *, progress_reporter=None: _bind_progress(runtime, progress_reporter)),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    report = json.loads((tmp_path / f"local-eval-report-{batch_id}-vs-champion.json").read_text(encoding="utf-8"))

    assert report["local_result_summary"]["head_to_head"]["winner_by_total_score"] == "target"
    assert report["local_result_summary"]["local_champion_selection"]["selected_label"] == "champion"


def test_local_eval_head_to_head_winner_uses_raw_totals_not_rounded_totals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (
        _task(uuid4(), "task one"),
        _task(uuid4(), "task two"),
    )
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": base64.b64encode(
                b"from harnyx_miner_sdk.decorators import entrypoint\n"
                b"from harnyx_miner_sdk.query import Query, Response\n"
                b'@entrypoint("query")\n'
                b"async def query(query: Query) -> Response:\n"
                b'    return Response(text=\"champion\")\n'
            ).decode("ascii"),
        },
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
        target_scores=(0.25000024, 0.25000024),
        champion_scores=(0.25000019, 0.25000019),
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(lambda *, progress_reporter=None: _bind_progress(runtime, progress_reporter)),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    report = json.loads((tmp_path / f"local-eval-report-{batch_id}-vs-champion.json").read_text(encoding="utf-8"))
    head_to_head = report["local_result_summary"]["head_to_head"]

    assert head_to_head["winner_by_total_score"] == "target"
    assert head_to_head["target_total_score"] == 0.5
    assert head_to_head["champion_total_score"] == 0.5


def test_local_eval_runs_target_and_champion_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (
        _task(uuid4(), "task one"),
        _task(uuid4(), "task two"),
    )
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": base64.b64encode(
                b"from harnyx_miner_sdk.decorators import entrypoint\n"
                b"from harnyx_miner_sdk.query import Query, Response\n"
                b'@entrypoint("query")\n'
                b"async def query(query: Query) -> Response:\n"
                b'    return Response(text="champion")\n'
            ).decode("ascii"),
        },
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
        delay_seconds=0.05,
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(lambda *, progress_reporter=None: _bind_progress(runtime, progress_reporter)),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert runtime.max_in_flight == 2


async def test_local_runtime_executes_target_and_champion_via_sandbox_and_reuses_tool_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    target_artifact = ScriptArtifactSpec(
        uid=3,
        artifact_id=uuid4(),
        content_hash="target-hash",
        size_bytes=64,
    )
    champion_artifact = ScriptArtifactSpec(
        uid=2,
        artifact_id=uuid4(),
        content_hash="champion-hash",
        size_bytes=64,
    )
    tasks = (_task(uuid4(), "solo task"),)
    runner = _CapturingRunner(
        results=[
            ArtifactEvaluationOutcome(
                submissions=(
                    _submission(
                        batch_id=batch_id,
                        artifact=target_artifact,
                        task=tasks[0],
                        score=0.9,
                        answer_text="target",
                    ),
                ),
                unresolved_tasks=(),
                timeout_observations_by_pair={},
            ),
            ArtifactEvaluationOutcome(
                submissions=(
                    _submission(
                        batch_id=batch_id,
                        artifact=champion_artifact,
                        task=tasks[0],
                        score=0.6,
                        answer_text="champion",
                    ),
                ),
                unresolved_tasks=(),
                timeout_observations_by_pair={},
            ),
        ]
    )
    sandbox_manager = _FakeSandboxManager()
    tool_host = _FakeToolHost()
    start_calls = 0

    async def _start_tool_host(
        *,
        tool_executor,
        token_semaphore,
    ) -> _FakeToolHost:
        nonlocal start_calls
        del tool_executor, token_semaphore
        await asyncio.sleep(0.01)
        start_calls += 1
        return tool_host

    monkeypatch.setattr(local_eval, "start_local_tool_host", _start_tool_host)
    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda *_args, **_kwargs: pytest.fail("local eval should not execute artifact code via host runpy"),
    )

    runtime = local_eval.LocalEvaluationRuntime(
        settings=cast(
            Any,
            SimpleNamespace(
                sandbox=SimpleNamespace(
                    sandbox_image="local/harnyx-sandbox:0.1.0-dev",
                    sandbox_pull_policy="missing",
                )
            ),
        ),
        tool_executor=cast(Any, _UnusedToolExecutor()),
        scoring_service=cast(Any, object()),
        scoring_config=EvaluationScoringConfig(
            provider="chutes",
            model="openai/gpt-oss-120b-TEE",
            timeout_seconds=30.0,
        ),
        _runner=cast(Any, runner),
        _state=SimpleNamespace(
            session_registry=object(),
            token_registry=object(),
            receipt_log=object(),
            session_manager=object(),
            token_semaphore=object(),
        ),
        _search_client=_FakeAsyncResource(),
        _tool_llm_provider=_FakeAsyncResource(),
        _scoring_llm_provider=_FakeAsyncResource(),
        _scoring_embedding_client=_FakeAsyncResource(),
        _sandbox_manager=cast(Any, sandbox_manager),
        _tool_host=None,
        _tool_host_lock=asyncio.Lock(),
        _progress_reporter=None,
    )

    await asyncio.gather(
        runtime.evaluate_artifact(
            artifact_label="target",
            agent_source=b"from harnyx_miner_sdk.decorators import entrypoint\n",
            artifact=target_artifact,
            batch_id=batch_id,
            tasks=tasks,
        ),
        runtime.evaluate_artifact(
            artifact_label="champion",
            agent_source=b"from harnyx_miner_sdk.decorators import entrypoint\n",
            artifact=champion_artifact,
            batch_id=batch_id,
            tasks=tasks,
        ),
    )
    await runtime.aclose()

    assert start_calls == 1
    assert tool_host.close_calls == 1
    assert len(sandbox_manager.started_options) == 2
    assert len(sandbox_manager.stopped_deployments) == 2
    assert [call["sandbox_client"] for call in runner.calls] == sandbox_manager.clients
    assert sandbox_manager.mount_paths_exist == [True, True]
    for options in sandbox_manager.started_options:
        assert options.host_port == 0
        assert options.network is None
        assert options.host_container_url == tool_host.host_container_url
        assert options.env["AGENT_PATH"].endswith("/agent.py")
        assert options.volumes[0][1] == DEFAULT_STATE_DIR
        assert options.volumes[0][2] == "ro"


def test_local_eval_does_not_write_reports_when_champion_outcome_has_artifact_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (_task(uuid4(), "solo task"),)
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": base64.b64encode(
                b"from harnyx_miner_sdk.decorators import entrypoint\n"
                b"from harnyx_miner_sdk.query import Query, Response\n"
                b'@entrypoint("query")\n'
                b"async def query(query: Query) -> Response:\n"
                b'    return Response(text="champion")\n'
            ).decode("ascii"),
        },
    )
    champion_failure = ArtifactFailure(
        error_code="sandbox_invocation_failed",
        message="artifact breaker tripped",
        failure_detail=ValidatorBatchFailureDetail(
            error_code="sandbox_invocation_failed",
            error_message="artifact breaker tripped",
            occurred_at=datetime.now(UTC),
            artifact_id=champion_artifact_id,
            uid=2,
        ),
        artifact_breaker_tripped=True,
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
        target_outcome=ArtifactEvaluationOutcome(
            submissions=(
                _submission(
                    batch_id=batch_id,
                    artifact=ScriptArtifactSpec(
                        uid=3,
                        artifact_id=uuid4(),
                        content_hash="target-hash",
                        size_bytes=64,
                    ),
                    task=tasks[0],
                    score=0.9,
                    answer_text="target",
                ),
            ),
            unresolved_tasks=(),
            timeout_observations_by_pair={},
        ),
        champion_outcome=ArtifactEvaluationOutcome(
            submissions=(),
            unresolved_tasks=(tasks[0],),
            timeout_observations_by_pair={},
            artifact_failure=champion_failure,
        ),
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(lambda *, progress_reporter=None: _bind_progress(runtime, progress_reporter)),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    with pytest.raises(SystemExit, match="sandbox_invocation_failed"):
        local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert not (tmp_path / f"local-eval-report-{batch_id}-vs-champion.json").exists()
    assert not (tmp_path / f"local-eval-report-{batch_id}-vs-champion.md").exists()


async def test_local_runtime_stops_started_sandbox_when_cancelled_during_startup(
) -> None:
    batch_id = uuid4()
    artifact = ScriptArtifactSpec(
        uid=3,
        artifact_id=uuid4(),
        content_hash="target-hash",
        size_bytes=64,
    )
    tasks = (_task(uuid4(), "solo task"),)
    sandbox_manager = _BlockingSandboxManager()
    runtime = local_eval.LocalEvaluationRuntime(
        settings=cast(
            Any,
            SimpleNamespace(
                sandbox=SimpleNamespace(
                    sandbox_image="local/harnyx-sandbox:0.1.0-dev",
                    sandbox_pull_policy="missing",
                )
            ),
        ),
        tool_executor=cast(Any, _UnusedToolExecutor()),
        scoring_service=cast(Any, object()),
        scoring_config=EvaluationScoringConfig(
            provider="chutes",
            model="openai/gpt-oss-120b-TEE",
            timeout_seconds=30.0,
        ),
        _runner=cast(Any, object()),
        _state=SimpleNamespace(
            session_registry=object(),
            token_registry=object(),
            receipt_log=object(),
            session_manager=object(),
            token_semaphore=object(),
        ),
        _search_client=None,
        _tool_llm_provider=None,
        _scoring_llm_provider=None,
        _scoring_embedding_client=None,
        _sandbox_manager=cast(Any, sandbox_manager),
        _tool_host=cast(Any, _FakeToolHost()),
        _tool_host_lock=asyncio.Lock(),
        _progress_reporter=None,
    )

    task = asyncio.create_task(
        runtime.evaluate_artifact(
            artifact_label="target",
            agent_source=b"from harnyx_miner_sdk.decorators import entrypoint\n",
            artifact=artifact,
            batch_id=batch_id,
            tasks=tasks,
        )
    )
    assert await asyncio.to_thread(sandbox_manager.start_entered.wait, 1.0)

    task.cancel()
    sandbox_manager.release_start.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sandbox_manager.returned_deployment is not None
    assert sandbox_manager.stopped_deployments == [sandbox_manager.returned_deployment]


def test_local_eval_vs_champion_fails_before_runtime_when_champion_script_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (_task(uuid4(), "solo task"),)
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="explicit",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": "",
        },
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
    )
    created = False

    def _create_runtime(*, progress_reporter=None) -> _FakeRuntime:
        nonlocal created
        created = True
        runtime.progress_reporter = progress_reporter
        return runtime

    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(local_eval.LocalEvaluationRuntime, "create", staticmethod(_create_runtime))
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    with pytest.raises(SystemExit, match="missing content_b64"):
        local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert created is False
    assert monitoring.script_calls == 1
    assert runtime.calls == []
    assert monitoring.closed is True


def test_local_eval_vs_champion_preflight_does_not_execute_fetched_champion_code_on_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (_task(uuid4(), "solo task"),)
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": base64.b64encode(b'raise RuntimeError("host execution must not happen")\n').decode("ascii"),
        },
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(lambda *, progress_reporter=None: _bind_progress(runtime, progress_reporter)),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")
    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda *_args, **_kwargs: pytest.fail("champion preflight should not execute code on the host"),
    )

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert monitoring.script_calls == 1
    assert len(runtime.calls) == 2


def test_local_eval_logs_progress_to_stderr_and_keeps_stdout_json_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (
        _task(uuid4(), "task one"),
        _task(uuid4(), "task two"),
    )
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    results = _recorded_rows(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=SelectedBatchContext(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            results=results,
        ),
        champion_script={
            "uid": 2,
            "artifact_id": str(champion_artifact_id),
            "content_hash": "champion-hash",
            "size_bytes": 128,
            "content_b64": base64.b64encode(
                b"from harnyx_miner_sdk.decorators import entrypoint\n"
                b"from harnyx_miner_sdk.query import Query, Response\n"
                b'@entrypoint("query")\n'
                b"async def query(query: Query) -> Response:\n"
                b'    return Response(text="champion")\n'
            ).decode("ascii"),
        },
    )
    runtime = _FakeRuntime(
        batch_id=batch_id,
        champion_artifact_id=champion_artifact_id,
        tasks=tasks,
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(lambda *, progress_reporter=None: _bind_progress(runtime, progress_reporter)),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)

    assert stdout_payload["batch_id"] == str(batch_id)
    assert stdout_payload["mode"] == "vs-champion"
    assert "[local-eval] resolving batch context" in captured.err
    assert "[local-eval] running target and champion evaluations concurrently" in captured.err
    assert "[local-eval] target task 1/2 complete" in captured.err
    assert "[local-eval] finished champion evaluation" in captured.err
    assert "[local-eval] reports written:" in captured.err


def test_platform_monitoring_client_pages_until_completed_batch() -> None:
    first_before = "2026-03-27T06:00:00Z"
    completed_batch_id = uuid4()

    class _StubClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def get(self, path: str, params=None):
            self.calls.append((path, params))
            request = httpx.Request("GET", f"https://platform.example.com{path}")
            if len(self.calls) == 1:
                return httpx.Response(
                    200,
                    json={
                        "batches": (
                            {
                                "batch_id": str(uuid4()),
                                "status": "processing",
                            },
                        ),
                        "next_before": first_before,
                    },
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "batches": (
                        {
                            "batch_id": str(completed_batch_id),
                            "status": "completed",
                        },
                    ),
                    "next_before": None,
                },
                request=request,
            )

        def close(self) -> None:
            return None

    client = PlatformMonitoringClient(base_url="https://platform.example.com")
    client._client.close()
    client._client = _StubClient()

    batch = client.find_latest_completed_batch()

    assert batch["batch_id"] == str(completed_batch_id)
    assert client._client.calls == [
        ("/v1/monitoring/miner-task-batches", {"limit": 100}),
        ("/v1/monitoring/miner-task-batches", {"limit": 100, "before": first_before}),
    ]


def _bind_progress(runtime: _FakeRuntime, progress_reporter: Any) -> _FakeRuntime:
    runtime.progress_reporter = progress_reporter
    return runtime


def test_resolve_batch_context_rejects_explicit_non_completed_batch() -> None:
    batch_id = uuid4()

    class _StubClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def get(self, path: str, params=None):
            self.calls.append((path, params))
            request = httpx.Request("GET", f"https://platform.example.com{path}")
            if path == f"/v1/monitoring/miner-task-batches/{batch_id}":
                return httpx.Response(
                    200,
                    json={
                        "summary": {
                            "batch_id": str(batch_id),
                            "status": "initializing",
                        }
                    },
                    request=request,
                )
            pytest.fail(f"unexpected path: {path}")

        def close(self) -> None:
            return None

    client = PlatformMonitoringClient(base_url="https://platform.example.com")
    client._client.close()
    client._client = _StubClient()

    with pytest.raises(
        RuntimeError,
        match=rf"miner-task batch {batch_id} is not completed \(status=initializing\)",
    ):
        client.resolve_batch_context(batch_id)

    assert client._client.calls == [
        (f"/v1/monitoring/miner-task-batches/{batch_id}", None),
    ]
