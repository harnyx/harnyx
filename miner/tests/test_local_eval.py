from __future__ import annotations

import asyncio
import base64
import json
import logging
import runpy
import stat
import threading
from collections import Counter
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
from harnyx_commons.miner_task_scoring import EvaluationScoringConfig
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.manager import SandboxDeployment
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.state import DEFAULT_STATE_DIR
from harnyx_miner import local_eval
from harnyx_miner.platform_monitoring import (
    PlatformMonitoringClient,
    PlatformMonitoringRequestError,
    RecordedBatchResultsSnapshot,
    RecordedResultsError,
    RecordedResultsScope,
    SelectedBatchContext,
)
from harnyx_miner_sdk.json_types import JsonValue
from harnyx_validator.application.dto.evaluation import MinerTaskRunSubmission, ScriptArtifactSpec, TokenUsageSummary
from harnyx_validator.application.services.evaluation_runner import (
    ArtifactEvaluationOutcome,
    ArtifactFailure,
    ValidatorBatchFailureDetail,
)
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


def _artifact_failure(artifact: ScriptArtifactSpec) -> ArtifactFailure:
    return ArtifactFailure(
        error_code="sandbox_invocation_failed",
        message="artifact failed",
        failure_detail=ValidatorBatchFailureDetail(
            error_code="sandbox_invocation_failed",
            error_message="artifact failed",
            occurred_at=datetime.now(UTC),
            artifact_id=artifact.artifact_id,
            uid=artifact.uid,
        ),
        artifact_breaker_tripped=True,
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


def _recorded_results_snapshot(
    *,
    batch_id: UUID,
    champion_artifact_id: UUID,
    rows: tuple[dict[str, object], ...],
) -> RecordedBatchResultsSnapshot:
    return RecordedBatchResultsSnapshot(
        rows=rows,
        error=None,
        scope=RecordedResultsScope(batch_id=batch_id, artifact_id=champion_artifact_id),
    )


def _unavailable_recorded_results_snapshot(*, path: str) -> RecordedBatchResultsSnapshot:
    return RecordedBatchResultsSnapshot(
        rows=None,
        error=RecordedResultsError.from_request_error(
            PlatformMonitoringRequestError(
                path=path,
                status_code=503,
                detail="upstream connect error",
            )
        ),
        scope=None,
    )


def _request_error_recorded_results_snapshot(*, path: str) -> RecordedBatchResultsSnapshot:
    return RecordedBatchResultsSnapshot(
        rows=None,
        error=RecordedResultsError.from_request_error(
            PlatformMonitoringRequestError(
                path=path,
                status_code=0,
                detail="connection terminated",
            )
        ),
        scope=None,
    )


def _selected_batch_context(
    *,
    batch_id,
    source: str,
    detail: dict[str, object],
    recorded_results: RecordedBatchResultsSnapshot,
) -> SelectedBatchContext:
    return SelectedBatchContext(
        batch_id=batch_id,
        source=source,
        detail=detail,
        recorded_results=recorded_results,
    )


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


class _RaisingMonitoringClient:
    def __init__(self, *, error: Exception) -> None:
        self.error = error
        self.closed = False

    def resolve_batch_context(self, batch_id) -> SelectedBatchContext:
        del batch_id
        raise self.error

    def get_script(self, artifact_id) -> dict[str, object]:
        del artifact_id
        pytest.fail("champion script fetch should not be reached")

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


class _FailingAsyncResource(_FakeAsyncResource):
    async def aclose(self) -> None:
        self.closed = True
        raise RuntimeError("close failed")


def test_local_eval_runtime_create_binds_sandbox_publish_to_loopback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    scoring_llm_provider = _FakeAsyncResource()

    class _FakeRegistry(_FakeAsyncResource):
        def resolve(self, name: str) -> _FakeAsyncResource:
            assert name == "chutes"
            return scoring_llm_provider

    settings = SimpleNamespace(
        llm=object(),
        bedrock=object(),
        vertex=object(),
        sandbox=SimpleNamespace(
            sandbox_image="local/harnyx-sandbox:0.1.0-dev",
            sandbox_pull_policy="missing",
        ),
    )

    def create_sandbox_manager(**kwargs: object) -> _FakeSandboxManager:
        captured.update(kwargs)
        return _FakeSandboxManager()

    monkeypatch.setattr(local_eval.Settings, "load", staticmethod(lambda: settings))
    monkeypatch.setattr(
        local_eval,
        "_build_state",
        lambda *_args, **_kwargs: _minimal_local_eval_state(tmp_path),
    )
    monkeypatch.setattr(
        local_eval,
        "build_tool_invocation_clients",
        lambda **_kwargs: SimpleNamespace(
            search_client=_FakeAsyncResource(),
            llm_provider_registry=_FakeRegistry(),
            tool_llm_provider=_FakeAsyncResource(),
        ),
    )
    monkeypatch.setattr(
        local_eval,
        "_resolve_scoring_judge_route",
        lambda _settings: SimpleNamespace(provider="chutes"),
    )
    monkeypatch.setattr(local_eval, "_build_tooling", lambda **_: (object(), _UnusedToolExecutor()))
    monkeypatch.setattr(
        local_eval,
        "_create_scoring_service",
        lambda *_args, **_kwargs: SimpleNamespace(
            _config=EvaluationScoringConfig(
                provider="chutes",
                model="openai/gpt-oss-120b-TEE",
                timeout_seconds=30.0,
            )
        ),
    )
    monkeypatch.setattr(local_eval, "create_sandbox_manager", create_sandbox_manager)

    runtime = local_eval.LocalEvaluationRuntime.create(
        run_progress_root=tmp_path / "run-progress",
        progress_reporter=None,
    )

    assert captured["host"] == "127.0.0.1"
    assert captured["published_port_bind_host"] == "127.0.0.1"
    assert runtime._sandbox_manager is not None


async def test_local_runtime_closes_llm_provider_registry_not_routed_wrappers() -> None:
    search_client = _FakeAsyncResource()
    llm_provider_registry = _FakeAsyncResource()
    tool_llm_provider = _FakeAsyncResource()
    scoring_llm_provider = _FakeAsyncResource()
    runtime = local_eval.LocalEvaluationRuntime(
        settings=cast(Any, SimpleNamespace()),
        tool_executor=cast(Any, _UnusedToolExecutor()),
        scoring_service=cast(Any, object()),
        scoring_config=EvaluationScoringConfig(
            provider="chutes",
            model="openai/gpt-oss-120b-TEE",
            timeout_seconds=30.0,
        ),
        _runner=cast(Any, object()),
        _state=SimpleNamespace(),
        _search_client=search_client,
        _llm_provider_registry=llm_provider_registry,
        _tool_llm_provider=tool_llm_provider,
        _scoring_llm_provider=scoring_llm_provider,
        _sandbox_manager=cast(Any, object()),
        _tool_host=None,
        _tool_host_lock=asyncio.Lock(),
        _run_id=uuid4().hex,
        _progress_reporter=None,
    )

    await runtime.aclose()

    assert search_client.closed is True
    assert llm_provider_registry.closed is True
    assert tool_llm_provider.closed is False
    assert scoring_llm_provider.closed is False


async def test_local_runtime_closes_llm_provider_registry_when_search_close_fails() -> None:
    search_client = _FailingAsyncResource()
    llm_provider_registry = _FakeAsyncResource()
    runtime = local_eval.LocalEvaluationRuntime(
        settings=cast(Any, SimpleNamespace()),
        tool_executor=cast(Any, _UnusedToolExecutor()),
        scoring_service=cast(Any, object()),
        scoring_config=EvaluationScoringConfig(
            provider="chutes",
            model="openai/gpt-oss-120b-TEE",
            timeout_seconds=30.0,
        ),
        _runner=cast(Any, object()),
        _state=SimpleNamespace(),
        _search_client=search_client,
        _llm_provider_registry=llm_provider_registry,
        _tool_llm_provider=_FakeAsyncResource(),
        _scoring_llm_provider=_FakeAsyncResource(),
        _sandbox_manager=cast(Any, object()),
        _tool_host=None,
        _tool_host_lock=asyncio.Lock(),
        _run_id=uuid4().hex,
        _progress_reporter=None,
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await runtime.aclose()

    assert search_client.closed is True
    assert llm_provider_registry.closed is True


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
        raise AssertionError(f"sandbox client invoke should not be reached in this unit test: entrypoint={entrypoint}")

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


class _FailingSandboxManager(_FakeSandboxManager):
    def __init__(self, error: RuntimeError) -> None:
        super().__init__()
        self._error = error

    def start(self, options: SandboxOptions) -> SandboxDeployment:
        self.started_options.append(options)
        self.mount_paths_exist.append(Path(options.volumes[0][0]).exists())
        raise self._error


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


class _CapturingProgress:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)

    def begin_artifact(
        self,
        *,
        label: str,
        artifact: ScriptArtifactSpec,
        task_count: int,
    ) -> None:
        del label, artifact, task_count

    def finish_artifact(
        self,
        *,
        label: str,
        artifact: ScriptArtifactSpec,
        submissions: Sequence[MinerTaskRunSubmission],
    ) -> None:
        del label, artifact, submissions


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


def _local_runtime(
    *,
    runner: object,
    sandbox_manager: object,
    tool_host: object | None = None,
    progress: object | None = None,
) -> local_eval.LocalEvaluationRuntime:
    return local_eval.LocalEvaluationRuntime(
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
            tool_concurrency_limiter=object(),
        ),
        _search_client=_FakeAsyncResource(),
        _llm_provider_registry=_FakeAsyncResource(),
        _tool_llm_provider=_FakeAsyncResource(),
        _scoring_llm_provider=_FakeAsyncResource(),
        _sandbox_manager=cast(Any, sandbox_manager),
        _tool_host=cast(Any, tool_host),
        _tool_host_lock=asyncio.Lock(),
        _run_id=uuid4().hex,
        _progress_reporter=cast(Any, progress),
    )


def _single_failure_context(root: Path) -> dict[str, object]:
    contexts = list(root.rglob("local-eval-context.json"))
    assert len(contexts) == 1
    return json.loads(contexts[0].read_text(encoding="utf-8"))


def _single_failure_agent(root: Path) -> bytes:
    agents = list(root.rglob("agent.py"))
    assert len(agents) == 1
    return agents[0].read_bytes()


def _minimal_local_eval_state(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        session_registry=object(),
        token_registry=object(),
        receipt_log=object(),
        session_manager=object(),
        evaluation_records=object(),
        progress_tracker=local_eval.FileBackedRunProgress(storage_root=tmp_path / "run-progress"),
        tool_concurrency_limiter=object(),
    )


def _precreate_public_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old public content", encoding="utf-8")
    path.chmod(0o644)


def _assert_private_mode(path: Path, expected_mode: int) -> None:
    assert stat.S_IMODE(path.stat().st_mode) == expected_mode


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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
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
    assert report["evaluation_config"]["artifact_task_parallelism"] == 20
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
    assert report["recorded_platform_context"]["results_status"]["state"] == "available"
    assert report["recorded_platform_context"]["results_scope"] == {
        "kind": "artifact",
        "batch_id": str(batch_id),
        "artifact_id": str(champion_artifact_id),
    }
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


def test_invocation_only_runtime_factory_skips_default_scoring_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = cast(
        Any,
        SimpleNamespace(
            llm=object(),
            bedrock=object(),
            vertex=object(),
        ),
    )
    state = SimpleNamespace(
        session_manager=object(),
        evaluation_records=object(),
        receipt_log=object(),
        progress_tracker=local_eval.FileBackedRunProgress(storage_root=tmp_path / "run-progress"),
    )
    scoring_service = cast(Any, object())
    scoring_config = EvaluationScoringConfig(
        provider="chutes",
        model="benchmark-invocation-only",
        scoring_version="benchmark-invocation-only",
    )

    monkeypatch.setattr(local_eval.Settings, "load", staticmethod(lambda: settings))
    monkeypatch.setattr(local_eval, "_build_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(
        local_eval,
        "build_tool_invocation_clients",
        lambda **_kwargs: SimpleNamespace(
            search_client=None,
            llm_provider_registry=object(),
            tool_llm_provider=None,
        ),
    )
    monkeypatch.setattr(local_eval, "_build_tooling", lambda **_kwargs: (object(), object()))
    monkeypatch.setattr(local_eval, "create_sandbox_manager", lambda **_kwargs: object())

    runtime = local_eval.LocalEvaluationRuntime.create_invocation_only(
        scoring_service=scoring_service,
        scoring_config=scoring_config,
        run_progress_root=tmp_path / "run-progress",
    )

    assert runtime.settings is settings
    assert runtime.scoring_service is scoring_service
    assert runtime.scoring_config is scoring_config
    assert runtime._scoring_llm_provider is None


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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="explicit",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
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
    assert report["recorded_platform_context"]["results_status"]["state"] == "available"
    assert report["recorded_platform_context"]["results_scope"] == {
        "kind": "artifact",
        "batch_id": str(batch_id),
        "artifact_id": str(champion_artifact_id),
    }
    assert len(report["recorded_platform_context"]["results"]) == 1
    assert len(runtime.calls) == 1


def test_local_eval_target_only_continues_when_recorded_results_fetch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (_task(uuid4(), "solo task"),)
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="explicit",
            detail=detail,
            recorded_results=_unavailable_recorded_results_snapshot(
                path=(f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{champion_artifact_id}/results")
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
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

    report = json.loads((tmp_path / f"local-eval-report-{batch_id}-target-only.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / f"local-eval-report-{batch_id}-target-only.md").read_text(encoding="utf-8")
    captured = capsys.readouterr()

    assert report["recorded_platform_context"]["results"] is None
    assert report["recorded_platform_context"]["results_status"] == {
        "state": "unavailable",
        "error": {
            "path": (f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{champion_artifact_id}/results"),
            "status_code": 503,
            "detail": "upstream connect error",
        },
    }
    assert report["recorded_platform_context"]["results_scope"] is None
    assert report["tasks"][0]["recorded_platform_rows"] is None
    assert "recorded platform results unavailable" in captured.err
    assert "Recorded monitoring rows were unavailable for this run" in markdown
    assert monitoring.script_calls == 0
    assert len(runtime.calls) == 1


def test_local_eval_vs_champion_continues_when_recorded_results_fetch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (_task(uuid4(), "solo task"),)
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            recorded_results=_unavailable_recorded_results_snapshot(
                path=(f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{champion_artifact_id}/results")
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    report = json.loads((tmp_path / f"local-eval-report-{batch_id}-vs-champion.json").read_text(encoding="utf-8"))

    assert monitoring.script_calls == 1
    assert report["recorded_platform_context"]["results"] is None
    assert report["recorded_platform_context"]["results_status"]["state"] == "unavailable"
    assert report["recorded_platform_context"]["results_scope"] is None
    assert report["tasks"][0]["recorded_platform_rows"] is None
    assert len(runtime.calls) == 2


def test_local_eval_target_only_continues_when_recorded_results_transport_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    tasks = (_task(uuid4(), "solo task"),)
    detail = _batch_detail(batch_id=batch_id, champion_artifact_id=champion_artifact_id, tasks=tasks)
    monitoring = _FakeMonitoringClient(
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="explicit",
            detail=detail,
            recorded_results=_request_error_recorded_results_snapshot(
                path=(f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{champion_artifact_id}/results")
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
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

    report = json.loads((tmp_path / f"local-eval-report-{batch_id}-target-only.json").read_text(encoding="utf-8"))

    assert report["recorded_platform_context"]["results"] is None
    assert report["recorded_platform_context"]["results_status"] == {
        "state": "unavailable",
        "error": {
            "path": (f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{champion_artifact_id}/results"),
            "status_code": 0,
            "detail": "connection terminated",
        },
    }
    assert report["recorded_platform_context"]["results_scope"] is None
    assert report["tasks"][0]["recorded_platform_rows"] is None
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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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
        target_scores=(0.25000024, 0.25000024),
        champion_scores=(0.25000019, 0.25000019),
    )
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(
        local_eval.LocalEvaluationRuntime,
        "create",
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert runtime.max_in_flight == 2


async def test_local_runtime_executes_target_and_champion_via_sandbox_and_reuses_tool_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_eval, "_LOCAL_EVAL_FAILURE_ROOT", tmp_path)
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
    tool_concurrency_limiter = object()
    captured_tool_concurrency_limiter: object | None = None

    async def _start_tool_host(
        *,
        tool_executor,
        tool_concurrency_limiter,
    ) -> _FakeToolHost:
        nonlocal captured_tool_concurrency_limiter, start_calls
        del tool_executor
        captured_tool_concurrency_limiter = tool_concurrency_limiter
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
            tool_concurrency_limiter=tool_concurrency_limiter,
        ),
        _search_client=_FakeAsyncResource(),
        _llm_provider_registry=_FakeAsyncResource(),
        _tool_llm_provider=_FakeAsyncResource(),
        _scoring_llm_provider=_FakeAsyncResource(),
        _sandbox_manager=cast(Any, sandbox_manager),
        _tool_host=None,
        _tool_host_lock=asyncio.Lock(),
        _run_id=uuid4().hex,
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
    assert captured_tool_concurrency_limiter is tool_concurrency_limiter
    assert tool_host.close_calls == 1
    assert len(sandbox_manager.started_options) == 2
    assert len(sandbox_manager.stopped_deployments) == 2
    assert Counter(id(call["sandbox_client"]) for call in runner.calls) == Counter(
        id(client) for client in sandbox_manager.clients
    )
    assert sandbox_manager.mount_paths_exist == [True, True]
    for options in sandbox_manager.started_options:
        assert options.host_port == 0
        assert options.network is None
        assert options.failure_diagnostics_dir is not None
        _assert_private_mode(Path(options.failure_diagnostics_dir), 0o700)
        assert options.host_container_url == tool_host.host_container_url
        assert options.env["AGENT_PATH"].endswith("/agent.py")
        assert options.volumes[0][1] == DEFAULT_STATE_DIR
        assert options.volumes[0][2] == "ro"


async def test_local_eval_writes_failure_bundle_when_sandbox_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_eval, "_LOCAL_EVAL_FAILURE_ROOT", tmp_path)
    batch_id = uuid4()
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="target-hash", size_bytes=64)
    original_error = RuntimeError("sandbox start exploded")
    progress = _CapturingProgress()
    sandbox_manager = _FailingSandboxManager(original_error)
    runtime = _local_runtime(
        runner=object(),
        sandbox_manager=sandbox_manager,
        tool_host=_FakeToolHost(),
        progress=progress,
    )
    failure_dir = tmp_path / runtime._run_id / f"target-{artifact.artifact_id.hex[:12]}"
    _precreate_public_file(failure_dir / "agent.py")
    _precreate_public_file(failure_dir / "local-eval-context.json")

    with pytest.raises(RuntimeError, match="sandbox start exploded"):
        await runtime.evaluate_artifact(
            artifact_label="target",
            agent_source=b"print('target')\n",
            artifact=artifact,
            batch_id=batch_id,
            tasks=(_task(uuid4(), "solo task"),),
        )

    context = _single_failure_context(tmp_path)
    assert context["failure_category"] == "sandbox_startup"
    assert context["error_type"] == "RuntimeError"
    assert context["error_message"] == "sandbox start exploded"
    assert _single_failure_agent(tmp_path) == b"print('target')\n"
    _assert_private_mode(tmp_path, 0o700)
    _assert_private_mode(tmp_path / runtime._run_id, 0o700)
    _assert_private_mode(failure_dir, 0o700)
    _assert_private_mode(failure_dir / "agent.py", 0o600)
    _assert_private_mode(failure_dir / "local-eval-context.json", 0o600)
    assert "failure category: sandbox_startup" in progress.messages
    assert any(message.startswith("failure bundle:") for message in progress.messages)


async def test_local_eval_writes_failure_bundle_when_artifact_outcome_has_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_eval, "_LOCAL_EVAL_FAILURE_ROOT", tmp_path)
    batch_id = uuid4()
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="target-hash", size_bytes=64)
    failure = _artifact_failure(artifact)
    progress = _CapturingProgress()
    sandbox_manager = _FakeSandboxManager()
    runtime = _local_runtime(
        runner=_CapturingRunner(
            results=[
                ArtifactEvaluationOutcome(
                    submissions=(),
                    unresolved_tasks=(),
                    timeout_observations_by_pair={},
                    artifact_failure=failure,
                )
            ]
        ),
        sandbox_manager=sandbox_manager,
        tool_host=_FakeToolHost(),
        progress=progress,
    )

    outcome = await runtime.evaluate_artifact(
        artifact_label="target",
        agent_source=b"print('target')\n",
        artifact=artifact,
        batch_id=batch_id,
        tasks=(_task(uuid4(), "solo task"),),
    )

    assert outcome.artifact_failure is failure
    context = _single_failure_context(tmp_path)
    assert context["failure_category"] == "evaluated_application"
    assert context["error_type"] == "ArtifactFailure"
    assert context["error_code"] == "sandbox_invocation_failed"
    assert _single_failure_agent(tmp_path) == b"print('target')\n"
    assert len(sandbox_manager.stopped_deployments) == 1
    assert "failure category: evaluated_application" in progress.messages
    assert any(message.startswith("failure bundle:") for message in progress.messages)


async def test_local_eval_failure_bundle_write_failure_preserves_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_eval, "_LOCAL_EVAL_FAILURE_ROOT", tmp_path)

    def fail_bundle_write(**kwargs: object) -> None:
        del kwargs
        raise OSError("disk full")

    monkeypatch.setattr(local_eval, "_write_local_failure_bundle", fail_bundle_write)
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="target-hash", size_bytes=64)
    progress = _CapturingProgress()
    runtime = _local_runtime(
        runner=object(),
        sandbox_manager=_FailingSandboxManager(RuntimeError("original sandbox failure")),
        tool_host=_FakeToolHost(),
        progress=progress,
    )

    with pytest.raises(RuntimeError, match="original sandbox failure"):
        await runtime.evaluate_artifact(
            artifact_label="target",
            agent_source=b"print('target')\n",
            artifact=artifact,
            batch_id=uuid4(),
            tasks=(_task(uuid4(), "solo task"),),
        )

    failure_dir = tmp_path / runtime._run_id / f"target-{artifact.artifact_id.hex[:12]}"
    assert progress.messages == [f"failure bundle write failed: path={failure_dir} error=disk full"]


async def test_local_eval_sandbox_startup_failure_is_not_recategorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_eval, "_LOCAL_EVAL_FAILURE_ROOT", tmp_path)
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="target-hash", size_bytes=64)
    progress = _CapturingProgress()
    runtime = _local_runtime(
        runner=object(),
        sandbox_manager=_FailingSandboxManager(RuntimeError("sandbox failed")),
        tool_host=_FakeToolHost(),
        progress=progress,
    )

    with pytest.raises(RuntimeError, match="sandbox failed"):
        await runtime.evaluate_artifact(
            artifact_label="target",
            agent_source=b"print('target')\n",
            artifact=artifact,
            batch_id=uuid4(),
            tasks=(_task(uuid4(), "solo task"),),
        )

    contexts = list(tmp_path.rglob("local-eval-context.json"))
    assert len(contexts) == 1
    context = json.loads(contexts[0].read_text(encoding="utf-8"))
    assert context["failure_category"] == "sandbox_startup"
    assert "failure category: local_eval_runtime" not in progress.messages


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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
    )
    monkeypatch.setattr(local_eval, "platform_base_url_from_env", lambda: "https://platform.example.com")

    with pytest.raises(SystemExit, match="sandbox_invocation_failed"):
        local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert not (tmp_path / f"local-eval-report-{batch_id}-vs-champion.json").exists()
    assert not (tmp_path / f"local-eval-report-{batch_id}-vs-champion.md").exists()


async def test_local_runtime_stops_started_sandbox_when_cancelled_during_startup() -> None:
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
            tool_concurrency_limiter=object(),
        ),
        _search_client=None,
        _llm_provider_registry=None,
        _tool_llm_provider=None,
        _scoring_llm_provider=None,
        _sandbox_manager=cast(Any, sandbox_manager),
        _tool_host=cast(Any, _FakeToolHost()),
        _tool_host_lock=asyncio.Lock(),
        _run_id=uuid4().hex,
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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="explicit",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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

    def _create_runtime(*, progress_reporter=None, run_progress_root=None) -> _FakeRuntime:
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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
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
        batch_context=_selected_batch_context(
            batch_id=batch_id,
            source="latest-completed",
            detail=detail,
            recorded_results=_recorded_results_snapshot(
                batch_id=batch_id,
                champion_artifact_id=champion_artifact_id,
                rows=results,
            ),
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
        staticmethod(
            lambda *, progress_reporter=None, run_progress_root=None: _bind_progress(runtime, progress_reporter)
        ),
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


def test_main_configures_cli_logging_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)
    calls: list[str] = []

    async def fake_amain(argv: Sequence[str] | None) -> None:
        calls.append(f"amain:{list(argv or ())}")

    def fake_run(coroutine: Any) -> None:
        calls.append("run")
        coroutine.close()

    monkeypatch.setattr(local_eval, "_configure_cli_logging", lambda: calls.append("configured"))
    monkeypatch.setattr(local_eval, "_amain", fake_amain)
    monkeypatch.setattr(local_eval.asyncio, "run", fake_run)

    local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert calls == ["configured", "run"]


def test_configure_cli_logging_writes_to_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    tools_logger = logging.getLogger("harnyx_commons.tools")
    old_level = root.level
    old_handlers = list(root.handlers)
    old_tools_level = tools_logger.level
    old_tools_disabled = tools_logger.disabled
    old_tools_propagate = tools_logger.propagate
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    tools_logger.setLevel(logging.WARNING)
    tools_logger.disabled = True
    tools_logger.propagate = False

    try:
        local_eval._configure_cli_logging()

        assert root.level == logging.DEBUG
        assert root.handlers
        assert root.handlers[0].stream is local_eval.sys.stderr
        assert tools_logger.isEnabledFor(logging.DEBUG)
        assert not tools_logger.disabled
        assert tools_logger.propagate
    finally:
        root.setLevel(old_level)
        root.handlers = old_handlers
        tools_logger.setLevel(old_tools_level)
        tools_logger.disabled = old_tools_disabled
        tools_logger.propagate = old_tools_propagate


def test_main_reports_invalid_log_level_as_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)
    monkeypatch.setenv("LOG_LEVEL", "NO_SUCH_LEVEL")

    with pytest.raises(SystemExit) as exc_info:
        local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert "Unknown level" in str(exc_info.value)


def test_local_eval_still_fails_before_runtime_when_batch_detail_fetch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    detail_path = f"/v1/monitoring/miner-task-batches/{batch_id}"
    monitoring = _RaisingMonitoringClient(
        error=PlatformMonitoringRequestError(
            path=detail_path,
            status_code=503,
            detail="upstream connect error",
        )
    )
    created = False

    def _create_runtime(*, progress_reporter=None, run_progress_root=None) -> _FakeRuntime:
        nonlocal created
        del progress_reporter
        created = True
        raise AssertionError("runtime should not be created")

    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(local_eval.LocalEvaluationRuntime, "create", staticmethod(_create_runtime))

    with pytest.raises(SystemExit, match=r"platform monitoring request failed \(503\): upstream connect error"):
        local_eval.main(["--agent-path", str(agent_path), "--batch-id", str(batch_id), "--output-dir", str(tmp_path)])

    assert created is False
    assert monitoring.closed is True


def test_local_eval_still_fails_before_runtime_when_latest_batch_lookup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_path = "/v1/monitoring/miner-task-batches"
    monitoring = _RaisingMonitoringClient(
        error=PlatformMonitoringRequestError(
            path=list_path,
            status_code=503,
            detail="upstream connect error",
        )
    )
    created = False

    def _create_runtime(*, progress_reporter=None, run_progress_root=None) -> _FakeRuntime:
        nonlocal created
        del progress_reporter
        created = True
        raise AssertionError("runtime should not be created")

    agent_path = tmp_path / "agent.py"
    _write_agent(agent_path)

    monkeypatch.setattr(local_eval.PlatformMonitoringClient, "from_env", staticmethod(lambda: monitoring))
    monkeypatch.setattr(local_eval.LocalEvaluationRuntime, "create", staticmethod(_create_runtime))

    with pytest.raises(SystemExit, match=r"platform monitoring request failed \(503\): upstream connect error"):
        local_eval.main(["--agent-path", str(agent_path), "--output-dir", str(tmp_path)])

    assert created is False
    assert monitoring.closed is True


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


def test_resolve_batch_context_records_results_failure_without_aborting() -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    results_path = f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{champion_artifact_id}/results"

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
                            "status": "completed",
                            "champion_artifact_id": str(champion_artifact_id),
                        }
                    },
                    request=request,
                )
            if path == results_path:
                return httpx.Response(503, text="upstream connect error", request=request)
            pytest.fail(f"unexpected path: {path}")

        def close(self) -> None:
            return None

    client = PlatformMonitoringClient(base_url="https://platform.example.com")
    client._client.close()
    client._client = _StubClient()

    context = client.resolve_batch_context(batch_id)

    assert context.recorded_results.rows is None
    assert context.recorded_results.error is not None
    assert context.recorded_results.error.path == results_path
    assert context.recorded_results.error.status_code == 503
    assert context.recorded_results.scope is None
    assert client._client.calls == [
        (f"/v1/monitoring/miner-task-batches/{batch_id}", None),
        (results_path, None),
    ]


def test_resolve_batch_context_records_results_transport_failure_without_aborting() -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    results_path = f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{champion_artifact_id}/results"

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
                            "status": "completed",
                            "champion_artifact_id": str(champion_artifact_id),
                        }
                    },
                    request=request,
                )
            if path == results_path:
                raise httpx.ConnectError("connection terminated", request=request)
            pytest.fail(f"unexpected path: {path}")

        def close(self) -> None:
            return None

    client = PlatformMonitoringClient(base_url="https://platform.example.com")
    client._client.close()
    client._client = _StubClient()

    context = client.resolve_batch_context(batch_id)

    assert context.recorded_results.rows is None
    assert context.recorded_results.error is not None
    assert context.recorded_results.error.path == results_path
    assert context.recorded_results.error.status_code == 0
    assert context.recorded_results.error.detail == "connection terminated"
    assert context.recorded_results.scope is None
    assert client._client.calls == [
        (f"/v1/monitoring/miner-task-batches/{batch_id}", None),
        (results_path, None),
    ]


def test_resolve_batch_context_without_champion_marks_recorded_results_unavailable() -> None:
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
                            "status": "completed",
                            "champion_artifact_id": None,
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

    context = client.resolve_batch_context(batch_id)

    assert context.recorded_results.rows is None
    assert context.recorded_results.scope is None
    assert context.recorded_results.error is not None
    assert context.recorded_results.error.path is None
    assert context.recorded_results.error.status_code is None
    assert context.recorded_results.error.detail == (
        f"batch {batch_id} does not expose a champion artifact for recorded context"
    )
    assert client._client.calls == [
        (f"/v1/monitoring/miner-task-batches/{batch_id}", None),
    ]


def test_resolve_batch_context_still_raises_when_latest_batch_lookup_fails() -> None:
    class _StubClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def get(self, path: str, params=None):
            self.calls.append((path, params))
            request = httpx.Request("GET", f"https://platform.example.com{path}")
            if path == "/v1/monitoring/miner-task-batches":
                return httpx.Response(503, text="upstream connect error", request=request)
            pytest.fail(f"unexpected path: {path}")

        def close(self) -> None:
            return None

    client = PlatformMonitoringClient(base_url="https://platform.example.com")
    client._client.close()
    client._client = _StubClient()

    with pytest.raises(PlatformMonitoringRequestError, match=r"platform monitoring request failed \(503\)"):
        client.resolve_batch_context(None)

    assert client._client.calls == [
        ("/v1/monitoring/miner-task-batches", {"limit": 100}),
    ]


def test_resolve_batch_context_still_raises_when_batch_detail_request_fails() -> None:
    batch_id = uuid4()

    class _StubClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def get(self, path: str, params=None):
            self.calls.append((path, params))
            request = httpx.Request("GET", f"https://platform.example.com{path}")
            if path == f"/v1/monitoring/miner-task-batches/{batch_id}":
                return httpx.Response(503, text="upstream connect error", request=request)
            pytest.fail(f"unexpected path: {path}")

        def close(self) -> None:
            return None

    client = PlatformMonitoringClient(base_url="https://platform.example.com")
    client._client.close()
    client._client = _StubClient()

    with pytest.raises(PlatformMonitoringRequestError, match=r"platform monitoring request failed \(503\)"):
        client.resolve_batch_context(batch_id)

    assert client._client.calls == [
        (f"/v1/monitoring/miner-task-batches/{batch_id}", None),
    ]


def test_get_recorded_results_uses_artifact_monitoring_route() -> None:
    batch_id = uuid4()
    champion_artifact_id = uuid4()
    results_path = f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{champion_artifact_id}/results"

    class _StubClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def get(self, path: str, params=None):
            self.calls.append((path, params))
            request = httpx.Request("GET", f"https://platform.example.com{path}")
            if path == results_path:
                return httpx.Response(200, json=[], request=request)
            pytest.fail(f"unexpected path: {path}")

        def close(self) -> None:
            return None

    client = PlatformMonitoringClient(base_url="https://platform.example.com")
    client._client.close()
    client._client = _StubClient()

    rows = client.get_recorded_results(batch_id=batch_id, artifact_id=champion_artifact_id)

    assert rows == ()
    assert client._client.calls == [(results_path, None)]
