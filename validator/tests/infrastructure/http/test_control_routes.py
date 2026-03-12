from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from caster_commons.domain.miner_task import (
    EvaluationDetails,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from caster_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from caster_commons.domain.tool_usage import (
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from caster_validator.application.accept_batch import AcceptEvaluationBatch
from caster_validator.application.dto.evaluation import (
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    TokenUsageSummary,
)
from caster_validator.application.status import StatusProvider
from caster_validator.domain.evaluation import MinerTaskRun
from caster_validator.infrastructure.http.routes import ValidatorControlDeps, add_control_routes
from caster_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from caster_validator.infrastructure.state.run_progress import InMemoryRunProgress, RunProgressSnapshot


def _create_test_app(provider: DemoControlDependencyProvider) -> FastAPI:
    app = FastAPI()
    add_control_routes(app, provider)
    return app


class StubAcceptBatch:
    def __init__(self) -> None:
        self.received_batch: MinerTaskBatchSpec | None = None

    def execute(self, batch: object) -> None:
        if not isinstance(batch, MinerTaskBatchSpec):
            raise AssertionError(f"expected MinerTaskBatchSpec, got {type(batch)!r}")
        self.received_batch = batch
        return None


class StubStatusProvider:
    def snapshot(self) -> dict[str, object]:
        return {"status": "ok"}


class FakeProgressTracker:
    def __init__(self, *, snapshot: RunProgressSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, _: UUID) -> RunProgressSnapshot:
        return self._snapshot


class DemoControlDependencyProvider:
    def __init__(self, *, snapshot: RunProgressSnapshot) -> None:
        self.accept_batch = StubAcceptBatch()
        self._deps = ValidatorControlDeps(
            accept_batch=self.accept_batch,
            status_provider=StubStatusProvider(),
            auth=_allow_all_auth,
            progress_tracker=FakeProgressTracker(snapshot=snapshot),
        )

    def __call__(self) -> ValidatorControlDeps:
        return self._deps


class RealAcceptBatchDependencyProvider:
    def __init__(self) -> None:
        self.inbox = InMemoryBatchInbox()
        self.status_provider = StatusProvider()
        self.progress_tracker = InMemoryRunProgress()
        self.accept_batch = AcceptEvaluationBatch(
            inbox=self.inbox,
            status=self.status_provider,
            progress=self.progress_tracker,
        )
        self._deps = ValidatorControlDeps(
            accept_batch=self.accept_batch,
            status_provider=self.status_provider,
            auth=_allow_all_auth,
            progress_tracker=self.progress_tracker,
        )

    def __call__(self) -> ValidatorControlDeps:
        return self._deps


def _allow_all_auth(_: Request, __: bytes) -> str:
    return "caller"


def _make_task_submission(*, batch_id: UUID) -> tuple[MinerTask, MinerTaskRunSubmission]:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What happened?"),
        reference_answer=ReferenceAnswer(text="The reference answer."),
    )
    total_tool_usage = ToolUsageSummary(
        search_tool=SearchToolUsageSummary(call_count=2, cost=0.005),
        search_tool_cost=0.005,
        llm=LlmUsageSummary(
            call_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.02,
            providers={
                "chutes": {
                    "openai/gpt-oss-20b": LlmModelUsageCost(
                        usage=LlmUsageTotals(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                            call_count=1,
                        ),
                        cost=0.02,
                    )
                }
            },
        ),
        llm_cost=0.02,
    )
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=7,
        artifact_id=uuid4(),
        task_id=task.task_id,
        response=Response(text="The miner answer."),
        details=EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=1.0,
                similarity_score=0.8,
                total_score=0.9,
                scoring_version="v1",
            ),
            total_tool_usage=total_tool_usage,
            elapsed_ms=2500.0,
        ),
        completed_at=datetime.now(UTC),
    )

    issued_at = datetime.now(UTC)
    session = Session(
        session_id=run.session_id,
        uid=run.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=0.1,
        usage=SessionUsage(total_cost_usd=0.025),
    )

    submission = MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=4,
        run=run,
        score=0.9,
        usage=TokenUsageSummary.empty(),
        session=session,
    )
    return task, submission


def _make_batch_payload(
    *,
    batch_id: UUID,
    task_id: UUID | None = None,
    artifact_id: UUID | None = None,
    query_text: str = "What happened?",
) -> dict[str, object]:
    return {
        "batch_id": str(batch_id),
        "cutoff_at": "2026-03-08T00:00:00+00:00",
        "created_at": "2026-03-08T00:00:00+00:00",
        "tasks": [
            {
                "task_id": str(task_id or uuid4()),
                "query": {"text": query_text},
                "reference_answer": {"text": "The reference answer."},
                "budget_usd": 0.05,
            }
        ],
        "artifacts": [
            {
                "uid": 7,
                "artifact_id": str(artifact_id or uuid4()),
                "content_hash": "hash-123",
                "size_bytes": 42,
            }
        ],
    }


def test_progress_endpoint_includes_specifics_and_task_fields() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 1,
        "completed": 1,
        "remaining": 0,
        "tasks": (task,),
        "miner_task_runs": (submission,),
    }

    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/progress")

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"] == str(batch_id)
    assert body["total"] == 1
    assert body["completed"] == 1
    assert body["remaining"] == 0

    run = body["miner_task_runs"][0]
    assert run["score"] == pytest.approx(0.9)
    assert run["run"]["query"]["text"] == "What happened?"
    assert run["run"]["reference_answer"]["text"] == "The reference answer."
    assert run["run"]["response"]["text"] == "The miner answer."

    specifics = run["specifics"]
    assert specifics["total_tool_usage"]["search_tool_cost"] == pytest.approx(0.005)
    assert specifics["total_tool_usage"]["llm_cost"] == pytest.approx(0.02)
    assert specifics["score_breakdown"]["total_score"] == pytest.approx(0.9)
    assert specifics["elapsed_ms"] == pytest.approx(2500.0)


def test_accept_batch_endpoint_accepts_platform_json_payload() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "miner_task_runs": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            "batch_id": str(batch_id),
            "cutoff_at": "2026-03-08T00:00:00+00:00",
            "created_at": "2026-03-08T00:00:00+00:00",
            "tasks": [
                {
                    "task_id": str(task_id),
                    "query": {"text": "What happened?"},
                    "reference_answer": {"text": "The reference answer."},
                    "budget_usd": 0.05,
                }
            ],
            "artifacts": [
                {
                    "uid": 7,
                    "artifact_id": str(artifact_id),
                    "content_hash": "hash-123",
                    "size_bytes": 42,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "batch_id": str(batch_id),
        "caller": "caller",
    }
    received = provider.accept_batch.received_batch
    assert received is not None
    assert received.batch_id == batch_id
    assert received.tasks[0].task_id == task_id
    assert received.tasks[0].query.text == "What happened?"
    assert received.artifacts[0].artifact_id == artifact_id
    assert received.cutoff_at == "2026-03-08T00:00:00+00:00"
    assert received.created_at == "2026-03-08T00:00:00+00:00"


def test_accept_batch_endpoint_rejects_legacy_iso_keys() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "miner_task_runs": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            "batch_id": str(batch_id),
            "cutoff_at_iso": "2026-03-08T00:00:00+00:00",
            "created_at_iso": "2026-03-08T00:00:00+00:00",
            "tasks": [
                {
                    "task_id": str(task_id),
                    "query": {"text": "What happened?"},
                    "reference_answer": {"text": "The reference answer."},
                    "budget_usd": 0.05,
                }
            ],
            "artifacts": [
                {
                    "uid": 7,
                    "artifact_id": str(artifact_id),
                    "content_hash": "hash-123",
                    "size_bytes": 42,
                }
            ],
        },
    )

    assert response.status_code == 422
    assert provider.accept_batch.received_batch is None


def test_accept_batch_endpoint_rejects_non_strict_artifact_uid_payload() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "miner_task_runs": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            "batch_id": str(batch_id),
            "cutoff_at": "2026-03-08T00:00:00+00:00",
            "created_at": "2026-03-08T00:00:00+00:00",
            "tasks": [
                {
                    "task_id": str(task_id),
                    "query": {"text": "What happened?"},
                    "reference_answer": {"text": "The reference answer."},
                    "budget_usd": 0.05,
                }
            ],
            "artifacts": [
                {
                    "uid": "7",
                    "artifact_id": str(artifact_id),
                    "content_hash": "hash-123",
                    "size_bytes": 42,
                }
            ],
        },
    )

    assert response.status_code == 422
    assert provider.accept_batch.received_batch is None


def test_accept_batch_endpoint_is_idempotent_for_exact_duplicate_replay() -> None:
    provider = RealAcceptBatchDependencyProvider()
    app = _create_test_app(provider)
    client = TestClient(app)
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    payload = _make_batch_payload(batch_id=batch_id, task_id=task_id, artifact_id=artifact_id)

    first = client.post("/validator/miner-task-batches/batch", json=payload)
    second = client.post("/validator/miner-task-batches/batch", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert len(provider.inbox) == 1
    assert provider.status_provider.state.queued_batches == 1
    snapshot = provider.progress_tracker.snapshot(batch_id)
    assert snapshot["total"] == 1
    assert snapshot["completed"] == 0
    assert snapshot["remaining"] == 1


def test_accept_batch_endpoint_rejects_conflicting_duplicate_replay() -> None:
    provider = RealAcceptBatchDependencyProvider()
    app = _create_test_app(provider)
    client = TestClient(app)
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    payload = _make_batch_payload(batch_id=batch_id, task_id=task_id, artifact_id=artifact_id)
    conflicting = _make_batch_payload(
        batch_id=batch_id,
        task_id=task_id,
        artifact_id=artifact_id,
        query_text="Different question?",
    )

    first = client.post("/validator/miner-task-batches/batch", json=payload)
    second = client.post("/validator/miner-task-batches/batch", json=conflicting)

    assert first.status_code == 200
    assert second.status_code == 400
    assert second.json() == {"detail": "batch_id already exists with different contents"}
    assert len(provider.inbox) == 1
    assert provider.status_provider.state.queued_batches == 1
