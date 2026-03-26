from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from harnyx_commons.domain.tool_usage import (
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    TokenUsageSummary,
)
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.http.routes import (
    ControlRouteAuth,
    ValidatorControlDeps,
    add_control_routes,
)
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.infrastructure.state.run_progress import InMemoryRunProgress, RunProgressSnapshot


def _create_test_app(provider: DemoControlDependencyProvider) -> FastAPI:
    app = FastAPI()
    add_control_routes(app, provider)
    return app


@dataclass(frozen=True)
class _StubHotkey:
    ss58_address: str = "5validator"

    def sign(self, payload: bytes) -> bytes:
        return b"sig:" + payload


class StubAcceptBatch:
    def __init__(self, *, lifecycle: str | None = "processing", error_code: str | None = None) -> None:
        self.received_batch: MinerTaskBatchSpec | None = None
        self.received_restore_runs: tuple[MinerTaskRunSubmission, ...] = ()
        self._lifecycle = lifecycle
        self._error_code = error_code

    def execute(
        self,
        batch: object,
        *,
        restore_runs: tuple[MinerTaskRunSubmission, ...] = (),
    ) -> None:
        if not isinstance(batch, MinerTaskBatchSpec):
            raise AssertionError(f"expected MinerTaskBatchSpec, got {type(batch)!r}")
        self.received_batch = batch
        self.received_restore_runs = restore_runs
        return None

    def lifecycle_for(self, batch_id: UUID) -> str | None:
        _ = batch_id
        return self._lifecycle

    def error_code_for(self, batch_id: UUID) -> str | None:
        _ = batch_id
        return self._error_code


class StubStatusProvider:
    def snapshot(self) -> dict[str, object]:
        return {"status": "ok", "running": False}


class FakeProgressTracker:
    def __init__(self, *, snapshot: RunProgressSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, _: UUID) -> RunProgressSnapshot:
        return self._snapshot


class DemoControlDependencyProvider:
    def __init__(
        self,
        *,
        snapshot: RunProgressSnapshot,
        auth: ControlRouteAuth | None = None,
        lifecycle: str | None = "processing",
        error_code: str | None = None,
        validator_hotkey: _StubHotkey | None = None,
    ) -> None:
        self.accept_batch = StubAcceptBatch(lifecycle=lifecycle, error_code=error_code)
        self._deps = ValidatorControlDeps(
            accept_batch=self.accept_batch,
            status_provider=StubStatusProvider(),
            auth=_allow_all_auth if auth is None else auth,
            progress_tracker=FakeProgressTracker(snapshot=snapshot),
            validator_hotkey=validator_hotkey or _StubHotkey(),
        )

    def __call__(self) -> ValidatorControlDeps:
        return self._deps


class RealAcceptBatchDependencyProvider:
    def __init__(self) -> None:
        self.validator_hotkey = _StubHotkey()
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
            validator_hotkey=self.validator_hotkey,
        )

    def __call__(self) -> ValidatorControlDeps:
        return self._deps


async def _allow_all_auth(_: str, __: str, ___: bytes, ____: str | None) -> str:
    return "caller"


def test_status_endpoint_awaits_auth_with_request_primitives() -> None:
    batch_id = uuid4()
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "miner_task_runs": (),
    }
    auth_calls: list[tuple[str, str, bytes, str | None]] = []

    async def _record_auth(
        method: str,
        path_qs: str,
        body: bytes,
        authorization_header: str | None,
    ) -> str:
        auth_calls.append((method, path_qs, body, authorization_header))
        return "caller"

    provider = DemoControlDependencyProvider(snapshot=snapshot, auth=_record_auth)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(
        "/validator/status?verbose=1",
        headers={"Authorization": "Bittensor ss58=\"5demo\",sig=\"00\""},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["hotkey"] == "5validator"
    assert auth_calls == [(
        "GET",
        "/validator/status?verbose=1",
        b"",
        'Bittensor ss58="5demo",sig="00"',
    )]


def test_status_endpoint_returns_signed_ownership_proof_when_timestamp_header_is_present() -> None:
    batch_id = uuid4()
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "miner_task_runs": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot, validator_hotkey=_StubHotkey("5proof"))
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(
        "/validator/status",
        headers={
            "Authorization": 'Bittensor ss58="5demo",sig="00"',
            "X-Harnyx-Status-Ts": "2026-03-26T04:00:00+00:00",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["hotkey"] == "5proof"
    expected = "\n".join(
        (
            "validator-status-v1",
            "path=/validator/status",
            "request_ts=2026-03-26T04:00:00+00:00",
            "hotkey=5proof",
            "status=ok",
            "running=False",
        )
    ).encode("utf-8")
    assert payload["signature_hex"] == (b"sig:" + expected).hex()


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
                    "openai/gpt-oss-20b-TEE": LlmModelUsageCost(
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


def _make_restore_run_payload(submission: MinerTaskRunSubmission) -> dict[str, object]:
    score_breakdown = submission.run.details.score_breakdown
    if score_breakdown is None:
        raise AssertionError("expected restore submission score breakdown")
    total_tool_usage = submission.run.details.total_tool_usage
    llm_usage = total_tool_usage.llm
    return {
        "batch_id": str(submission.batch_id),
        "validator": {"uid": submission.validator_uid},
        "run": {
            "artifact_id": str(submission.run.artifact_id),
            "task_id": str(submission.run.task_id),
            "completed_at": submission.run.completed_at.isoformat(),
            "response": {"text": submission.run.response.text if submission.run.response else ""},
        },
        "score": submission.score,
        "usage": {
            "total_prompt_tokens": submission.usage.total_prompt_tokens,
            "total_completion_tokens": submission.usage.total_completion_tokens,
            "total_tokens": submission.usage.total_tokens,
            "call_count": submission.usage.call_count,
            "by_provider": submission.usage.by_provider,
        },
        "session": {
            "session_id": str(submission.session.session_id),
            "uid": submission.session.uid,
            "status": submission.session.status.value,
            "issued_at": submission.session.issued_at.isoformat(),
            "expires_at": submission.session.expires_at.isoformat(),
        },
        "specifics": {
            "score_breakdown": {
                "comparison_score": score_breakdown.comparison_score,
                "similarity_score": score_breakdown.similarity_score,
                "total_score": score_breakdown.total_score,
                "scoring_version": score_breakdown.scoring_version,
            },
            "total_tool_usage": {
                "search_tool": {
                    "call_count": total_tool_usage.search_tool.call_count,
                    "cost": total_tool_usage.search_tool.cost,
                },
                "search_tool_cost": total_tool_usage.search_tool_cost,
                "llm": {
                    "call_count": llm_usage.call_count,
                    "prompt_tokens": llm_usage.prompt_tokens,
                    "completion_tokens": llm_usage.completion_tokens,
                    "total_tokens": llm_usage.total_tokens,
                    "cost": llm_usage.cost,
                    "providers": {
                        provider: {
                            model: {
                                "usage": {
                                    "prompt_tokens": model_cost.usage.prompt_tokens,
                                    "completion_tokens": model_cost.usage.completion_tokens,
                                    "total_tokens": model_cost.usage.total_tokens,
                                    "call_count": model_cost.usage.call_count,
                                },
                                "cost": model_cost.cost,
                            }
                            for model, model_cost in models.items()
                        }
                        for provider, models in llm_usage.providers.items()
                    },
                },
                "llm_cost": total_tool_usage.llm_cost,
            },
            "elapsed_ms": submission.run.details.elapsed_ms,
        },
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
    assert body["status"] == "processing"
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


def test_progress_endpoint_keeps_ordered_runs_visible_when_lifecycle_is_failed() -> None:
    batch_id = uuid4()
    first_task, first_submission = _make_task_submission(batch_id=batch_id)
    second_task, second_submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 2,
        "completed": 2,
        "remaining": 0,
        "tasks": (first_task, second_task),
        "miner_task_runs": (first_submission, second_submission),
    }

    provider = DemoControlDependencyProvider(
        snapshot=snapshot,
        lifecycle="failed",
        error_code="sandbox_invocation_failed",
    )
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/progress")

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"] == str(batch_id)
    assert body["status"] == "failed"
    assert body["error_code"] == "sandbox_invocation_failed"
    assert body["total"] == 2
    assert body["completed"] == 2
    assert body["remaining"] == 0
    assert [run["run"]["task_id"] for run in body["miner_task_runs"]] == [
        str(first_task.task_id),
        str(second_task.task_id),
    ]
    assert [run["run"]["query"]["text"] for run in body["miner_task_runs"]] == [
        first_task.query.text,
        second_task.query.text,
    ]


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
    assert provider.accept_batch.received_restore_runs == ()


def test_accept_batch_endpoint_forwards_restore_runs() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
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
            **_make_batch_payload(
                batch_id=batch_id,
                task_id=task.task_id,
                artifact_id=submission.run.artifact_id,
                query_text=task.query.text,
            ),
            "restore_runs": [_make_restore_run_payload(submission)],
        },
    )

    assert response.status_code == 200
    assert len(provider.accept_batch.received_restore_runs) == 1
    restored = provider.accept_batch.received_restore_runs[0]
    assert restored.batch_id == submission.batch_id
    assert restored.validator_uid == submission.validator_uid
    assert restored.run.artifact_id == submission.run.artifact_id
    assert restored.run.task_id == submission.run.task_id
    assert restored.run.response == submission.run.response
    assert restored.run.completed_at == submission.run.completed_at
    assert restored.score == submission.score
    assert restored.session.session_id == submission.session.session_id
    assert restored.session.uid == submission.session.uid


def test_accept_batch_endpoint_rejects_invalid_restore_session_status() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
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
    restore_payload = _make_restore_run_payload(submission)
    session_payload = restore_payload["session"]
    if not isinstance(session_payload, dict):
        raise AssertionError("expected restore session payload dict")
    session_payload["status"] = "not_a_real_status"

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            **_make_batch_payload(
                batch_id=batch_id,
                task_id=task.task_id,
                artifact_id=submission.run.artifact_id,
                query_text=task.query.text,
            ),
            "restore_runs": [restore_payload],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "'not_a_real_status' is not a valid SessionStatus"
    assert provider.accept_batch.received_batch is None
    assert provider.accept_batch.received_restore_runs == ()


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


def test_progress_endpoint_returns_unknown_for_unaccepted_batch() -> None:
    batch_id = uuid4()
    snapshot: RunProgressSnapshot = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "miner_task_runs": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot, lifecycle=None)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/progress")

    assert response.status_code == 200
    assert response.json() == {
        "batch_id": str(batch_id),
        "status": "unknown",
        "error_code": None,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "miner_task_runs": [],
    }


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
