from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest, SimilarityJudgeResult
from harnyx_validator.application.status import BatchActivityTracker, StatusProvider
from harnyx_validator.infrastructure.http.routes import ValidatorControlDeps, add_control_routes
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageSnapshot


class StubSimilarityJudge:
    def __init__(self) -> None:
        self.requests: list[SimilarityJudgeRequest] = []

    async def judge(self, request: SimilarityJudgeRequest) -> SimilarityJudgeResult:
        self.requests.append(request)
        return SimilarityJudgeResult(
            verdict="not_duplicate",
            reasoning="provider reasoning trace",
            reasoning_tokens=19,
            model="google/gemma-4-31B-turbo-TEE",
            provider="custom-openai-compatible:gemma4-cloud-run-turbo",
        )


async def _allow_all_auth(_: str, __: str, ___: bytes, ____: str | None) -> str:
    return "caller"


class _StubHotkey:
    ss58_address = "5validator"

    def sign(self, payload: bytes) -> bytes:
        return b"sig:" + payload


class _StubResourceUsageProvider:
    def snapshot(self) -> ValidatorResourceUsageSnapshot:
        raise AssertionError("similarity route should not sample validator resource usage")


def _client(judge: StubSimilarityJudge) -> TestClient:
    deps = ValidatorControlDeps(
        status_provider=StatusProvider(),
        auth=_allow_all_auth,
        validator_hotkey=_StubHotkey(),
        resource_usage_provider=_StubResourceUsageProvider(),
        batch_activity=BatchActivityTracker(),
        is_chutes_configured=False,
        is_openrouter_configured=False,
        similarity_judge=judge,
    )
    app = FastAPI()
    add_control_routes(app, lambda: deps)
    return TestClient(app)


def _payload(*, candidate_artifact_id: UUID, incumbent_artifact_id: UUID) -> dict[str, object]:
    return {
        "candidate_artifact_id": str(candidate_artifact_id),
        "incumbent_artifact_id": str(incumbent_artifact_id),
        "candidate_miner_uid": 20,
        "incumbent_miner_uid": 10,
        "incumbent_script": "def old(): pass",
        "candidate_diff": "+ def new(): pass",
    }


def test_similarity_route_runs_validator_owned_judge() -> None:
    batch_id = uuid4()
    candidate_artifact_id = uuid4()
    incumbent_artifact_id = uuid4()
    judge = StubSimilarityJudge()
    client = _client(judge)

    response = client.post(
        f"/validator/miner-task-batches/{batch_id}/similarity",
        json=_payload(
            candidate_artifact_id=candidate_artifact_id,
            incumbent_artifact_id=incumbent_artifact_id,
        ),
    )

    assert response.status_code == 200
    assert response.json() == {
        "verdict": "not_duplicate",
        "reasoning": "provider reasoning trace",
        "reasoning_tokens": 19,
        "model": "google/gemma-4-31B-turbo-TEE",
        "provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
    }
    assert judge.requests == [
        SimilarityJudgeRequest(
            batch_id=batch_id,
            candidate_artifact_id=candidate_artifact_id,
            incumbent_artifact_id=incumbent_artifact_id,
            candidate_miner_uid=20,
            incumbent_miner_uid=10,
            incumbent_script="def old(): pass",
            candidate_diff="+ def new(): pass",
        )
    ]


def test_similarity_route_rejects_prompt_payload() -> None:
    batch_id = uuid4()
    candidate_artifact_id = uuid4()
    incumbent_artifact_id = uuid4()
    judge = StubSimilarityJudge()
    client = _client(judge)
    payload = _payload(
        candidate_artifact_id=candidate_artifact_id,
        incumbent_artifact_id=incumbent_artifact_id,
    )
    payload["prompt"] = "platform-provided evaluator instructions"

    response = client.post(f"/validator/miner-task-batches/{batch_id}/similarity", json=payload)

    assert response.status_code == 422
    assert judge.requests == []
