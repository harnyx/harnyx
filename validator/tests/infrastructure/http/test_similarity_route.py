from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from harnyx_commons.domain.judge_usage import JudgeModelUsage, JudgeUsageSummary
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
            classification="novel",
            reasoning="provider reasoning trace",
            reasoning_tokens=19,
            model="google/gemma-4-31B-turbo-TEE",
            provider="custom-openai-compatible:gemma4-cloud-run-turbo",
            judge_usage=JudgeUsageSummary(
                call_count=1,
                prompt_tokens=31,
                completion_tokens=13,
                total_tokens=44,
                reasoning_tokens=19,
                actual_cost_usd=0.04,
                models=(
                    JudgeModelUsage(
                        provider="custom-openai-compatible:gemma4-cloud-run-turbo",
                        model="google/gemma-4-31B-turbo-TEE",
                        call_count=1,
                        prompt_tokens=31,
                        completion_tokens=13,
                        total_tokens=44,
                        reasoning_tokens=19,
                        actual_cost_usd=0.04,
                        actual_cost_source="provider_actual",
                    ),
                ),
            ),
        )


class FailingSimilarityJudge:
    def __init__(self) -> None:
        self.requests: list[SimilarityJudgeRequest] = []

    async def judge(self, request: SimilarityJudgeRequest) -> SimilarityJudgeResult:
        self.requests.append(request)
        exc = RuntimeError("provider retries exhausted")
        exc.__dict__["judge_usage"] = JudgeUsageSummary(
            call_count=2,
            prompt_tokens=37,
            completion_tokens=15,
            total_tokens=52,
            reasoning_tokens=9,
            actual_cost_usd=None,
            models=(
                JudgeModelUsage(
                    provider="chutes",
                    model="moonshotai/Kimi-K2.5-TEE",
                    call_count=2,
                    prompt_tokens=37,
                    completion_tokens=15,
                    total_tokens=52,
                    reasoning_tokens=9,
                    actual_cost_usd=None,
                    actual_cost_source="unavailable",
                ),
            ),
        )
        raise exc


async def _allow_all_auth(_: str, __: str, ___: bytes, ____: str | None) -> str:
    return "caller"


class _StubHotkey:
    ss58_address = "5validator"

    def sign(self, payload: bytes) -> bytes:
        return b"sig:" + payload


class _StubResourceUsageProvider:
    def snapshot(self) -> ValidatorResourceUsageSnapshot:
        raise AssertionError("similarity route should not sample validator resource usage")


def _client(judge: StubSimilarityJudge | FailingSimilarityJudge | None) -> TestClient:
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
        "classification": "novel",
        "reasoning": "provider reasoning trace",
        "reasoning_tokens": 19,
        "model": "google/gemma-4-31B-turbo-TEE",
        "provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
        "judge_usage": {
            "call_count": 1,
            "prompt_tokens": 31,
            "completion_tokens": 13,
            "total_tokens": 44,
            "reasoning_tokens": 19,
            "actual_cost_usd": 0.04,
            "models": [
                {
                    "provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
                    "model": "google/gemma-4-31B-turbo-TEE",
                    "call_count": 1,
                    "prompt_tokens": 31,
                    "completion_tokens": 13,
                    "total_tokens": 44,
                    "reasoning_tokens": 19,
                    "actual_cost_usd": 0.04,
                    "actual_cost_source": "provider_actual",
                    "actual_cost_provider": None,
                    "actual_cost_evidence": None,
                }
            ],
        },
    }
    assert judge.requests == [
        SimilarityJudgeRequest(
            batch_id=batch_id,
            candidate_artifact_id=candidate_artifact_id,
            reference_artifact_id=incumbent_artifact_id,
            candidate_miner_uid=20,
            reference_miner_uid=10,
            reference_script="def old(): pass",
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


def test_similarity_route_returns_failed_judge_usage() -> None:
    batch_id = uuid4()
    candidate_artifact_id = uuid4()
    incumbent_artifact_id = uuid4()
    judge = FailingSimilarityJudge()
    client = _client(judge)

    response = client.post(
        f"/validator/miner-task-batches/{batch_id}/similarity",
        json=_payload(
            candidate_artifact_id=candidate_artifact_id,
            incumbent_artifact_id=incumbent_artifact_id,
        ),
    )

    assert response.status_code == 502
    assert response.json() == {
        "error_code": "similarity_judge_failed",
        "retryable": False,
        "detail": "provider retries exhausted",
        "judge_usage": {
            "call_count": 2,
            "prompt_tokens": 37,
            "completion_tokens": 15,
            "total_tokens": 52,
            "reasoning_tokens": 9,
            "actual_cost_usd": None,
            "models": [
                {
                    "provider": "chutes",
                    "model": "moonshotai/Kimi-K2.5-TEE",
                    "call_count": 2,
                    "prompt_tokens": 37,
                    "completion_tokens": 15,
                    "total_tokens": 52,
                    "reasoning_tokens": 9,
                    "actual_cost_usd": None,
                    "actual_cost_source": "unavailable",
                    "actual_cost_provider": None,
                    "actual_cost_evidence": None,
                }
            ],
        },
    }


def test_similarity_route_preserves_missing_config_service_unavailable() -> None:
    batch_id = uuid4()
    candidate_artifact_id = uuid4()
    incumbent_artifact_id = uuid4()
    client = _client(None)

    response = client.post(
        f"/validator/miner-task-batches/{batch_id}/similarity",
        json=_payload(
            candidate_artifact_id=candidate_artifact_id,
            incumbent_artifact_id=incumbent_artifact_id,
        ),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "similarity judge is not configured"}
