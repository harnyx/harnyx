from __future__ import annotations

from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from harnyx_commons.domain.judge_usage import JudgeModelUsage, JudgeUsageSummary
from harnyx_commons.llm.provider import LlmProviderConfigurationError, LlmRetryExhaustedError
from harnyx_commons.llm.providers.openai_stream import OpenAiStreamError
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
        "retryable": True,
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


@pytest.mark.parametrize("stream", [False, True])
def test_similarity_route_preserves_missing_config_service_unavailable(stream) -> None:
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
        headers={"Accept": "text/event-stream"} if stream else {},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error_code": "similarity_judge_failed", "retryable": False,
        "detail": "similarity judge is not configured", "judge_usage": None,
    }


def test_similarity_stream_preserves_result_and_attempt_failure_history() -> None:
    from harnyx_commons.llm.judge_observability import emit_judge_event
    from harnyx_commons.miner_task_similarity_stream import SIMILARITY_FRAME_ADAPTER, SimilarityResultFrame

    class FallbackJudge(StubSimilarityJudge):
        async def judge(self, request: SimilarityJudgeRequest) -> SimilarityJudgeResult:
            emit_judge_event("finished", {
                "attempt_id": "attempt-a", "model": "model-a", "provider": "chutes",
                "outcome": "failed", "timeout_phase": "prefill", "elapsed_ms": 300000.0,
                "first_output_ms": None, "http_error_body": "must not be exported",
                "timeout": {"total": 900.0, "prefill": 300.0, "inactivity": 60.0},
            })
            emit_judge_event("finished", {
                "attempt_id": "attempt-b", "model": "model-b", "provider": "chutes",
                "outcome": "response_received", "elapsed_ms": 10000.0, "first_output_ms": 120.0,
            })
            return await super().judge(request)

    judge = FallbackJudge()
    response = _client(judge).post(
        f"/validator/miner-task-batches/{uuid4()}/similarity",
        json=_payload(candidate_artifact_id=uuid4(), incumbent_artifact_id=uuid4()),
        headers={"Accept": "text/event-stream"},
    )
    frames = [SIMILARITY_FRAME_ADAPTER.validate_json(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ")]
    terminal = frames[-1]
    assert isinstance(terminal, SimilarityResultFrame)
    assert terminal.result["classification"] == "novel"
    assert terminal.result["judge_usage"]["call_count"] == 1
    assert [item.model for item in terminal.diagnostics.attempts] == ["model-a", "model-b"]
    assert terminal.diagnostics.attempts[0].timeout_phase == "prefill"
    assert terminal.diagnostics.attempts[0].first_output_ms is None
    assert "must not be exported" not in response.text
    assert len(judge.requests) == 1


def test_similarity_stream_keeps_failed_usage_and_retry_policy() -> None:
    import json

    response = _client(FailingSimilarityJudge()).post(
        f"/validator/miner-task-batches/{uuid4()}/similarity",
        json=_payload(candidate_artifact_id=uuid4(), incumbent_artifact_id=uuid4()),
        headers={"Accept": "text/event-stream"},
    )
    terminal = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")][-1]
    assert response.status_code == 200
    assert terminal["type"] == "error"
    assert terminal["status_code"] == 502
    assert terminal["error"]["retryable"] is True
    assert terminal["error"]["judge_usage"]["call_count"] == 2


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure,retryable", [
    ("timeout", True), ("invalid_output", True), ("configuration", False),
    (401, False), (403, False), (404, False), (429, True), (503, True),
    ("401", False), ("403", False), ("404", False), ("429", True), ("503", True),
])
def test_similarity_failure_allows_retry_except_authentication_and_configuration(stream, failure, retryable) -> None:
    """A failed fallback must get a second Platform chance unless operator action is required."""
    import json

    class Judge:
        async def judge(self, request):
            if failure == "timeout":
                raise LlmRetryExhaustedError("total_timeout") from TimeoutError()
            if failure == "invalid_output":
                raise ValueError("invalid structured output")
            if failure == "configuration":
                cause = LlmProviderConfigurationError("missing provider credentials")
            elif isinstance(failure, str):
                cause = OpenAiStreamError(message="streamed provider error", code=failure)
            else:
                response = httpx.Response(failure, request=httpx.Request("POST", "https://provider.invalid"))
                cause = httpx.HTTPStatusError("upstream rejected request", request=response.request, response=response)
            raise RuntimeError("wrapped provider failure") from cause

    response = _client(Judge()).post(
        f"/validator/miner-task-batches/{uuid4()}/similarity",
        json=_payload(candidate_artifact_id=uuid4(), incumbent_artifact_id=uuid4()),
        headers={"Accept": "text/event-stream"} if stream else {},
    )
    if stream:
        terminal = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")][-1]
        payload = terminal["error"]
        assert terminal["status_code"] == 502
    else:
        assert response.status_code == 502
        payload = response.json()
    assert payload["retryable"] is retryable


def test_similarity_stream_keeps_unexpected_error_distinct_from_model_failure() -> None:
    import json

    class BrokenJudge(StubSimilarityJudge):
        async def judge(self, request: SimilarityJudgeRequest) -> SimilarityJudgeResult:
            raise LookupError("unexpected failure")

    response = _client(BrokenJudge()).post(
        f"/validator/miner-task-batches/{uuid4()}/similarity",
        json=_payload(candidate_artifact_id=uuid4(), incumbent_artifact_id=uuid4()),
        headers={"Accept": "text/event-stream"},
    )
    terminal = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")][-1]
    assert terminal["status_code"] == 500
    assert terminal["error"]["error_code"] == "internal_server_error"


def test_stream_negotiation_does_not_bypass_authentication(monkeypatch) -> None:
    async def reject(*args):
        raise HTTPException(status_code=401, detail="authentication required")

    monkeypatch.setitem(globals(), "_allow_all_auth", reject)
    judge = StubSimilarityJudge()
    response = _client(judge).post(
        f"/validator/miner-task-batches/{uuid4()}/similarity",
        json=_payload(candidate_artifact_id=uuid4(), incumbent_artifact_id=uuid4()),
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/json"
    assert judge.requests == []
