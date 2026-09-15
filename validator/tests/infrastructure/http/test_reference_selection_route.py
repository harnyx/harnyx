"""Protect authenticated two-order reference judging, including FAST and partial failure."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import bittensor as bt
import httpx
import pytest
from fastapi import FastAPI

from harnyx_commons.bittensor import build_canonical_request, verify_signed_request
from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.endpoint_answer import EndpointAnswer
from harnyx_commons.llm.schema import LlmResponse, LlmUsage
from harnyx_commons.miner_task_scoring import EvaluationScoringConfig, EvaluationScoringService
from harnyx_commons.reference_selection import ReferenceSelectionRequest
from harnyx_miner_sdk.endpoint_protocol import EndpointCallback, query_digest
from harnyx_miner_sdk.query import Response
from harnyx_validator.application.reference_selection import ReferenceJudge
from harnyx_validator.application.status import BatchActivityTracker, StatusProvider
from harnyx_validator.infrastructure.http.routes import ValidatorControlDeps, add_control_routes

from .test_similarity_route import _StubResourceUsageProvider


def signed_answer(query):
    miner = bt.Keypair.create_from_uri("//Bob")
    identity = uuid4()
    path = f"/v1/endpoint-assignments/{identity}/callback"
    body = EndpointCallback(
        assignment_id=identity,
        query_digest=query_digest(query),
        nonce="a" * 32,
        expires_at=datetime(2026, 1, 1, tzinfo=UTC),
        response=Response(text="Candidate"),
    ).model_dump_json()
    return EndpointAnswer(
        assignment_id=identity,
        expected_hotkey=miner.ss58_address,
        callback_body_utf8=body,
        signature_hex=miner.sign(build_canonical_request("POST", path, body.encode())).hex(),
        signed_callback_path=path,
        receipt_logs=(),
    )


class JudgeProvider:
    def __init__(self, fail_second=False):
        self.requests = []
        self.fail_second = fail_second

    async def invoke(self, request):
        self.requests.append(request)
        if self.fail_second and len(self.requests) == 2:
            raise ValueError("second judgment failed")
        return LlmResponse(id="test", choices=(), usage=LlmUsage(), postprocessed={"preferred_position": "first"})


@pytest.mark.anyio
@pytest.mark.parametrize("fast", [False, True])
async def test_signed_route_uses_both_orders_and_rejects_bad_signature(fast):
    platform = bt.Keypair.create_from_uri("//Alice")
    provider = JudgeProvider()
    judge = ReferenceJudge(EvaluationScoringService(provider, EvaluationScoringConfig(provider="chutes", model="test")))

    async def auth(method, path, body, header):
        return await asyncio.to_thread(
            verify_signed_request,
            method=method,
            path_qs=path,
            body=body,
            authorization_header=header,
            allowed_ss58=(platform.ss58_address,),
        )

    deps = ValidatorControlDeps(
        StatusProvider(), auth, platform, _StubResourceUsageProvider(), BatchActivityTracker(), reference_judge=judge
    )
    app = FastAPI()
    add_control_routes(app, lambda: deps)
    query = Query(text="Question", fast=fast)
    work = ReferenceSelectionRequest(
        batch_id=uuid4(),
        task=MinerTask(task_id=uuid4(), query=query, reference_answer=ReferenceAnswer(text="Dataset"), budget_usd=1),
        candidate=signed_answer(query),
    )
    path = f"/validator/miner-task-batches/{work.batch_id}/reference-selection"
    body = work.model_dump_json().encode()
    signature = platform.sign(build_canonical_request("POST", path, body)).hex()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bittensor ss58="{platform.ss58_address}",sig="{signature}"',
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://validator.test") as client:
        response = await client.post(path, content=body, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["comparison_score"] == 0.5
        assert len(provider.requests) == 2
        prompts = [request.messages[1].content[0].text for request in provider.requests]
        assert prompts[0] != prompts[1]
        assert "//Bob" not in "".join(prompts)
        invalid = await client.post(path, content=body + b" ", headers=headers)
        assert invalid.status_code == 401
        assert len(provider.requests) == 2
        provider.fail_second = True
        provider.requests.clear()
        failed = await client.post(path, content=body, headers=headers)
        assert failed.status_code == 502
        assert "comparison_score" not in failed.json()
