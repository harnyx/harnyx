from __future__ import annotations

import re
from uuid import uuid4

import bittensor as bt
import httpx
import pytest

from caster_commons.bittensor import build_canonical_request
from caster_validator.infrastructure.tools.platform_client import HttpPlatformClient

_HEADER_PATTERN = re.compile(
    r'^Bittensor\s+ss58="(?P<ss58>[^"]+)",\s*sig="(?P<sig>[0-9a-f]+)"$'
)


def _keypair() -> bt.Keypair:
    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _assert_signed(request: httpx.Request, keypair: bt.Keypair) -> None:
    header = request.headers.get("Authorization")
    assert header is not None
    match = _HEADER_PATTERN.match(header)
    assert match is not None
    assert match.group("ss58") == keypair.ss58_address
    path = request.url.raw_path.decode()
    query = request.url.query
    if query:
        path = f"{path}?{query}"
    body = request.content or b""
    canonical = build_canonical_request(request.method, path, body)
    signature = bytes.fromhex(match.group("sig"))
    assert keypair.verify(canonical, signature)


def test_get_champion_weights_returns_weights() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        if request.method == "GET" and request.url.path == "/v1/weights":
            payload = {
                "weights": {"42": 0.7, "7": 0.3},
                "final_top": [42, 7, None],
            }
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    weights = client.get_champion_weights()

    assert weights.weights == {42: 0.7, 7: 0.3}
    assert weights.final_top == (42, 7, None)


def test_get_miner_task_batch_parses_claim_budget_usd() -> None:
    batch_id = uuid4()
    claim_id = uuid4()
    artifact_id = uuid4()
    budget_usd = 0.123
    feed_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        expected_path = f"/v1/miner-task-batches/batch/{batch_id}"
        if request.method == "GET" and request.url.path == expected_path:
            payload = {
                "batch_id": str(batch_id),
                "entrypoint": "evaluate_criterion",
                "cutoff_at": "2025-10-17T12:00:00Z",
                "created_at": "2025-10-17T12:00:00Z",
                "claims": [
                    {
                        "claim_id": str(claim_id),
                        "text": "smoke",
                        "rubric": {
                            "title": "Accuracy",
                            "description": "desc",
                            "verdict_options": {
                                "options": [
                                    {"value": -1, "description": "Fail"},
                                    {"value": 1, "description": "Pass"},
                                ]
                            },
                        },
                        "reference_answer": {
                            "verdict": 1,
                            "justification": "ok",
                            "citations": [],
                        },
                        "budget_usd": budget_usd,
                        "context": {
                            "feed_id": str(feed_id),
                            "enqueue_seq": 5,
                        },
                    },
                ],
                "candidates": [
                    {
                        "uid": 7,
                        "artifact_id": str(artifact_id),
                        "content_hash": "abc",
                        "size_bytes": 1,
                    }
                ],
                "champion_uid": 7,
                "status": "created",
                "status_message": None,
            }
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    batch = client.get_miner_task_batch(batch_id)

    assert batch.batch_id == batch_id
    assert batch.claims[0].claim_id == claim_id
    assert batch.claims[0].budget_usd == pytest.approx(budget_usd)
    assert batch.claims[0].context is not None
    assert batch.claims[0].context.feed_id == feed_id
    assert batch.claims[0].context.enqueue_seq == 5
