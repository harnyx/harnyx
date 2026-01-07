from __future__ import annotations

import re

import bittensor as bt
import httpx

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
