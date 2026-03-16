from __future__ import annotations

import json
import re

import bittensor as bt
import httpx
import pytest

from harnyx_commons.bittensor import build_canonical_request
from harnyx_validator.infrastructure.tools.repo_search_provider import HttpRepoSearchToolProvider

pytestmark = pytest.mark.anyio("asyncio")

_HEADER_PATTERN = re.compile(
    r'^Bittensor\s+ss58="(?P<ss58>[^"]+)",\s*sig="(?P<sig>[0-9a-f]+)"$'
)


def _keypair() -> bt.Keypair:
    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _assert_signed_post(
    request: httpx.Request,
    *,
    keypair: bt.Keypair,
    expected_path: str,
    expected_body: bytes,
) -> None:
    assert request.method == "POST"
    assert request.url.path == expected_path
    assert request.content == expected_body
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Accept"] == "application/json"

    header = request.headers.get("Authorization")
    assert header is not None
    match = _HEADER_PATTERN.match(header)
    assert match is not None
    assert match.group("ss58") == keypair.ss58_address

    path = request.url.raw_path.decode()
    query = request.url.query
    if query:
        path = f"{path}?{query}"
    canonical = build_canonical_request(request.method, path, request.content)
    signature = bytes.fromhex(match.group("sig"))
    assert keypair.verify(canonical, signature)


async def test_search_repo_posts_signed_payload_and_returns_mapping() -> None:
    keypair = _keypair()
    expected_body = json.dumps(
        {
            "repo_url": "https://github.com/org/repo",
            "commit_sha": "a" * 40,
            "query": "proxy wiring",
            "limit": 10,
            "path_glob": "docs/*.md",
        },
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed_post(
            request,
            keypair=keypair,
            expected_path="/v1/repo-search/search",
            expected_body=expected_body,
        )
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {
                        "path": "docs/a.md",
                        "url": "https://github.com/org/repo/blob/sha/docs/a.md",
                        "bm25": 1.5,
                    }
                ]
            },
        )

    provider = HttpRepoSearchToolProvider(
        base_url="https://platform.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    payload = await provider.search_repo(
        repo_url="https://github.com/org/repo",
        commit_sha="a" * 40,
        query="proxy wiring",
        path_glob="docs/*.md",
        limit=10,
    )

    assert payload == {
        "data": [
            {
                "path": "docs/a.md",
                "url": "https://github.com/org/repo/blob/sha/docs/a.md",
                "bm25": 1.5,
            }
        ]
    }


async def test_get_repo_file_posts_signed_payload_and_returns_mapping() -> None:
    keypair = _keypair()
    expected_body = json.dumps(
        {
            "repo_url": "https://github.com/org/repo",
            "commit_sha": "b" * 40,
            "path": "docs/a.md",
            "start_line": 5,
            "end_line": 12,
        },
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed_post(
            request,
            keypair=keypair,
            expected_path="/v1/repo-search/get-file",
            expected_body=expected_body,
        )
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {
                        "path": "docs/a.md",
                        "url": "https://github.com/org/repo/blob/sha/docs/a.md",
                        "text": "hello",
                    }
                ]
            },
        )

    provider = HttpRepoSearchToolProvider(
        base_url="https://platform.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    payload = await provider.get_repo_file(
        repo_url="https://github.com/org/repo",
        commit_sha="b" * 40,
        path="docs/a.md",
        start_line=5,
        end_line=12,
    )

    assert payload == {
        "data": [
            {
                "path": "docs/a.md",
                "url": "https://github.com/org/repo/blob/sha/docs/a.md",
                "text": "hello",
            }
        ]
    }


async def test_non_200_raises_runtime_error() -> None:
    keypair = _keypair()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=503)

    provider = HttpRepoSearchToolProvider(
        base_url="https://platform.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError, match="platform returned 503 for POST /v1/repo-search/search"):
        await provider.search_repo(
            repo_url="https://github.com/org/repo",
            commit_sha="c" * 40,
            query="query",
            path_glob=None,
            limit=3,
        )


async def test_non_mapping_payload_raises_runtime_error() -> None:
    keypair = _keypair()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json=["not", "an", "object"])

    provider = HttpRepoSearchToolProvider(
        base_url="https://platform.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        RuntimeError,
        match="platform repo-search response for /v1/repo-search/get-file must be a JSON object",
    ):
        await provider.get_repo_file(
            repo_url="https://github.com/org/repo",
            commit_sha="d" * 40,
            path="docs/a.md",
            start_line=None,
            end_line=None,
        )
