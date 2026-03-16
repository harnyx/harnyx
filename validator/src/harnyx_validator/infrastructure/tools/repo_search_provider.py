"""Validator tool provider for repository search via platform proxy endpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import bittensor as bt
import httpx

from harnyx_commons.bittensor import build_canonical_request
from harnyx_commons.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class HttpRepoSearchToolProvider:
    base_url: str
    hotkey: bt.Keypair
    timeout_seconds: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("platform base_url must not be empty")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout_seconds,
            transport=self.transport,
        )

    def _signed_header(self, method: str, path_qs: str, body: bytes) -> str:
        canonical = build_canonical_request(method, path_qs, body)
        signature = self.hotkey.sign(canonical)
        return f'Bittensor ss58="{self.hotkey.ss58_address}",sig="{signature.hex()}"'

    def _request_headers(self, method: str, path_qs: str, body: bytes) -> dict[str, str]:
        return {
            "Authorization": self._signed_header(method, path_qs, body),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def search_repo(
        self,
        *,
        repo_url: str,
        commit_sha: str,
        query: str,
        path_glob: str | None,
        limit: int,
    ) -> JsonObject:
        payload: JsonObject = {
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "query": query,
            "limit": limit,
        }
        if path_glob is not None:
            payload["path_glob"] = path_glob
        return await self._post_json("/v1/repo-search/search", payload)

    async def get_repo_file(
        self,
        *,
        repo_url: str,
        commit_sha: str,
        path: str,
        start_line: int | None,
        end_line: int | None,
    ) -> JsonObject:
        payload: JsonObject = {
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "path": path,
        }
        if start_line is not None:
            payload["start_line"] = start_line
        if end_line is not None:
            payload["end_line"] = end_line
        return await self._post_json("/v1/repo-search/get-file", payload)

    async def _post_json(self, path: str, payload: JsonObject) -> JsonObject:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        headers = self._request_headers("POST", path, body)
        async with self._client() as client:
            response = await client.post(path, content=body, headers=headers)
        if response.status_code != httpx.codes.OK:
            raise RuntimeError(f"platform returned {response.status_code} for POST {path}")
        data = response.json()
        if not isinstance(data, Mapping):
            raise RuntimeError(f"platform repo-search response for {path} must be a JSON object")
        return cast(JsonObject, dict(data))


__all__ = ["HttpRepoSearchToolProvider"]
