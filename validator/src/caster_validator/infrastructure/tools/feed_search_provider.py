"""Validator tool provider for searching similar prior feed items via the platform API."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import bittensor as bt
import httpx

from caster_commons.bittensor import build_canonical_request
from caster_commons.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class HttpFeedSearchToolProvider:
    base_url: str
    hotkey: bt.Keypair
    timeout_seconds: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None

    def __post_init__(self) -> None:
        if not self.base_url:
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

    async def search_items(
        self,
        *,
        feed_id: UUID,
        enqueue_seq: int,
        search_queries: Sequence[str],
        num_hit: int,
    ) -> JsonObject:
        path = "/v1/feeds/search"
        body_obj = {
            "feed_id": str(feed_id),
            "enqueue_seq": enqueue_seq,
            "search_queries": list(search_queries),
            "num_hit": num_hit,
        }
        body = json.dumps(body_obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        headers = {
            "Authorization": self._signed_header("POST", path, body),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with self._client() as client:
            response = await client.post(path, content=body, headers=headers)
        if response.status_code != httpx.codes.OK:
            raise RuntimeError(f"platform returned {response.status_code} for POST {path}")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise RuntimeError("platform feed-search response must be a JSON object")
        return cast(JsonObject, payload)


__all__ = ["HttpFeedSearchToolProvider"]
