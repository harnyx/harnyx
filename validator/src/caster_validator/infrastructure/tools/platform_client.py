"""HTTP client for the centralized evaluation platform."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import bittensor as bt
import httpx

from caster_commons.bittensor import build_canonical_request
from caster_validator.application.dto.evaluation import MinerTaskBatchSpec
from caster_validator.application.ports.platform import ChampionWeights, PlatformPort
from caster_validator.infrastructure.parsers import parse_batch


class PlatformClientError(RuntimeError):
    """Raised when the platform responds with an unexpected status."""


@dataclass
class HttpPlatformClient(PlatformPort):
    """Implementation of PlatformPort backed by HTTPX."""

    base_url: str
    hotkey: bt.Keypair
    timeout_seconds: float = 10.0
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("platform base_url must not be empty")

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        )

    def _signed_header(self, method: str, path_qs: str, body: bytes) -> str:
        canonical = build_canonical_request(method, path_qs, body)
        signature = self.hotkey.sign(canonical)
        return (
            f'Bittensor ss58="{self.hotkey.ss58_address}",' f'sig="{signature.hex()}"'
        )

    def _request_headers(self, method: str, path_qs: str, body: bytes) -> dict[str, str]:
        headers = {
            "Authorization": self._signed_header(method, path_qs, body),
            "Accept": "application/json",
        }
        if body:
            headers["Content-Type"] = "application/json"
        return headers

    def get_miner_task_batch(self, batch_id: UUID) -> MinerTaskBatchSpec:
        path = f"/v1/miner-task-batches/batch/{batch_id}"
        with self._client() as client:
            response = client.get(
                path,
                headers=self._request_headers("GET", path, b""),
            )
        if response.status_code != httpx.codes.OK:
            raise PlatformClientError(
                f"platform returned {response.status_code} for GET {path}",
            )
        return parse_batch(response.json())

    def fetch_artifact(self, batch_id: UUID, artifact_id: UUID) -> bytes:
        path = f"/v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}"
        with self._client() as client:
            response = client.get(
                path,
                headers=self._request_headers("GET", path, b""),
            )
        if response.status_code != httpx.codes.OK:
            raise PlatformClientError(
                f"platform returned {response.status_code} for GET {path}",
            )
        return response.content

    def get_champion_weights(self) -> ChampionWeights:
        path = "/v1/weights"
        with self._client() as client:
            response = client.get(
                path,
                headers=self._request_headers("GET", path, b""),
            )
        if response.status_code != httpx.codes.OK:
            raise PlatformClientError(
                f"platform returned {response.status_code} for GET /v1/weights"
            )
        payload = response.json()
        weights = {int(uid): float(weight) for uid, weight in payload.get("weights", {}).items()}
        champion_uid_raw = payload.get("champion_uid")
        champion_uid = int(champion_uid_raw) if champion_uid_raw is not None else None
        return ChampionWeights(champion_uid=champion_uid, weights=weights)


__all__ = ["HttpPlatformClient", "PlatformClientError"]
