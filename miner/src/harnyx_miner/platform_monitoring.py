from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID

import httpx
from dotenv import load_dotenv

_QueryParamValue = str | int | float | None


@dataclass(frozen=True, slots=True)
class SelectedBatchContext:
    batch_id: UUID
    source: str
    detail: dict[str, object]
    results: tuple[dict[str, object], ...]


def platform_base_url_from_env() -> str:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    base_url = (os.getenv("PLATFORM_BASE_URL") or "").strip()
    if not base_url:
        raise RuntimeError("PLATFORM_BASE_URL must be set (for example: https://api.harnyx.ai)")
    return base_url.rstrip("/")


class PlatformMonitoringClient:
    def __init__(self, *, base_url: str, timeout_seconds: float = 30.0) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            follow_redirects=True,
        )

    @classmethod
    def from_env(cls) -> PlatformMonitoringClient:
        return cls(base_url=platform_base_url_from_env())

    def close(self) -> None:
        self._client.close()

    def find_latest_completed_batch(self) -> dict[str, object]:
        before: str | None = None
        while True:
            params: dict[str, _QueryParamValue] = {"limit": 100}
            if before is not None:
                params["before"] = before
            payload = self._get_json_object("/v1/monitoring/miner-task-batches", params=params)
            batches = _require_sequence(payload.get("batches"), label="monitoring batches")
            for raw_batch in batches:
                batch = _require_mapping(raw_batch, label="monitoring batch")
                if str(batch.get("status")) == "completed":
                    return dict(batch)
            next_before = payload.get("next_before")
            if next_before in (None, ""):
                break
            before = str(next_before)
        raise RuntimeError("no completed public miner-task batch is available")

    def get_batch_detail(self, batch_id: UUID) -> dict[str, object]:
        return self._get_json_object(f"/v1/monitoring/miner-task-batches/{batch_id}")

    def get_batch_results(self, batch_id: UUID) -> tuple[dict[str, object], ...]:
        payload = self._request_json(
            f"/v1/monitoring/miner-task-batches/{batch_id}/results",
            params={
                "include_failed_delivery_rows": "true",
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError("monitoring batch results response must be a JSON array")
        return tuple(dict(_require_mapping(row, label="monitoring result row")) for row in payload)

    def get_script(self, artifact_id: UUID) -> dict[str, object]:
        return self._get_json_object(
            f"/v1/monitoring/miner-scripts/{artifact_id}",
            params={"include_content": "true"},
        )

    def resolve_batch_context(self, batch_id: UUID | None) -> SelectedBatchContext:
        source = "explicit"
        resolved_batch_id = batch_id
        if resolved_batch_id is None:
            latest = self.find_latest_completed_batch()
            resolved_batch_id = UUID(str(latest.get("batch_id")))
            source = "latest-completed"
        detail = self.get_batch_detail(resolved_batch_id)
        _require_completed_batch_detail(detail, batch_id=resolved_batch_id)
        results = self.get_batch_results(resolved_batch_id)
        return SelectedBatchContext(
            batch_id=resolved_batch_id,
            source=source,
            detail=detail,
            results=results,
        )

    def _get_json_object(
        self,
        path: str,
        *,
        params: Mapping[str, _QueryParamValue] | None = None,
    ) -> dict[str, object]:
        data = self._request_json(path, params=params)
        return dict(_require_mapping(data, label=f"monitoring payload for {path}"))

    def _request_json(
        self,
        path: str,
        *,
        params: Mapping[str, _QueryParamValue] | None = None,
    ) -> object:
        response = self._client.get(path, params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = (response.text or "").strip()
            raise RuntimeError(f"platform monitoring request failed ({response.status_code}): {detail}") from exc
        return response.json()


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def _require_completed_batch_detail(detail: Mapping[str, object], *, batch_id: UUID) -> None:
    summary = _require_mapping(detail.get("summary"), label="monitoring batch summary")
    status = str(summary.get("status") or "")
    if status != "completed":
        raise RuntimeError(f"miner-task batch {batch_id} is not completed (status={status or 'unknown'})")


__all__ = [
    "PlatformMonitoringClient",
    "SelectedBatchContext",
    "platform_base_url_from_env",
]
