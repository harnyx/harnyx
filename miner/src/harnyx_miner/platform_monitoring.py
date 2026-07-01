from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import httpx

from harnyx_miner.env import load_public_env

_QueryParamValue = str | int | float | None


class PlatformMonitoringRequestError(RuntimeError):
    def __init__(
        self,
        *,
        path: str,
        status_code: int,
        detail: str,
    ) -> None:
        self.path = path
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"platform monitoring request failed ({status_code}): {detail}")


@dataclass(frozen=True, slots=True)
class RecordedResultsScope:
    batch_id: UUID
    artifact_id: UUID
    kind: str = "artifact"


@dataclass(frozen=True, slots=True)
class RecordedResultsError:
    detail: str
    path: str | None = None
    status_code: int | None = None

    @classmethod
    def from_request_error(cls, exc: PlatformMonitoringRequestError) -> RecordedResultsError:
        return cls(path=exc.path, status_code=exc.status_code, detail=exc.detail)


@dataclass(frozen=True, slots=True)
class RecordedBatchResultsSnapshot:
    rows: tuple[dict[str, object], ...] | None
    error: RecordedResultsError | None
    scope: RecordedResultsScope | None

    @classmethod
    def unavailable_without_baseline(cls, batch_id: UUID) -> RecordedBatchResultsSnapshot:
        return cls(
            rows=None,
            error=RecordedResultsError(
                detail=f"batch {batch_id} does not expose a champion artifact for recorded context"
            ),
            scope=None,
        )


@dataclass(frozen=True, slots=True)
class SelectedBatchContext:
    batch_id: UUID
    source: str
    detail: dict[str, object]
    recorded_results: RecordedBatchResultsSnapshot


def platform_base_url_from_env() -> str:
    load_public_env()
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

    def get_recorded_results(
        self,
        *,
        batch_id: UUID,
        artifact_id: UUID,
    ) -> tuple[dict[str, object], ...]:
        task_index = self._request_json(
            f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{artifact_id}/tasks",
        )
        task_ids = _completed_task_ids_from_index_payload(task_index)
        rows: list[dict[str, object]] = []
        for task_id in task_ids:
            payload = self._request_json(
                f"/v1/monitoring/miner-task-batches/{batch_id}/artifacts/{artifact_id}/tasks/{task_id}/results",
            )
            if not isinstance(payload, list):
                raise RuntimeError("monitoring task results response must be a JSON array")
            rows.extend(
                dict(row)
                for row in (_require_mapping(item, label="monitoring task result row") for item in payload)
                if "lifecycle_status" not in row
            )
        return tuple(rows)

    def get_recorded_results_snapshot(
        self,
        *,
        batch_id: UUID,
        artifact_id: UUID,
    ) -> RecordedBatchResultsSnapshot:
        scope = RecordedResultsScope(batch_id=batch_id, artifact_id=artifact_id)
        try:
            rows = self.get_recorded_results(batch_id=batch_id, artifact_id=artifact_id)
        except PlatformMonitoringRequestError as exc:
            return RecordedBatchResultsSnapshot(
                rows=None,
                error=RecordedResultsError.from_request_error(exc),
                scope=None,
            )
        return RecordedBatchResultsSnapshot(rows=rows, error=None, scope=scope)

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
        recorded_artifact_id = _recorded_context_artifact_id(detail)
        recorded_results = (
            self.get_recorded_results_snapshot(
                batch_id=resolved_batch_id,
                artifact_id=recorded_artifact_id,
            )
            if recorded_artifact_id is not None
            else RecordedBatchResultsSnapshot.unavailable_without_baseline(resolved_batch_id)
        )
        return SelectedBatchContext(
            batch_id=resolved_batch_id,
            source=source,
            detail=detail,
            recorded_results=recorded_results,
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
        try:
            response = self._client.get(path, params=params)
        except httpx.RequestError as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            raise PlatformMonitoringRequestError(
                path=path,
                status_code=0,
                detail=detail,
            ) from exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = (response.text or "").strip()
            raise PlatformMonitoringRequestError(
                path=path,
                status_code=response.status_code,
                detail=detail,
            ) from exc
        return response.json()


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def _completed_task_ids_from_index_payload(payload: object) -> tuple[UUID, ...]:
    rows = _require_sequence(payload, label="monitoring task index")
    task_ids: list[UUID] = []
    seen: set[UUID] = set()
    for raw_row in rows:
        row = _require_mapping(raw_row, label="monitoring task index row")
        if "lifecycle_status" in row:
            continue
        task_id_value = row.get("task_id")
        if task_id_value in (None, ""):
            raise RuntimeError("monitoring task index row missing task_id")
        task_id = UUID(str(task_id_value))
        if task_id not in seen:
            seen.add(task_id)
            task_ids.append(task_id)
    return tuple(task_ids)


def _require_completed_batch_detail(detail: Mapping[str, object], *, batch_id: UUID) -> None:
    summary = _require_mapping(detail.get("summary"), label="monitoring batch summary")
    status = str(summary.get("status") or "")
    if status != "completed":
        raise RuntimeError(f"miner-task batch {batch_id} is not completed (status={status or 'unknown'})")


def _recorded_context_artifact_id(detail: Mapping[str, object]) -> UUID | None:
    summary = _require_mapping(detail.get("summary"), label="monitoring batch summary")
    champion_artifact_id = summary.get("champion_artifact_id")
    if champion_artifact_id in (None, ""):
        return None
    return UUID(str(champion_artifact_id))


__all__ = [
    "PlatformMonitoringRequestError",
    "PlatformMonitoringClient",
    "RecordedBatchResultsSnapshot",
    "RecordedResultsError",
    "RecordedResultsScope",
    "SelectedBatchContext",
    "platform_base_url_from_env",
]
