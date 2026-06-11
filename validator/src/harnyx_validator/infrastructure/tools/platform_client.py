"""HTTP client for the centralized evaluation platform."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast
from urllib.parse import urlencode
from uuid import UUID

import bittensor as bt
import httpx

from harnyx_commons.bittensor import build_canonical_request
from harnyx_commons.domain.tool_call import ToolExecutionFacts
from harnyx_commons.errors import BudgetExceededError, ToolInvocationTimeoutError, ToolProviderError
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.protocol_headers import PLATFORM_TOOL_PROXY_TOKEN_HEADER
from harnyx_commons.tools.types import ToolName
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec
from harnyx_validator.application.ports.platform import (
    ChampionWeights,
    PlatformPort,
    PlatformToolProxyGrant,
    PlatformToolProxyPlatformPort,
    PlatformToolProxyToolResult,
    RestoreAttemptNumberHighWater,
    RestoreMetadata,
    RestoreRunsPage,
)
from harnyx_validator.infrastructure.http.schemas import (
    ProviderEvidenceModel,
    RestoreMinerTaskRunSubmissionModel,
)
from harnyx_validator.infrastructure.parsers import parse_batch
from harnyx_validator.infrastructure.transient_network import classify_transient_network_failure

_GET_ATTEMPTS = 2
_RESTORE_GET_TIMEOUT_SECONDS = 300.0
_PLATFORM_TOOL_PROXY_GRANT_RETRY_DELAYS_SECONDS = (0.25, 1.0)


class PlatformClientError(RuntimeError):
    """Raised when the platform responds with an unexpected status."""

    def __init__(self, *, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class PlatformToolProxyInvocationError(RuntimeError):
    """Raised when platform-tool-proxy rejects a tool invocation for non-provider reasons."""

    def __init__(self, *, status_code: int, error_code: str | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class PlatformToolProxyInterruptedError(RuntimeError):
    """Raised when a sent platform-tool-proxy execution is interrupted before a response."""

    error_code = "platform_interrupted"
    status_code = 0


class PlatformToolProxyProviderError(ToolProviderError):
    """Raised when platform-tool-proxy reports an upstream provider failure."""

    error_code = "provider_failed"

    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class PlatformToolProxyToolTimeoutError(ToolInvocationTimeoutError):
    """Raised when platform-tool-proxy reports a tool timeout."""

    error_code = "tool_timeout"

    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class PlatformToolProxyBudgetExceededError(BudgetExceededError):
    """Raised when platform-tool-proxy reports grant budget exhaustion."""

    error_code = "budget_exhausted"

    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


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
        return f'Bittensor ss58="{self.hotkey.ss58_address}",sig="{signature.hex()}"'

    def _request_headers(self, method: str, path_qs: str, body: bytes) -> dict[str, str]:
        headers = {
            "Authorization": self._signed_header(method, path_qs, body),
            "Accept": "application/json",
        }
        if body:
            headers["Content-Type"] = "application/json"
        return headers

    def _get(self, path: str, *, timeout_seconds: float | None = None) -> httpx.Response:
        for attempt in range(_GET_ATTEMPTS):
            try:
                with self._client() as client:
                    kwargs: dict[str, Any] = {}
                    if timeout_seconds is not None:
                        kwargs["timeout"] = timeout_seconds
                    return client.get(
                        path,
                        headers=self._request_headers("GET", path, b""),
                        **kwargs,
                    )
            except httpx.TransportError as exc:
                if classify_transient_network_failure(exc) is None or attempt == _GET_ATTEMPTS - 1:
                    raise
        raise RuntimeError("platform GET retry loop exhausted without response")

    def get_miner_task_batch(self, batch_id: UUID) -> MinerTaskBatchSpec:
        path = f"/v1/miner-task-batches/batch/{batch_id}"
        response = self._get(path)
        if response.status_code != httpx.codes.OK:
            raise PlatformClientError(
                status_code=response.status_code,
                message=f"platform returned {response.status_code} for GET {path}",
            )
        return parse_batch(response.json())

    def fetch_artifact(self, batch_id: UUID, artifact_id: UUID) -> bytes:
        path = f"/v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}"
        response = self._get(path)
        if response.status_code != httpx.codes.OK:
            raise PlatformClientError(
                status_code=response.status_code,
                message=f"platform returned {response.status_code} for GET {path}",
            )
        return response.content

    def get_champion_weights(self) -> ChampionWeights:
        path = "/v1/weights"
        response = self._get(path)
        if response.status_code != httpx.codes.OK:
            raise PlatformClientError(
                status_code=response.status_code,
                message=f"platform returned {response.status_code} for GET /v1/weights",
            )
        payload = response.json()
        weights = {int(uid): float(weight) for uid, weight in payload.get("weights", {}).items()}
        champion_uid_raw = payload.get("champion_uid")
        champion_uid = int(champion_uid_raw) if champion_uid_raw is not None else None
        return ChampionWeights(champion_uid=champion_uid, weights=weights)

    def get_restore_metadata(self, batch_id: UUID) -> RestoreMetadata:
        path = f"/v1/miner-task-batches/{batch_id}/restore"
        response = self._get(path, timeout_seconds=_RESTORE_GET_TIMEOUT_SECONDS)
        if response.status_code != httpx.codes.OK:
            raise PlatformClientError(
                status_code=response.status_code,
                message=f"platform returned {response.status_code} for GET {path}",
            )
        payload = response.json()
        provider_evidence = tuple(
            ProviderEvidenceModel.model_validate(entry).to_domain()
            for entry in payload.get("provider_model_evidence", ())
        )
        return RestoreMetadata(
            batch_id=UUID(str(payload["batch_id"])),
            snapshot_received_at=datetime.fromisoformat(str(payload["snapshot_received_at"])),
            total_restore_runs=int(payload["total_restore_runs"]),
            page_limit=int(payload["page_limit"]),
            last_progress_detail_sequence=int(payload.get("last_progress_detail_sequence", 0)),
            provider_model_evidence=provider_evidence,
            terminated_miner_task_attempts=tuple(
                _restore_attempt_high_water(entry)
                for entry in payload.get("terminated_miner_task_attempts", ())
            ),
            consumed_platform_tool_proxy_attempts=tuple(
                _restore_attempt_high_water(entry)
                for entry in payload.get("consumed_platform_tool_proxy_attempts", ())
            ),
        )

    def get_restore_runs_page(
        self,
        *,
        batch: MinerTaskBatchSpec,
        snapshot_received_at: datetime,
        cursor: int,
        limit: int,
    ) -> RestoreRunsPage:
        query = urlencode(
            {
                "snapshot_received_at": snapshot_received_at.isoformat(),
                "cursor": cursor,
                "limit": limit,
            }
        )
        path = f"/v1/miner-task-batches/{batch.batch_id}/restore/runs?{query}"
        response = self._get(path, timeout_seconds=_RESTORE_GET_TIMEOUT_SECONDS)
        if response.status_code != httpx.codes.OK:
            raise PlatformClientError(
                status_code=response.status_code,
                message=f"platform returned {response.status_code} for GET {path}",
            )
        payload: dict[str, Any] = response.json()
        items = tuple(
            RestoreMinerTaskRunSubmissionModel.model_validate(entry).to_domain(batch=batch)
            for entry in payload.get("items", ())
        )
        return RestoreRunsPage(
            batch_id=UUID(str(payload["batch_id"])),
            snapshot_received_at=datetime.fromisoformat(str(payload["snapshot_received_at"])),
            cursor=int(payload["cursor"]),
            limit=int(payload["limit"]),
            next_cursor=int(payload["next_cursor"]) if payload.get("next_cursor") is not None else None,
            items=items,
        )


@dataclass
class AsyncPlatformToolProxyPlatformClient(PlatformToolProxyPlatformPort):
    """Async HTTP client for platform tool proxy endpoints."""

    base_url: str
    hotkey: bt.Keypair
    timeout_seconds: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None
    grant_retry_delays_seconds: tuple[float, ...] = _PLATFORM_TOOL_PROXY_GRANT_RETRY_DELAYS_SECONDS
    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("platform base_url must not be empty")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _signed_header(self, method: str, path_qs: str, body: bytes) -> str:
        canonical = build_canonical_request(method, path_qs, body)
        signature = self.hotkey.sign(canonical)
        return f'Bittensor ss58="{self.hotkey.ss58_address}",sig="{signature.hex()}"'

    async def create_platform_tool_proxy_grant(
        self,
        *,
        batch_id: UUID,
        artifact_id: UUID,
        task_id: UUID,
        validator_session_id: UUID,
        attempt_number: int,
    ) -> PlatformToolProxyGrant:
        path = "/v1/platform-tool-proxy/grants"
        body = _json_body(
            {
                "batch_id": str(batch_id),
                "artifact_id": str(artifact_id),
                "task_id": str(task_id),
                "validator_session_id": str(validator_session_id),
                "attempt_number": attempt_number,
            }
        )
        headers = {
            "Authorization": self._signed_header("POST", path, body),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = await self._post_grant_with_transient_retry(path, body, headers)
        if response.status_code != httpx.codes.OK:
            error_code = _platform_error_code(response)
            if error_code == "platform_tool_proxy_denied":
                raise PlatformToolProxyInvocationError(
                    status_code=response.status_code,
                    error_code=error_code,
                    message=_platform_error_message(response) or "platform tool proxy grant denied",
                )
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_tool_proxy_grant_failed",
                message=_platform_error_message(response) or "platform tool proxy grant failed",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_tool_proxy_grant_failed",
                message="platform tool proxy grant response is invalid",
            ) from exc
        if not isinstance(payload, dict):
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_tool_proxy_grant_failed",
                message="platform tool proxy grant response is invalid",
            )
        token = payload.get("token")
        expires_at = payload.get("expires_at")
        if not isinstance(token, str) or not isinstance(expires_at, str):
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_tool_proxy_grant_failed",
                message="platform tool proxy grant response is invalid",
            )
        try:
            parsed_expires_at = datetime.fromisoformat(expires_at)
        except ValueError as exc:
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_tool_proxy_grant_failed",
                message="platform tool proxy grant response is invalid",
            ) from exc
        return PlatformToolProxyGrant(
            token=token,
            expires_at=parsed_expires_at,
        )

    async def _post_grant_with_transient_retry(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        attempts = len(self.grant_retry_delays_seconds) + 1
        for attempt_index in range(attempts):
            try:
                return await self._client.post(path, content=body, headers=headers)
            except httpx.HTTPError as exc:
                retry_remaining = attempt_index < attempts - 1
                transient = classify_transient_network_failure(exc)
                if transient is None or not retry_remaining:
                    raise PlatformToolProxyInvocationError(
                        status_code=0,
                        error_code="platform_tool_proxy_grant_failed",
                        message="platform tool proxy grant request failed",
                    ) from exc
                await asyncio.sleep(self.grant_retry_delays_seconds[attempt_index])
        raise RuntimeError("platform tool proxy grant retry loop exhausted without response")

    async def execute_platform_tool_proxy_tool(
        self,
        *,
        token: str,
        uid: int,
        artifact_id: UUID,
        task_id: UUID,
        validator_session_id: UUID,
        attempt_number: int,
        receipt_id: str,
        tool: ToolName,
        args: tuple[JsonValue, ...],
        kwargs: dict[str, JsonValue],
        transport_timeout_seconds: float,
    ) -> PlatformToolProxyToolResult:
        path = "/v1/platform-tool-proxy/tools/execute"
        body = _json_body(
            {
                "uid": uid,
                "artifact_id": str(artifact_id),
                "task_id": str(task_id),
                "validator_session_id": str(validator_session_id),
                "attempt_number": attempt_number,
                "receipt_id": receipt_id,
                "tool": tool,
                "args": list(args),
                "kwargs": kwargs,
            }
        )
        headers = {
            PLATFORM_TOOL_PROXY_TOKEN_HEADER: token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = await self._client.post(
                path,
                content=body,
                headers=headers,
                timeout=transport_timeout_seconds,
            )
        except httpx.ReadTimeout as exc:
            raise PlatformToolProxyToolTimeoutError(
                status_code=0,
                message="platform tool proxy execution timed out while awaiting tool result",
            ) from exc
        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            raise PlatformToolProxyInterruptedError(
                "platform tool proxy execution interrupted before a response"
            ) from exc
        except httpx.HTTPError as exc:
            raise PlatformToolProxyInvocationError(
                status_code=0,
                error_code="platform_tool_proxy_execution_failed",
                message="platform tool proxy transport failed",
            ) from exc
        if response.status_code != httpx.codes.OK:
            error_code = _platform_error_code(response)
            if error_code == "platform_interrupted":
                raise PlatformToolProxyInterruptedError(
                    _platform_error_message(response) or "platform tool proxy execution interrupted"
                )
            if error_code == "tool_timeout":
                raise PlatformToolProxyToolTimeoutError(
                    status_code=response.status_code,
                    message=_platform_error_message(response) or "tool timed out",
                )
            if error_code == "provider_failed":
                raise PlatformToolProxyProviderError(
                    status_code=response.status_code,
                    message=_platform_error_message(response) or "tool provider failed",
                )
            if error_code == "budget_exhausted":
                raise PlatformToolProxyBudgetExceededError(
                    status_code=response.status_code,
                    message=_platform_error_message(response) or "platform tool proxy budget exhausted",
                )
            if error_code in _SELECTED_PROVIDER_OR_TOOL_REQUEST_MINER_OWNED_PROXY_ERROR_CODES:
                raise PlatformToolProxyInvocationError(
                    status_code=response.status_code,
                    error_code=error_code,
                    message=(
                        _platform_error_message(response)
                        or "platform tool proxy selected-provider/tool request failed"
                    ),
                )
            if error_code in _VALIDATOR_OWNED_PLATFORM_TOOL_PROXY_ERROR_CODES:
                raise PlatformToolProxyInvocationError(
                    status_code=response.status_code,
                    error_code=error_code,
                    message=_platform_error_message(response) or "platform tool proxy control failure",
                )
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_error",
                message=(
                    _platform_error_message(response)
                    or f"platform returned {response.status_code} for POST {path}"
                ),
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_error",
                message="platform tool proxy response is invalid",
            ) from exc
        if not isinstance(payload, dict):
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_error",
                message="platform tool proxy response is invalid",
            )
        response_payload = payload.get("response")
        if not isinstance(response_payload, dict):
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_error",
                message="platform tool proxy response is invalid",
            )
        execution_payload = payload.get("execution")
        execution = None
        if isinstance(execution_payload, dict):
            try:
                execution = ToolExecutionFacts(
                    elapsed_ms=_optional_float(execution_payload.get("elapsed_ms")),
                    ttft_ms=_optional_float(execution_payload.get("ttft_ms")),
                )
            except PlatformClientError as exc:
                raise PlatformToolProxyInvocationError(
                    status_code=response.status_code,
                    error_code="platform_error",
                    message=str(exc),
                ) from exc
        try:
            actual_cost_usd = _optional_float(payload.get("actual_cost_usd"))
        except PlatformClientError as exc:
            raise PlatformToolProxyInvocationError(
                status_code=response.status_code,
                error_code="platform_error",
                message=str(exc),
            ) from exc
        return PlatformToolProxyToolResult(
            response=cast(JsonObject, response_payload),
            execution=execution,
            actual_cost_usd=actual_cost_usd,
            actual_cost_provider=(
                str(payload["actual_cost_provider"]) if payload.get("actual_cost_provider") is not None else None
            ),
        )


def _json_body(payload: JsonObject) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlatformClientError(status_code=None, message="platform tool proxy numeric field is invalid")
    return float(value)


def _restore_attempt_high_water(value: object) -> RestoreAttemptNumberHighWater:
    if not isinstance(value, dict):
        raise PlatformClientError(status_code=None, message="restore attempt high-water entry is invalid")
    entry = cast(dict[str, object], value)
    raw_artifact_id = entry.get("artifact_id")
    raw_task_id = entry.get("task_id")
    raw_max_attempt_number = entry.get("max_attempt_number")
    if not isinstance(raw_artifact_id, str | UUID) or not isinstance(raw_task_id, str | UUID):
        raise PlatformClientError(status_code=None, message="restore attempt high-water ids are invalid")
    if not isinstance(raw_max_attempt_number, int):
        raise PlatformClientError(status_code=None, message="restore attempt high-water attempt number is invalid")
    max_attempt_number = raw_max_attempt_number
    if max_attempt_number < 1:
        raise PlatformClientError(status_code=None, message="restore attempt high-water must be positive")
    return RestoreAttemptNumberHighWater(
        artifact_id=UUID(str(raw_artifact_id)),
        task_id=UUID(str(raw_task_id)),
        max_attempt_number=max_attempt_number,
    )


_SELECTED_PROVIDER_OR_TOOL_REQUEST_MINER_OWNED_PROXY_ERROR_CODES = frozenset(
    {
        "miner_credential_missing",
        "duplicate_call",
        "concurrency_exhausted",
        "unsupported_provider",
        "unsupported_model",
        "invalid_request",
    }
)

_VALIDATOR_OWNED_PLATFORM_TOOL_PROXY_ERROR_CODES = frozenset(
    {
        "platform_tool_proxy_denied",
        "platform_error",
        "platform_tool_proxy_grant_failed",
        "platform_tool_proxy_execution_failed",
    }
)


def _platform_error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    code = payload.get("error_code")
    return code if isinstance(code, str) else None


def _platform_error_message(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    return message if isinstance(message, str) and message.strip() else None


__all__ = [
    "AsyncPlatformToolProxyPlatformClient",
    "HttpPlatformClient",
    "PlatformClientError",
    "PlatformToolProxyBudgetExceededError",
    "PlatformToolProxyInterruptedError",
    "PlatformToolProxyInvocationError",
    "PlatformToolProxyProviderError",
    "PlatformToolProxyToolTimeoutError",
]
