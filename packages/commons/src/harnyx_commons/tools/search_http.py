"""Shared HTTP mechanics for direct JSON web-provider adapters."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from harnyx_commons.config.external_client import ExternalClientRetrySettings
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.retry_utils import RetryPolicy, backoff_ms
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_effective_provider_timeout_seconds

_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class JsonSearchProviderClient:
    """Own retry, concurrency, timeout, authentication, and safe request logging."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        timeout: float,
        auth_header: str,
        auth_value_prefix: str,
        client: httpx.AsyncClient | None,
        retry_policy: RetryPolicy | None,
        max_concurrent: int | None,
        include_payloads_in_logs: bool,
    ) -> None:
        if not api_key.strip():
            raise ValueError(f"{provider.title()} API key must be provided")
        self._provider = provider
        self._owns_client = client is None
        self._timeout = timeout
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)
        self._headers = {
            auth_header: f"{auth_value_prefix}{api_key}",
            "content-type": "application/json",
        }
        self._retry_policy = retry_policy or ExternalClientRetrySettings().retry_policy
        self._include_payloads_in_logs = include_payloads_in_logs
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None
        self._logger = logging.getLogger(f"harnyx_commons.tools.{provider}.calls")

    async def post_json(
        self, path: str, payload: Mapping[str, object], *, requested_timeout: float | None
    ) -> dict[str, Any]:
        timeout = platform_tool_proxy_effective_provider_timeout_seconds(self._timeout, requested_timeout)
        if self._semaphore is None:
            return await self._post_with_retries(path, payload, timeout=timeout, wait_ms=0.0)
        started = time.perf_counter()
        async with self._semaphore:
            wait_ms = (time.perf_counter() - started) * 1000
            return await self._post_with_retries(path, payload, timeout=timeout, wait_ms=wait_ms)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_with_retries(
        self, path: str, payload: Mapping[str, object], *, timeout: float, wait_ms: float
    ) -> dict[str, Any]:
        reasons: list[str] = []
        started = time.perf_counter()
        for attempt in range(self._retry_policy.attempts):
            try:
                response = await self._client.post(
                    path, headers=self._headers, json=dict(payload), timeout=timeout
                )
                response.raise_for_status()
                try:
                    raw = response.json()
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ToolProviderError(
                        "tool provider response invalid", provider=self._provider
                    ) from exc
                if not isinstance(raw, dict):
                    raise ToolProviderError("tool provider response invalid", provider=self._provider)
                log_data: dict[str, object] = {
                    "path": path,
                    "status_code": response.status_code,
                    "attempts": attempt + 1,
                    "retry_reasons": tuple(reasons),
                    "latency_ms_total": round((time.perf_counter() - started) * 1000, 2),
                    "wait_ms": round(wait_ms, 2),
                }
                extra: dict[str, object] = {"data": log_data}
                if self._include_payloads_in_logs:
                    extra["json_fields"] = {"request": {"path": path, "json": dict(payload)}}
                self._logger.info(f"{self._provider}.request.complete", extra=extra)
                return raw
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                reasons.append(f"http_{status}")
                if status not in _RETRYABLE_STATUSES or attempt + 1 >= self._retry_policy.attempts:
                    raise ToolProviderError(
                        "tool provider failed",
                        failure_code=(
                            ToolProviderFailureCode.AUTHENTICATION_FAILED
                            if status in {401, 403}
                            else ToolProviderFailureCode.PROVIDER_FAILED
                        ),
                        provider=self._provider,
                        http_status=status,
                    ) from exc
                await asyncio.sleep(_retry_delay_seconds(exc.response, attempt, self._retry_policy))
            except httpx.HTTPError as exc:
                reasons.append(exc.__class__.__name__)
                if attempt + 1 >= self._retry_policy.attempts:
                    raise ToolProviderError("tool provider failed", provider=self._provider) from exc
                await asyncio.sleep(backoff_ms(attempt, self._retry_policy) / 1000)
        raise ToolProviderError("tool provider failed", provider=self._provider)


def _retry_delay_seconds(response: httpx.Response, attempt: int, policy: RetryPolicy) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return backoff_ms(attempt, policy) / 1000


__all__ = ["JsonSearchProviderClient"]
