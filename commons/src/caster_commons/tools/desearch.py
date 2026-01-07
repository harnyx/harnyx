"""HTTP client adapter for the DeSearch API."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx

from caster_commons.config.external_client import ExternalClientRetrySettings
from caster_commons.llm.retry_utils import RetryPolicy, backoff_ms
from caster_commons.tools.search_models import (
    SearchWebSearchRequest,
    SearchWebSearchResponse,
    SearchXResult,
    SearchXSearchRequest,
    SearchXSearchResponse,
)

_LOGGER = logging.getLogger("caster_commons.tools.desearch.calls")


class DeSearchClient:
    """Lightweight async client for DeSearch endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        retry_policy: RetryPolicy | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeSearch API key must be provided")
        normalized_base = base_url.rstrip("/")
        self._owns_client = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            base_url=normalized_base,
            timeout=timeout,
        )
        self._api_key = api_key
        self._retry_policy = retry_policy or ExternalClientRetrySettings().retry_policy
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None
        )

    async def _post(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        *,
        expect_data: bool = True,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return await self._request(
            "post",
            path,
            json_payload=dict(payload),
            expect_data=expect_data,
            allow_not_found=allow_not_found,
        )

    async def _get(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        *,
        expect_data: bool = True,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return await self._request(
            "get",
            path,
            params=dict(params),
            expect_data=expect_data,
            allow_not_found=allow_not_found,
        )

    # Convenience wrappers ------------------------------------------------------------------------

    async def search_links_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        params = request.to_query_params()
        data = await self._get("web", params)
        if data is None:
            raise RuntimeError("desearch web search returned empty response")
        return SearchWebSearchResponse.model_validate(data)

    async def search_links_twitter(
        self,
        request: SearchXSearchRequest,
    ) -> SearchXSearchResponse:
        params = request.to_query_params()
        data = await self._get("twitter", params)
        if data is None:
            raise RuntimeError("desearch twitter search returned empty response")
        if isinstance(data, list):
            data = {"data": data}
        return SearchXSearchResponse.model_validate(data)

    async def fetch_twitter_post(self, *, post_id: str) -> SearchXResult | None:
        if not post_id:
            raise ValueError("desearch fetch_twitter_post requires non-empty id")
        data = await self._get(
            "twitter/post",
            {"id": post_id},
            expect_data=False,
            allow_not_found=True,
        )
        if data is None:
            return None
        return SearchXResult.model_validate(data)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # internal

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        expect_data: bool = True,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        if self._semaphore is None:
            return await self._request_with_retries(
                method,
                path,
                json_payload=json_payload,
                params=params,
                expect_data=expect_data,
                allow_not_found=allow_not_found,
                wait_ms=0.0,
            )

        wait_start = time.perf_counter()
        async with self._semaphore:
            wait_ms = (time.perf_counter() - wait_start) * 1000
            return await self._request_with_retries(
                method,
                path,
                json_payload=json_payload,
                params=params,
                expect_data=expect_data,
                allow_not_found=allow_not_found,
                wait_ms=wait_ms,
            )

    async def _request_with_retries(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None,
        params: dict[str, Any] | None,
        expect_data: bool,
        allow_not_found: bool,
        wait_ms: float,
    ) -> dict[str, Any] | None:
        reasons: list[str] = []
        total_latency_ms = 0.0
        for attempt in range(self._retry_policy.attempts):
            attempt_start = time.perf_counter()
            try:
                resp = await self._send(method, path, json_payload=json_payload, params=params)
                raw_response, data = await self._parse_response(resp)
                total_latency_ms += (time.perf_counter() - attempt_start) * 1000
                if expect_data:
                    if self._missing_data(data):
                        if self._should_retry(attempt):
                            reasons.append("missing_data")
                            await self._sleep(attempt)
                            continue
                        raise RuntimeError("desearch returned missing data")
                    if self._empty_payload(data):
                        if self._should_retry(attempt):
                            reasons.append("empty_data")
                            await self._sleep(attempt)
                            continue
                data["attempts"] = attempt + 1
                data["retry_reasons"] = tuple(reasons)
                _LOGGER.info(
                    "desearch.request.complete",
                    extra={
                        "data": {
                            "method": method.upper(),
                            "path": path,
                            "status_code": resp.status_code,
                            "attempts": attempt + 1,
                            "latency_ms_total": round(total_latency_ms, 2),
                            "retry_reasons": tuple(reasons),
                            "result_count": len(data["data"]) if isinstance(data.get("data"), list) else None,
                            "wait_ms": round(wait_ms, 2),
                        },
                        "json_fields": {
                            "request": {
                                "method": method.upper(),
                                "path": path,
                                "params": params,
                                "json": json_payload,
                            },
                            "response_raw": raw_response,
                        },
                    },
                )
                return data
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
                status = exc.response.status_code if exc.response else None
                if allow_not_found and status == 404:
                    total_latency_ms += (time.perf_counter() - attempt_start) * 1000
                    _LOGGER.info(
                        "desearch.request.not_found",
                        extra={
                            "data": {
                                "method": method.upper(),
                                "path": path,
                                "status_code": status,
                                "attempts": attempt + 1,
                                "latency_ms_total": round(total_latency_ms, 2),
                                "retry_reasons": tuple(reasons),
                                "wait_ms": round(wait_ms, 2),
                            },
                            "json_fields": {
                                "request": {
                                    "method": method.upper(),
                                    "path": path,
                                    "params": params,
                                    "json": json_payload,
                                },
                            },
                        },
                    )
                    return None
                retryable = status == 429 or (status is not None and status >= 500)
                reasons.append(f"http_{status}")
                if not (retryable and self._should_retry(attempt)):
                    raise RuntimeError(f"desearch request failed with status {status}") from exc
                await self._sleep(attempt)
            except httpx.HTTPError as exc:  # pragma: no cover
                reasons.append(exc.__class__.__name__)
                if not self._should_retry(attempt):
                    raise RuntimeError(f"desearch request error: {exc}") from exc
                await self._sleep(attempt)
        raise RuntimeError(
            f"desearch request failed after {self._retry_policy.attempts} attempts: {reasons}"
        )

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None,
        params: dict[str, Any] | None,
    ) -> httpx.Response:
        headers = {"Authorization": self._api_key}
        if method == "post":
            headers["content-type"] = "application/json"
            return await self._client.post(path, headers=headers, json=json_payload)
        return await self._client.get(path, headers=headers, params=params)

    @staticmethod
    async def _parse_response(resp: httpx.Response) -> tuple[object, dict[str, Any]]:
        resp.raise_for_status()
        raw = resp.json()
        data = raw
        if isinstance(data, list):
            data = {"data": data}
        if not isinstance(data, dict):
            raise RuntimeError("desearch response was not an object")
        return raw, data

    @staticmethod
    def _missing_data(data: dict[str, Any]) -> bool:
        return data.get("data") is None

    @staticmethod
    def _empty_payload(data: dict[str, Any]) -> bool:
        payload = data.get("data")
        return isinstance(payload, list) and len(payload) == 0

    def _should_retry(self, attempt: int) -> bool:
        return attempt + 1 < self._retry_policy.attempts

    async def _sleep(self, attempt: int) -> None:
        backoff = backoff_ms(attempt, self._retry_policy)
        await asyncio.sleep(backoff / 1000)


__all__ = ["DeSearchClient"]
