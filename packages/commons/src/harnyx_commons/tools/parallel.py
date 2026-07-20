"""HTTP client adapter for the Parallel Search and Extract APIs."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from harnyx_commons.config.external_client import ExternalClientRetrySettings
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.pricing import price_parallel_extract, price_parallel_search
from harnyx_commons.llm.retry_utils import RetryPolicy, backoff_ms
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    FetchPageResult,
    SearchAiResult,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchWebResult,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)

_LOGGER = logging.getLogger("harnyx_commons.tools.parallel.calls")


class _ParallelSearchResultPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = Field(min_length=1)
    title: str | None = None
    publish_date: str | None = None
    excerpt_text: str | None = Field(default=None, validation_alias="excerpts")

    @field_validator("url", "title", "publish_date", mode="before")
    @classmethod
    def _coerce_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("excerpt_text", mode="before")
    @classmethod
    def _coalesce_excerpts(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError("parallel excerpts must be a list")
        excerpts = [str(item).strip() for item in value if str(item).strip()]
        if not excerpts:
            return None
        return "\n\n".join(excerpts)


class _ParallelSearchResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    search_id: str | None = None
    results: tuple[_ParallelSearchResultPayload, ...]
    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None

    @field_validator("search_id", mode="before")
    @classmethod
    def _coerce_search_id(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class _ParallelExtractResultPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = Field(min_length=1)
    title: str | None = None
    full_content: str = Field(min_length=1)

    @field_validator("url", "title", "full_content", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class _ParallelExtractResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    extract_id: str | None = None
    results: tuple[_ParallelExtractResultPayload, ...]
    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None

    @field_validator("extract_id", mode="before")
    @classmethod
    def _coerce_extract_id(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class ParallelClient:
    """Lightweight async client for the Parallel Search and Extract APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        retry_policy: RetryPolicy | None = None,
        max_concurrent: int | None = None,
        include_payloads_in_logs: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("Parallel API key must be provided")
        normalized_base = base_url.rstrip("/")
        self._owns_client = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            base_url=normalized_base,
            timeout=timeout,
        )
        self._api_key = api_key
        self._retry_policy = retry_policy or ExternalClientRetrySettings().retry_policy
        self._include_payloads_in_logs = include_payloads_in_logs
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None
        )

    async def search_web(
        self,
        request: SearchWebSearchRequest,
    ) -> SearchProviderResult[SearchWebSearchResponse]:
        payload: dict[str, object] = {
            "search_queries": list(request.search_queries),
        }
        if request.num is not None:
            payload["max_results"] = request.num
        data = _ParallelSearchResponsePayload.model_validate(await self._post("/v1beta/search", payload))
        response = SearchWebSearchResponse(
            data=[
                SearchWebResult(
                    link=result.url,
                    snippet=result.excerpt_text,
                    title=result.title,
                )
                for result in data.results
            ],
            attempts=data.attempts,
            retry_reasons=data.retry_reasons,
        )
        return SearchProviderResult(
            response=response,
            billing=_parallel_search_billing(
                billable_units=len(data.results),
                provider_request_id=data.search_id,
            ),
        )

    async def search_ai(
        self,
        request: SearchAiSearchRequest,
    ) -> SearchProviderResult[SearchAiSearchResponse]:
        payload: dict[str, object] = {
            "objective": request.prompt,
            "max_results": request.count,
        }
        data = _ParallelSearchResponsePayload.model_validate(await self._post("/v1beta/search", payload))
        response = SearchAiSearchResponse(
            data=[
                SearchAiResult(
                    url=result.url,
                    note=result.excerpt_text,
                    title=result.title,
                )
                for result in data.results
            ],
            attempts=data.attempts,
            retry_reasons=data.retry_reasons,
        )
        return SearchProviderResult(
            response=response,
            billing=_parallel_search_billing(
                billable_units=len(data.results),
                provider_request_id=data.search_id,
            ),
        )

    async def fetch_page(
        self,
        request: FetchPageRequest,
    ) -> SearchProviderResult[FetchPageResponse]:
        payload = {
            "urls": [request.url],
            "full_content": True,
            "excerpts": False,
        }
        data = _ParallelExtractResponsePayload.model_validate(await self._post("/v1beta/extract", payload))
        if len(data.results) != 1:
            raise ToolProviderError("tool provider failed")
        result = data.results[0]
        response = FetchPageResponse(
            data=[
                FetchPageResult(
                    url=result.url,
                    content=result.full_content,
                    title=result.title,
                )
            ],
            attempts=data.attempts,
            retry_reasons=data.retry_reasons,
        )
        return SearchProviderResult(
            response=response,
            billing=_parallel_fetch_billing(
                billable_units=len(data.results),
                provider_request_id=data.extract_id,
            ),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, path: str, payload: Mapping[str, object]) -> dict[str, Any]:
        if self._semaphore is None:
            return await self._post_with_retries(path, payload, wait_ms=0.0)

        wait_start = time.perf_counter()
        async with self._semaphore:
            wait_ms = (time.perf_counter() - wait_start) * 1000
            return await self._post_with_retries(path, payload, wait_ms=wait_ms)

    async def _post_with_retries(
        self,
        path: str,
        payload: Mapping[str, object],
        *,
        wait_ms: float,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        total_latency_ms = 0.0
        for attempt in range(self._retry_policy.attempts):
            attempt_start = time.perf_counter()
            try:
                response = await self._client.post(
                    path,
                    headers={
                        "x-api-key": self._api_key,
                        "content-type": "application/json",
                    },
                    json=dict(payload),
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("parallel response was not an object")
                total_latency_ms += (time.perf_counter() - attempt_start) * 1000
                data["attempts"] = attempt + 1
                data["retry_reasons"] = tuple(reasons)
                log_extra: dict[str, object] = {
                    "data": {
                        "path": path,
                        "status_code": response.status_code,
                        "attempts": attempt + 1,
                        "latency_ms_total": round(total_latency_ms, 2),
                        "retry_reasons": tuple(reasons),
                        "wait_ms": round(wait_ms, 2),
                    }
                }
                if self._include_payloads_in_logs:
                    log_extra["json_fields"] = {
                        "request": {"path": path, "json": dict(payload)},
                        "response_raw": data,
                    }
                _LOGGER.info("parallel.request.complete", extra=log_extra)
                return data
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                reasons.append(f"http_{status}")
                if not _should_retry_status(status) or not self._should_retry(attempt):
                    raise ToolProviderError(
                        "tool provider failed",
                        failure_code=(
                            ToolProviderFailureCode.AUTHENTICATION_FAILED
                            if status == 401
                            else ToolProviderFailureCode.PROVIDER_FAILED
                        ),
                        provider="parallel",
                        http_status=status,
                    ) from exc
                await self._sleep(attempt)
            except httpx.HTTPError as exc:
                reasons.append(exc.__class__.__name__)
                if not self._should_retry(attempt):
                    raise ToolProviderError("tool provider failed") from exc
                await self._sleep(attempt)

        raise ToolProviderError("tool provider failed")

    def _should_retry(self, attempt: int) -> bool:
        return attempt + 1 < self._retry_policy.attempts

    async def _sleep(self, attempt: int) -> None:
        await asyncio.sleep(backoff_ms(attempt, self._retry_policy) / 1000)


def _should_retry_status(status: int | None) -> bool:
    return status == 429 or (status is not None and status >= 500)


def _parallel_search_billing(
    *,
    billable_units: int,
    provider_request_id: str | None,
) -> ProviderBillingMetadata:
    return _parallel_billing(
        billable_units=billable_units,
        actual_cost_usd=price_parallel_search(billable_results=billable_units),
        provider_request_id=provider_request_id,
    )


def _parallel_fetch_billing(
    *,
    billable_units: int,
    provider_request_id: str | None,
) -> ProviderBillingMetadata:
    return _parallel_billing(
        billable_units=billable_units,
        actual_cost_usd=price_parallel_extract(url_count=billable_units),
        provider_request_id=provider_request_id,
    )


def _parallel_billing(
    *,
    billable_units: int,
    actual_cost_usd: float,
    provider_request_id: str | None,
) -> ProviderBillingMetadata:
    return ProviderBillingMetadata(
        actual_cost_provider="parallel",
        actual_cost_usd=actual_cost_usd,
        billable_units=billable_units,
        provider_request_id=provider_request_id,
        source="response_results",
    )


__all__ = ["ParallelClient"]
