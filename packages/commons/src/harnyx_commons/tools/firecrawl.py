"""HTTP adapter for Firecrawl search and scrape APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from harnyx_commons.config.external_client import ExternalClientRetrySettings
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.retry_utils import RetryPolicy, backoff_ms
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_effective_provider_timeout_seconds
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    FetchPageResult,
    SearchWebResult,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)
from harnyx_miner_sdk.tools.search_provider_extra import FirecrawlFetchExtra, FirecrawlSearchExtra

_LOGGER = logging.getLogger("harnyx_commons.tools.firecrawl.calls")
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class _FirecrawlBillingPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str | None = Field(default=None, alias="id")
    credits_used: int | None = Field(default=None, alias="creditsUsed", ge=0)

    @field_validator("request_id", mode="before")
    @classmethod
    def _normalize_request_id(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class _FirecrawlSearchResultPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = Field(min_length=1)
    title: str | None = None
    description: str | None = None

    @field_validator("url")
    @classmethod
    def _normalize_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Firecrawl search result URL must be non-empty")
        return normalized

    @field_validator("title", "description", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class _FirecrawlSearchDataPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    web: tuple[_FirecrawlSearchResultPayload, ...]

    @field_validator("web", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class _FirecrawlSearchResponsePayload(_FirecrawlBillingPayload):
    model_config = ConfigDict(extra="ignore")

    success: bool
    data: _FirecrawlSearchDataPayload


class _FirecrawlScrapeMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    source_url: str | None = Field(default=None, alias="sourceURL")
    url: str | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("source_url", "url", mode="before")
    @classmethod
    def _normalize_optional_url(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Firecrawl scrape metadata URL must be a string")
        normalized = value.strip()
        return normalized or None


class _FirecrawlScrapeDataPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    markdown: str | None = None
    raw_html: str | None = Field(default=None, alias="rawHtml")
    metadata: _FirecrawlScrapeMetadataPayload = Field(default_factory=_FirecrawlScrapeMetadataPayload)

    @field_validator("markdown", "raw_html")
    @classmethod
    def _normalize_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class _FirecrawlScrapeResponsePayload(_FirecrawlBillingPayload):
    model_config = ConfigDict(extra="ignore")

    success: bool
    data: _FirecrawlScrapeDataPayload


class FirecrawlClient:
    """Async client that normalizes Firecrawl into Harnyx web-tool contracts."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        retry_policy: RetryPolicy | None = None,
        max_concurrent: int | None = None,
        include_payloads_in_logs: bool = True,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Firecrawl API key must be provided")
        self._owns_client = client is None
        self._timeout = timeout
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)
        self._api_key = api_key
        self._retry_policy = retry_policy or ExternalClientRetrySettings().retry_policy
        self._include_payloads_in_logs = include_payloads_in_logs
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None

    async def search_web(
        self,
        request: SearchWebSearchRequest,
    ) -> SearchProviderResult[SearchWebSearchResponse]:
        if request.num == 0:
            return SearchProviderResult(
                response=SearchWebSearchResponse(data=[]),
                billing=_billing(service="search"),
            )
        query = " OR ".join(f"({term.strip()})" for term in request.search_queries)
        if len(query) > 500:
            raise ValueError("Firecrawl search query must not exceed 500 characters")
        extra = request.provider_extra or FirecrawlSearchExtra()
        if not isinstance(extra, FirecrawlSearchExtra):
            raise ValueError("Firecrawl search requires FirecrawlSearchExtra")
        payload: dict[str, object] = {
            "query": query,
            "sources": ["web"],
            **extra.to_provider_payload(),
        }
        if request.num is not None:
            payload["limit"] = min(request.num, 100)
        raw = await self._post("/v2/search", payload, requested_timeout=request.timeout)
        billing = _billing_from_raw(raw, service="search")
        try:
            parsed = _FirecrawlSearchResponsePayload.model_validate(raw)
            if not parsed.success:
                raise ValueError("Firecrawl search response reported failure")
        except (ValidationError, ValueError) as exc:
            raise ToolProviderError("tool provider response invalid", provider="firecrawl", billing=billing) from exc
        return SearchProviderResult(
            response=SearchWebSearchResponse(
                data=[
                    SearchWebResult(link=item.url, title=item.title, snippet=item.description)
                    for item in parsed.data.web
                ],
            ),
            billing=billing,
        )

    async def fetch_page(
        self,
        request: FetchPageRequest,
    ) -> SearchProviderResult[FetchPageResponse]:
        extra = request.provider_extra or FirecrawlFetchExtra()
        if not isinstance(extra, FirecrawlFetchExtra):
            raise ValueError("Firecrawl fetch requires FirecrawlFetchExtra")
        raw = await self._post(
            "/v2/scrape",
            {
                "url": request.url,
                "onlyMainContent": True,
                **extra.to_provider_payload(),
            },
            requested_timeout=request.timeout,
        )
        billing = _billing_from_raw(raw, service="scrape")
        try:
            parsed = _FirecrawlScrapeResponsePayload.model_validate(raw)
            if not parsed.success:
                raise ValueError("Firecrawl scrape response reported failure")
            result_url = parsed.data.metadata.source_url or parsed.data.metadata.url or request.url
            content_by_format = {
                "markdown": parsed.data.markdown,
                "rawHtml": parsed.data.raw_html,
            }
            results: list[FetchPageResult] = []
            for requested_format in extra.formats:
                content = content_by_format[requested_format]
                if content is None:
                    raise ValueError(
                        f"Firecrawl scrape response omitted requested format {requested_format!r}"
                    )
                results.append(
                    FetchPageResult(
                        url=result_url,
                        content=content,
                        title=parsed.data.metadata.title,
                    )
                )
        except (ValidationError, ValueError) as exc:
            raise ToolProviderError("tool provider response invalid", provider="firecrawl", billing=billing) from exc
        return SearchProviderResult(
            response=FetchPageResponse(data=results),
            billing=billing,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(
        self,
        path: str,
        payload: Mapping[str, object],
        *,
        requested_timeout: float | None,
    ) -> dict[str, Any]:
        timeout = platform_tool_proxy_effective_provider_timeout_seconds(self._timeout, requested_timeout)
        if self._semaphore is None:
            return await self._post_with_retries(path, payload, timeout=timeout, wait_ms=0.0)
        wait_started = time.perf_counter()
        async with self._semaphore:
            wait_ms = (time.perf_counter() - wait_started) * 1000
            return await self._post_with_retries(path, payload, timeout=timeout, wait_ms=wait_ms)

    async def _post_with_retries(
        self,
        path: str,
        payload: Mapping[str, object],
        *,
        timeout: float,
        wait_ms: float,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        started = time.perf_counter()
        for attempt in range(self._retry_policy.attempts):
            try:
                response = await self._client.post(
                    path,
                    headers={"authorization": f"Bearer {self._api_key}", "content-type": "application/json"},
                    json=dict(payload),
                    timeout=timeout,
                )
                response.raise_for_status()
                try:
                    raw = response.json()
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ToolProviderError(
                        "tool provider response invalid",
                        provider="firecrawl",
                    ) from exc
                if not isinstance(raw, dict):
                    raise ToolProviderError("tool provider response invalid", provider="firecrawl")
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
                _LOGGER.info("firecrawl.request.complete", extra=extra)
                return raw
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                reasons.append(f"http_{status}")
                if status not in _RETRYABLE_STATUSES or attempt + 1 >= self._retry_policy.attempts:
                    raise ToolProviderError(
                        "tool provider failed",
                        failure_code=(
                            ToolProviderFailureCode.AUTHENTICATION_FAILED
                            if status == 401
                            else ToolProviderFailureCode.PROVIDER_FAILED
                        ),
                        provider="firecrawl",
                        http_status=status,
                    ) from exc
                await asyncio.sleep(_retry_delay_seconds(exc.response, attempt, self._retry_policy))
            except httpx.HTTPError as exc:
                reasons.append(exc.__class__.__name__)
                if attempt + 1 >= self._retry_policy.attempts:
                    raise ToolProviderError("tool provider failed", provider="firecrawl") from exc
                await asyncio.sleep(backoff_ms(attempt, self._retry_policy) / 1000)
        raise ToolProviderError("tool provider failed", provider="firecrawl")


def _billing_from_raw(raw: Mapping[str, Any], *, service: str) -> ProviderBillingMetadata:
    parsed = _FirecrawlBillingPayload.model_validate(raw)
    return _billing(service=service, request_id=parsed.request_id, credits_used=parsed.credits_used)


def _billing(
    *,
    service: str,
    request_id: str | None = None,
    credits_used: int | None = None,
) -> ProviderBillingMetadata:
    return ProviderBillingMetadata(
        actual_cost_provider="firecrawl",
        source="response_body" if request_id is not None or credits_used is not None else "missing_provider_metadata",
        provider_request_id=request_id,
        usage_count=credits_used,
        service=service,
    )


def _retry_delay_seconds(response: httpx.Response, attempt: int, policy: RetryPolicy) -> float:
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    return max(0.0, retry_at.timestamp() - time.time())
                except (TypeError, ValueError, OverflowError):
                    pass
    return backoff_ms(attempt, policy) / 1000


__all__ = ["FirecrawlClient"]
