"""HTTP client adapter for the Parallel Search and Extract APIs."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from harnyx_commons.config.external_client import ExternalClientRetrySettings
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.pricing import price_parallel_extract, price_parallel_search
from harnyx_commons.llm.retry_utils import RetryPolicy, backoff_ms
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_effective_provider_timeout_seconds
from harnyx_commons.tools.extraction_models import (
    ExtractedPage,
    ExtractPagesRequest,
    ExtractPagesResponse,
    PageExtractionError,
    PageExtractionWarning,
    canonicalize_extraction_url,
)
from harnyx_commons.tools.provider_billing import (
    ProviderBillingMetadata,
    ProviderBillingSource,
    SearchProviderResult,
)
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
from harnyx_miner_sdk.tools.search_provider_extra import ParallelFetchExtra, ParallelSearchExtra

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
    model_config = ConfigDict(extra="ignore", strict=True)

    url: str = Field(min_length=1)
    title: str | None = None
    publish_date: str | None = None
    excerpts: tuple[str, ...] = ()
    full_content: str | None = None

    @field_validator("excerpts", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class _ParallelExtractErrorPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    url: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    http_status_code: int | None = None
    content: str | None = None


class _ParallelWarningPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str = Field(min_length=1)
    message: str = Field(min_length=1)


class _ParallelUsageItemPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)
    count: int = Field(ge=0)


class _ParallelExtractResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    extract_id: str = Field(min_length=1)
    results: tuple[_ParallelExtractResultPayload, ...]
    errors: tuple[_ParallelExtractErrorPayload, ...]
    warnings: tuple[_ParallelWarningPayload, ...] | None = None
    usage: tuple[_ParallelUsageItemPayload, ...] | None = None
    session_id: str = Field(min_length=1)
    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None

    @field_validator("results", "errors", "warnings", "usage", "retry_reasons", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class _ParallelExtractBillingPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    extract_id: str | None = Field(default=None, min_length=1)
    usage: tuple[_ParallelUsageItemPayload, ...] | None = None

    @field_validator("usage", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


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
        self._timeout = timeout
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
        extra = request.provider_extra or ParallelSearchExtra()
        if not isinstance(extra, ParallelSearchExtra):
            raise ValueError("Parallel search requires ParallelSearchExtra")
        payload: dict[str, object] = {
            "search_queries": list(request.search_queries),
        }
        extra_payload = extra.to_provider_payload()
        advanced_settings = {
            key: extra_payload.pop(key)
            for key in ("source_policy", "fetch_policy", "excerpt_settings", "location")
            if key in extra_payload
        }
        payload.update(extra_payload)
        if request.num is not None:
            advanced_settings["max_results"] = request.num
        if advanced_settings:
            payload["advanced_settings"] = advanced_settings
        data = _ParallelSearchResponsePayload.model_validate(
            await self._post("/v1/search", payload, requested_timeout=request.timeout)
        )
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
                mode=extra.mode,
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
        data = _ParallelSearchResponsePayload.model_validate(
            await self._post("/v1beta/search", payload, requested_timeout=request.timeout)
        )
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
        extra = request.provider_extra or ParallelFetchExtra()
        if not isinstance(extra, ParallelFetchExtra):
            raise ValueError("Parallel fetch requires ParallelFetchExtra")
        extraction = await self._extract_pages(
            ExtractPagesRequest(urls=(request.url,)),
            requested_timeout=request.timeout,
            provider_extra=extra,
        )
        if extraction.response.errors:
            error = extraction.response.errors[0]
            raise ToolProviderError(
                "tool provider failed",
                provider="parallel",
                http_status=error.http_status_code,
            )
        if len(extraction.response.pages) != 1:
            raise ToolProviderError("tool provider failed", provider="parallel")
        result = extraction.response.pages[0]
        content = result.content or "\n\n".join(excerpt.strip() for excerpt in result.excerpts if excerpt.strip())
        if not content:
            raise ToolProviderError("tool provider failed", provider="parallel")
        response = FetchPageResponse(
            data=[
                FetchPageResult(
                    url=result.url,
                    content=content,
                    title=result.title,
                )
            ],
            attempts=extraction.response.attempts,
            retry_reasons=extraction.response.retry_reasons,
        )
        return SearchProviderResult(
            response=response,
            billing=extraction.billing,
        )

    async def extract_pages(
        self,
        request: ExtractPagesRequest,
    ) -> SearchProviderResult[ExtractPagesResponse]:
        return await self._extract_pages(request, requested_timeout=None, provider_extra=None)

    async def _extract_pages(
        self,
        request: ExtractPagesRequest,
        *,
        requested_timeout: float | None,
        provider_extra: ParallelFetchExtra | None,
    ) -> SearchProviderResult[ExtractPagesResponse]:
        advanced_settings: dict[str, object] = {
            "full_content": (
                True
                if request.max_chars_per_result is None
                else {"max_chars_per_result": request.max_chars_per_result}
            )
        }
        if provider_extra is not None:
            extra_payload = provider_extra.to_provider_payload()
            if "full_content" in extra_payload:
                advanced_settings["full_content"] = extra_payload.pop("full_content")
            for key in ("fetch_policy", "excerpt_settings"):
                if key in extra_payload:
                    advanced_settings[key] = extra_payload.pop(key)
        fetch_policy: dict[str, object] = {}
        if request.max_age_seconds is not None:
            fetch_policy["max_age_seconds"] = request.max_age_seconds
        if request.disable_cache_fallback is not None:
            fetch_policy["disable_cache_fallback"] = request.disable_cache_fallback
        if fetch_policy:
            advanced_settings["fetch_policy"] = fetch_policy
        payload: dict[str, object] = {
            "urls": list(request.urls),
            "advanced_settings": advanced_settings,
        }
        if provider_extra is not None and provider_extra.max_chars_total is not None:
            payload["max_chars_total"] = provider_extra.max_chars_total
        if request.objective is not None:
            payload["objective"] = request.objective
        if provider_extra is not None and provider_extra.objective is not None:
            payload["objective"] = provider_extra.objective
        if request.client_model is not None:
            payload["client_model"] = request.client_model
        try:
            raw_response = await self._post(
                "/v1/extract",
                payload,
                requested_timeout=requested_timeout,
            )
        except ToolProviderError:
            raise
        except (ValueError, RuntimeError) as exc:
            raise ToolProviderError(
                "tool provider response invalid",
                provider="parallel",
                billing=_parallel_fetch_billing(
                    billable_units=len(request.urls),
                    provider_request_id=None,
                    source="request_body",
                ),
            ) from exc
        billing = _parallel_extract_billing_from_raw(
            raw_response,
            submitted_url_count=len(request.urls),
        )
        try:
            data = _ParallelExtractResponsePayload.model_validate(raw_response)
            _validate_extract_url_partition(request.urls, data)
        except (ValidationError, ValueError) as exc:
            raise ToolProviderError(
                "tool provider response invalid",
                provider="parallel",
                billing=billing,
            ) from exc
        return SearchProviderResult(
            response=ExtractPagesResponse(
                pages=tuple(
                    ExtractedPage(
                        url=result.url,
                        title=result.title,
                        publish_date=result.publish_date,
                        excerpts=result.excerpts,
                        content=result.full_content,
                    )
                    for result in data.results
                ),
                errors=tuple(
                    PageExtractionError(
                        url=error.url,
                        error_type=error.error_type,
                        http_status_code=error.http_status_code,
                        content=error.content,
                    )
                    for error in data.errors
                ),
                warnings=tuple(
                    PageExtractionWarning(warning_type=warning.type, message=warning.message)
                    for warning in (data.warnings or ())
                ),
                session_id=data.session_id,
                attempts=data.attempts,
                retry_reasons=data.retry_reasons,
            ),
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
        provider_timeout = platform_tool_proxy_effective_provider_timeout_seconds(
            self._timeout,
            requested_timeout,
        )
        if self._semaphore is None:
            return await self._post_with_retries(path, payload, wait_ms=0.0, timeout=provider_timeout)

        wait_start = time.perf_counter()
        async with self._semaphore:
            wait_ms = (time.perf_counter() - wait_start) * 1000
            return await self._post_with_retries(path, payload, wait_ms=wait_ms, timeout=provider_timeout)

    async def _post_with_retries(
        self,
        path: str,
        payload: Mapping[str, object],
        *,
        wait_ms: float,
        timeout: float,
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
                    timeout=timeout,
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
    mode: Literal["turbo", "basic", "advanced"] | None = None,
) -> ProviderBillingMetadata:
    return _parallel_billing(
        billable_units=billable_units,
        actual_cost_usd=price_parallel_search(billable_results=billable_units, mode=mode),
        provider_request_id=provider_request_id,
    )


def _parallel_fetch_billing(
    *,
    billable_units: int,
    provider_request_id: str | None,
    source: ProviderBillingSource = "response_results",
) -> ProviderBillingMetadata:
    return _parallel_billing(
        billable_units=billable_units,
        actual_cost_usd=price_parallel_extract(url_count=billable_units),
        provider_request_id=provider_request_id,
        source=source,
        service="extract",
    )


def _parallel_billing(
    *,
    billable_units: int,
    actual_cost_usd: float,
    provider_request_id: str | None,
    source: ProviderBillingSource = "response_results",
    service: str | None = None,
) -> ProviderBillingMetadata:
    return ProviderBillingMetadata(
        actual_cost_provider="parallel",
        actual_cost_usd=actual_cost_usd,
        billable_units=billable_units,
        provider_request_id=provider_request_id,
        usage_count=billable_units,
        service=service,
        source=source,
    )


def _parallel_extract_billable_units(
    *,
    submitted_url_count: int,
    usage: tuple[_ParallelUsageItemPayload, ...] | None,
) -> tuple[int, ProviderBillingSource]:
    supported_usage = tuple(
        item.count for item in (usage or ()) if item.name == "sku_extract_excerpts"
    )
    if supported_usage:
        return sum(supported_usage), "response_body"
    return submitted_url_count, "request_body"


def _parallel_extract_billing_from_raw(
    raw_response: Mapping[str, object],
    *,
    submitted_url_count: int,
) -> ProviderBillingMetadata:
    try:
        payload = _ParallelExtractBillingPayload.model_validate(raw_response)
    except ValidationError:
        return _parallel_fetch_billing(
            billable_units=submitted_url_count,
            provider_request_id=None,
            source="request_body",
        )
    billable_units, billing_source = _parallel_extract_billable_units(
        submitted_url_count=submitted_url_count,
        usage=payload.usage,
    )
    return _parallel_fetch_billing(
        billable_units=billable_units,
        provider_request_id=payload.extract_id,
        source=billing_source,
    )


def _validate_extract_url_partition(
    requested_urls: tuple[str, ...],
    response: _ParallelExtractResponsePayload,
) -> None:
    requested = {canonicalize_extraction_url(url) for url in requested_urls}
    results = tuple(canonicalize_extraction_url(item.url) for item in response.results)
    errors = tuple(canonicalize_extraction_url(item.url) for item in response.errors)
    result_urls = set(results)
    error_urls = set(errors)
    if len(result_urls) != len(results) or len(error_urls) != len(errors):
        raise ValueError("parallel extract response contained duplicate URLs")
    if result_urls & error_urls:
        raise ValueError("parallel extract response URL appeared in results and errors")
    if result_urls | error_urls != requested:
        raise ValueError("parallel extract response did not partition requested URLs")
__all__ = ["ParallelClient"]
