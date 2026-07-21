"""HTTP client adapter for the DeSearch API."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, cast

import httpx
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel, ConfigDict, Field, field_validator

from harnyx_commons.config.external_client import ExternalClientRetrySettings
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.retry_utils import RetryPolicy, backoff_ms
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_effective_provider_timeout_seconds
from harnyx_commons.tools.desearch_ai_protocol import (
    DeSearchAiDocsResponse,
    parse_desearch_ai_response,
)
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
    SearchXResult,
    SearchXSearchRequest,
    SearchXSearchResponse,
)

_LOGGER = logging.getLogger("harnyx_commons.tools.desearch.calls")
_DESEARCH_AI_MIN_COUNT = 10
_DESEARCH_AI_MAX_COUNT = 200


class _DeSearchCrawlResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=1)
    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()


class _DeSearchRetryMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None


class DeSearchAiTool(StrEnum):
    WEB = "web"
    HACKER_NEWS = "hackernews"
    REDDIT = "reddit"
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    ARXIV = "arxiv"


class DeSearchAiModel(StrEnum):
    NOVA = "NOVA"
    ORBIT = "ORBIT"
    HORIZON = "HORIZON"


class DeSearchAiDateFilter(StrEnum):
    PAST_24_HOURS = "PAST_24_HOURS"
    PAST_2_DAYS = "PAST_2_DAYS"
    PAST_WEEK = "PAST_WEEK"
    PAST_2_WEEKS = "PAST_2_WEEKS"
    PAST_MONTH = "PAST_MONTH"
    PAST_2_MONTHS = "PAST_2_MONTHS"
    PAST_YEAR = "PAST_YEAR"
    PAST_2_YEARS = "PAST_2_YEARS"


class DeSearchAiResultType(StrEnum):
    ONLY_LINKS = "ONLY_LINKS"
    LINKS_WITH_FINAL_SUMMARY = "LINKS_WITH_FINAL_SUMMARY"


_MINER_AI_TOOLS: tuple[DeSearchAiTool, ...] = (
    DeSearchAiTool.WEB,
    DeSearchAiTool.HACKER_NEWS,
    DeSearchAiTool.REDDIT,
    DeSearchAiTool.WIKIPEDIA,
    DeSearchAiTool.YOUTUBE,
    DeSearchAiTool.ARXIV,
)


@dataclass(frozen=True, slots=True)
class DeSearchTwitterSearchPage:
    request: SearchXSearchRequest
    response: SearchXSearchResponse
    page: int
    cursor: int | None
    posts: list[SearchXResult]
    post_ids: list[int]


@dataclass(frozen=True, slots=True)
class DeSearchWebSearchPage:
    request: SearchWebSearchRequest
    response: SearchWebSearchResponse
    page: int
    start: int
    results: list[SearchWebResult]


@dataclass(frozen=True, slots=True)
class _DeSearchRequestData:
    data: dict[str, Any]
    billing: ProviderBillingMetadata | None


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
        include_payloads_in_logs: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("DeSearch API key must be provided")
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

    async def _post(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        *,
        expect_data: bool = True,
        allow_not_found: bool = False,
        response_format: str = "json",
        requested_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        result = await self._request_with_billing(
            "post",
            path,
            json_payload=dict(payload),
            expect_data=expect_data,
            allow_not_found=allow_not_found,
            response_format=response_format,
            requested_timeout=requested_timeout,
        )
        return result.data if result is not None else None

    async def _post_with_billing(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        *,
        expect_data: bool = True,
        allow_not_found: bool = False,
        response_format: str = "json",
        requested_timeout: float | None = None,
    ) -> _DeSearchRequestData | None:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return await self._request_with_billing(
            "post",
            path,
            json_payload=dict(payload),
            expect_data=expect_data,
            allow_not_found=allow_not_found,
            response_format=response_format,
            requested_timeout=requested_timeout,
        )

    async def _get(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        *,
        expect_data: bool = True,
        allow_not_found: bool = False,
        response_format: str = "json",
        requested_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        result = await self._request_with_billing(
            "get",
            path,
            params=dict(params),
            expect_data=expect_data,
            allow_not_found=allow_not_found,
            response_format=response_format,
            requested_timeout=requested_timeout,
        )
        return result.data if result is not None else None

    async def _get_with_billing(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        *,
        expect_data: bool = True,
        allow_not_found: bool = False,
        response_format: str = "json",
        requested_timeout: float | None = None,
    ) -> _DeSearchRequestData | None:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return await self._request_with_billing(
            "get",
            path,
            params=dict(params),
            expect_data=expect_data,
            allow_not_found=allow_not_found,
            response_format=response_format,
            requested_timeout=requested_timeout,
        )

    # Convenience wrappers ------------------------------------------------------------------------

    async def search_web(
        self,
        request: SearchWebSearchRequest,
    ) -> SearchProviderResult[SearchWebSearchResponse]:
        data = await self._search_links_web_page_with_billing(request, start=None)
        if data is None:
            raise ToolProviderError("tool provider failed")
        response = SearchWebSearchResponse.model_validate(data.data)
        return SearchProviderResult(
            response=response,
            billing=_require_provider_billing(
                data.billing,
                provider="desearch",
                billable_units=len(response.data),
            ),
        )

    async def search_ai(
        self,
        request: SearchAiSearchRequest,
    ) -> SearchProviderResult[SearchAiSearchResponse]:
        raw = await self.ai_search_with_billing(
            prompt=request.prompt,
            tools=_MINER_AI_TOOLS,
            model=DeSearchAiModel.HORIZON,
            count=request.count,
            date_filter=None,
            result_type=DeSearchAiResultType.LINKS_WITH_FINAL_SUMMARY,
            system_message="",
            requested_timeout=request.timeout,
        )
        metadata = _DeSearchRetryMetadataPayload.model_validate(raw.data)
        response = SearchAiSearchResponse(
            data=_extract_desearch_ai_results(raw.data),
            attempts=metadata.attempts,
            retry_reasons=metadata.retry_reasons,
        )
        return SearchProviderResult(
            response=response,
            billing=_require_provider_billing(
                raw.billing,
                provider="desearch",
                billable_units=len(response.data),
            ),
        )

    async def fetch_page(
        self,
        request: FetchPageRequest,
    ) -> SearchProviderResult[FetchPageResponse]:
        data = await self._get_with_billing(
            "web/crawl",
            {"url": request.url, "format": "text"},
            expect_data=False,
            response_format="text",
            requested_timeout=request.timeout,
        )
        if data is None:
            raise ToolProviderError("tool provider failed")
        payload = _DeSearchCrawlResponsePayload.model_validate(data.data)
        response = FetchPageResponse(
            data=[FetchPageResult(url=request.url, content=payload.content)],
            attempts=payload.attempts,
            retry_reasons=payload.retry_reasons,
        )
        return SearchProviderResult(
            response=response,
            billing=_require_provider_billing(
                data.billing,
                provider="desearch",
                billable_units=len(response.data),
            ),
        )

    async def search_links_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        data = await self._search_links_web_page(request, start=None)
        if data is None:
            raise ToolProviderError("tool provider failed")
        return SearchWebSearchResponse.model_validate(data)

    async def _search_links_web_page(
        self,
        request: SearchWebSearchRequest,
        *,
        start: int | None,
    ) -> dict[str, Any] | None:
        result = await self._search_links_web_page_with_billing(request, start=start)
        return result.data if result is not None else None

    async def _search_links_web_page_with_billing(
        self,
        request: SearchWebSearchRequest,
        *,
        start: int | None,
    ) -> _DeSearchRequestData | None:
        params: dict[str, Any] = {
            "query": _build_web_query(request.search_queries),
        }
        if request.num is not None:
            params["num"] = request.num
        if start is not None:
            params["start"] = start
        return await self._get_with_billing("web", params, requested_timeout=request.timeout)

    async def search_links_twitter(
        self,
        request: SearchXSearchRequest,
    ) -> SearchXSearchResponse:
        params = request.to_query_params()
        data = await self._get("twitter", params)
        if data is None:
            raise ToolProviderError("tool provider failed")
        response = SearchXSearchResponse.model_validate(data)
        self._log_search_links_twitter_summary(request, response)
        return response

    async def iter_search_links_twitter_pages(
        self,
        request: SearchXSearchRequest,
    ) -> AsyncIterator[DeSearchTwitterSearchPage]:
        base_query = request.query.strip()
        if not base_query:
            raise ValueError("desearch iter_search_links_twitter_pages requires non-empty query")
        if "max_id:" in base_query:
            raise ValueError("desearch iter_search_links_twitter_pages request.query must not contain max_id:")

        cursor: int | None = None
        page_index = 0
        while True:
            page_query = base_query if cursor is None else f"{base_query} max_id:{cursor}"
            page_request = request.model_copy(update={"query": page_query})

            response = await self.search_links_twitter(page_request)
            page_index += 1
            if not response.data:
                return

            page_posts = response.data
            post_ids = _parse_tweet_ids(page_posts)

            if cursor is not None:
                page_posts, post_ids, ignored_count = _filter_tweets_for_max_id(page_posts, cursor=cursor)
                if ignored_count:
                    _LOGGER.info(
                        "desearch.search_links_twitter.pagination.ignored_out_of_range",
                        extra={
                            "data": {
                                "page": page_index,
                                "cursor": cursor,
                                "ignored_count": ignored_count,
                            }
                        },
                    )
                if not page_posts:
                    _LOGGER.info(
                        "desearch.search_links_twitter.pagination.stopped_max_id_ignored",
                        extra={"data": {"page": page_index, "cursor": cursor}},
                    )
                    return

            yield DeSearchTwitterSearchPage(
                request=page_request,
                response=response,
                page=page_index,
                cursor=cursor,
                posts=page_posts,
                post_ids=post_ids,
            )

            if not post_ids:
                return
            cursor = min(post_ids) - 1

    async def iter_search_links_web_pages(
        self,
        request: SearchWebSearchRequest,
    ) -> AsyncIterator[DeSearchWebSearchPage]:
        _build_web_query(request.search_queries)
        start = 0
        page_index = 0
        last_first_link: str | None = None
        while True:
            response_data = await self._search_links_web_page(request, start=start)
            if response_data is None:
                raise ToolProviderError("tool provider failed")
            response = SearchWebSearchResponse.model_validate(response_data)
            page_index += 1
            if not response.data:
                return

            first_link = response.data[0].link if response.data else None
            if last_first_link is not None and first_link == last_first_link:
                _LOGGER.info(
                    "desearch.search_links_web.pagination.stopped_start_ignored",
                    extra={"data": {"page": page_index, "start": start}},
                )
                return
            last_first_link = first_link

            yield DeSearchWebSearchPage(
                request=request,
                response=response,
                page=page_index,
                start=start,
                results=response.data,
            )
            start += len(response.data)

    @staticmethod
    def _log_search_links_twitter_summary(
        request: SearchXSearchRequest,
        response: SearchXSearchResponse,
    ) -> None:
        created_at_by_id: dict[int, str | None] = {}
        min_id: int | None = None
        max_id: int | None = None
        for post in response.data:
            if post.id is None:
                continue
            tweet_id = int(post.id)
            created_at_by_id[tweet_id] = post.created_at
            if min_id is None or tweet_id < min_id:
                min_id = tweet_id
            if max_id is None or tweet_id > max_id:
                max_id = tweet_id

        sample_tweets = [tweet.model_dump(exclude_none=True) for tweet in response.data[:20]]

        _LOGGER.info(
            "desearch.search_links_twitter.summary",
            extra={
                "data": {
                    "request": request.model_dump(exclude_none=True),
                    "return_count": len(response.data),
                    "min_id": min_id,
                    "max_id": max_id,
                    "min_created_at": created_at_by_id.get(min_id) if min_id is not None else None,
                    "max_created_at": created_at_by_id.get(max_id) if max_id is not None else None,
                },
                "json_fields": {"sample_tweets": sample_tweets},
            },
        )

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

    async def ai_search(
        self,
        *,
        prompt: str,
        tools: tuple[DeSearchAiTool, ...],
        model: DeSearchAiModel,
        count: int,
        date_filter: DeSearchAiDateFilter | None,
        result_type: DeSearchAiResultType,
        system_message: str,
    ) -> object:
        result = await self.ai_search_with_billing(
            prompt=prompt,
            tools=tools,
            model=model,
            count=count,
            date_filter=date_filter,
            result_type=result_type,
            system_message=system_message,
        )
        return result.data

    async def ai_search_with_billing(
        self,
        *,
        prompt: str,
        tools: tuple[DeSearchAiTool, ...],
        model: DeSearchAiModel,
        count: int,
        date_filter: DeSearchAiDateFilter | None,
        result_type: DeSearchAiResultType,
        system_message: str,
        requested_timeout: float | None = None,
    ) -> _DeSearchRequestData:
        if not prompt.strip():
            raise ValueError("desearch ai_search requires non-empty prompt")
        if count <= 0:
            raise ValueError("desearch ai_search requires count > 0")

        payload_items: dict[str, object | None] = {
            "prompt": prompt,
            "tools": [tool.value for tool in tools],
            # NOTE: desearch.ai /desearch/ai/search appears to ignore or reject `model` currently.
            # We keep this parameter in the public interface for future compatibility,
            # but do not send it in the payload until the API stabilizes.
            # "model": model.value,
            "date_filter": date_filter.value if date_filter is not None else None,
            "result_type": result_type.value,
            "system_message": system_message,
            "streaming": False,
            "count": max(_DESEARCH_AI_MIN_COUNT, min(_DESEARCH_AI_MAX_COUNT, count)),
        }
        payload = {key: value for key, value in payload_items.items() if value is not None}
        data = await self._post_with_billing(
            "desearch/ai/search",
            payload,
            expect_data=False,
            requested_timeout=requested_timeout,
        )
        if data is None:
            raise ToolProviderError("tool provider failed")
        return data

    async def ai_search_twitter_posts(
        self,
        *,
        prompt: str,
        count: int,
        date_filter: DeSearchAiDateFilter | None,
        system_message: str = "",
    ) -> DeSearchAiDocsResponse:
        raw = await self.ai_search(
            prompt=prompt,
            tools=(DeSearchAiTool.TWITTER,),
            model=DeSearchAiModel.HORIZON,
            count=count,
            date_filter=date_filter,
            result_type=DeSearchAiResultType.LINKS_WITH_FINAL_SUMMARY,
            system_message=system_message,
        )
        return parse_desearch_ai_response(raw)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # internal

    async def _request_with_billing(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        expect_data: bool = True,
        allow_not_found: bool = False,
        response_format: str = "json",
        requested_timeout: float | None = None,
    ) -> _DeSearchRequestData | None:
        provider_timeout = platform_tool_proxy_effective_provider_timeout_seconds(
            self._timeout,
            requested_timeout,
        )
        if self._semaphore is None:
            return await self._request_with_retries(
                method,
                path,
                json_payload=json_payload,
                params=params,
                expect_data=expect_data,
                allow_not_found=allow_not_found,
                wait_ms=0.0,
                response_format=response_format,
                timeout=provider_timeout,
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
                response_format=response_format,
                timeout=provider_timeout,
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
        response_format: str,
        timeout: float,
    ) -> _DeSearchRequestData | None:
        reasons: list[str] = []
        total_latency_ms = 0.0
        attempts_made = 0
        tracer = trace.get_tracer("harnyx_commons.tools.desearch")
        with tracer.start_as_current_span(
            "desearch.request",
            kind=SpanKind.CLIENT,
            attributes={
                "http.method": method.upper(),
                "http.target": path,
            },
            record_exception=self._include_payloads_in_logs,
            set_status_on_exception=self._include_payloads_in_logs,
        ) as span:
            try:
                for attempt in range(self._retry_policy.attempts):
                    attempts_made = attempt + 1
                    attempt_start = time.perf_counter()
                    try:
                        resp = await self._send(
                            method,
                            path,
                            json_payload=json_payload,
                            params=params,
                            timeout=timeout,
                        )
                        raw_response, data, billing = await self._parse_response(
                            resp,
                            response_format=response_format,
                        )
                        total_latency_ms += (time.perf_counter() - attempt_start) * 1000
                        if expect_data:
                            if self._missing_data(data):
                                if self._should_retry(attempt):
                                    reasons.append("missing_data")
                                    await self._sleep(attempt)
                                    continue
                                span.set_attributes({"desearch.error": "missing_data"})
                                raise ToolProviderError("tool provider failed")
                            if self._empty_payload(data):
                                if self._should_retry(attempt):
                                    reasons.append("empty_data")
                                    await self._sleep(attempt)
                                    continue

                        data["attempts"] = attempt + 1
                        data["retry_reasons"] = tuple(reasons)
                        span.set_attributes(
                            {
                                "http.status_code": resp.status_code,
                                "desearch.result_count": (
                                    len(data["data"]) if isinstance(data.get("data"), list) else 0
                                ),
                            }
                        )

                        log_extra: dict[str, object] = {
                            "data": {
                                "method": method.upper(),
                                "path": path,
                                "status_code": resp.status_code,
                                "attempts": attempt + 1,
                                "latency_ms_total": round(total_latency_ms, 2),
                                "retry_reasons": tuple(reasons),
                                "result_count": len(data["data"]) if isinstance(data.get("data"), list) else None,
                                "wait_ms": round(wait_ms, 2),
                            }
                        }
                        if self._include_payloads_in_logs:
                            log_extra["json_fields"] = {
                                "request": {
                                    "method": method.upper(),
                                    "path": path,
                                    "params": params,
                                    "json": json_payload,
                                },
                                "response_raw": raw_response,
                            }
                        _LOGGER.info("desearch.request.complete", extra=log_extra)
                        return _DeSearchRequestData(data=data, billing=billing)
                    except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
                        status = exc.response.status_code if exc.response else None
                        if allow_not_found and status == 404:
                            total_latency_ms += (time.perf_counter() - attempt_start) * 1000
                            span.set_attributes(
                                {
                                    "http.status_code": 404,
                                    "desearch.not_found": True,
                                }
                            )
                            log_extra = {
                                "data": {
                                    "method": method.upper(),
                                    "path": path,
                                    "status_code": status,
                                    "attempts": attempt + 1,
                                    "latency_ms_total": round(total_latency_ms, 2),
                                    "retry_reasons": tuple(reasons),
                                    "wait_ms": round(wait_ms, 2),
                                }
                            }
                            if self._include_payloads_in_logs:
                                log_extra["json_fields"] = {
                                    "request": {
                                        "method": method.upper(),
                                        "path": path,
                                        "params": params,
                                        "json": json_payload,
                                    }
                                }
                            _LOGGER.info("desearch.request.not_found", extra=log_extra)
                            return None
                        retryable = status == 429 or (status is not None and status >= 500)
                        reasons.append(f"http_{status}")
                        if not (retryable and self._should_retry(attempt)):
                            span.set_attributes(
                                {
                                    "http.status_code": status or 0,
                                    "desearch.error": f"http_{status}",
                                }
                            )
                            raise ToolProviderError(
                                "tool provider failed",
                                failure_code=(
                                    ToolProviderFailureCode.AUTHENTICATION_FAILED
                                    if status == 403
                                    else ToolProviderFailureCode.PROVIDER_FAILED
                                ),
                                provider="desearch",
                                http_status=status,
                            ) from exc
                        await self._sleep(attempt)
                    except httpx.HTTPError as exc:  # pragma: no cover
                        reasons.append(exc.__class__.__name__)
                        if not self._should_retry(attempt):
                            span.set_attributes({"desearch.error": exc.__class__.__name__})
                            raise ToolProviderError("tool provider failed") from exc
                        await self._sleep(attempt)
                    except ValueError as exc:
                        span.set_attributes({"desearch.error": exc.__class__.__name__})
                        raise ToolProviderError("tool provider failed") from exc
            finally:
                final_attributes: dict[str, AttributeValue] = {
                    "desearch.attempts": attempts_made,
                    "desearch.latency_ms_total": round(total_latency_ms, 2),
                    "desearch.wait_ms": round(wait_ms, 2),
                }
                if reasons:
                    final_attributes["desearch.retry_reasons"] = tuple(reasons)
                span.set_attributes(final_attributes)

        raise ToolProviderError("tool provider failed")

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None,
        params: dict[str, Any] | None,
        timeout: float,
    ) -> httpx.Response:
        headers = {"Authorization": self._api_key}
        if method == "post":
            headers["content-type"] = "application/json"
            return await self._client.post(path, headers=headers, json=json_payload, timeout=timeout)
        return await self._client.get(path, headers=headers, params=params, timeout=timeout)

    @staticmethod
    async def _parse_response(
        resp: httpx.Response,
        *,
        response_format: str,
    ) -> tuple[object, dict[str, Any], ProviderBillingMetadata | None]:
        resp.raise_for_status()
        billing = _billing_metadata_from_headers(resp.headers)
        if response_format == "text":
            raw = resp.text
            return raw, {"content": raw}, billing
        raw = resp.json()
        body_billing = _billing_metadata_from_json(raw)
        if body_billing is not None:
            billing = body_billing
        data = raw
        if isinstance(data, list):
            data = {"data": data}
        if not isinstance(data, dict):
            raise RuntimeError("desearch response was not an object")
        return raw, data, billing

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


def _billing_metadata_from_headers(headers: httpx.Headers) -> ProviderBillingMetadata | None:
    cost = _optional_float(headers.get("x-desearch-cost-usd"))
    usage_count = _optional_int(headers.get("x-desearch-usage-count"))
    if cost is None and usage_count is None:
        return None
    return ProviderBillingMetadata(
        actual_cost_provider="desearch",
        actual_cost_usd=cost,
        usage_count=usage_count,
        service=_optional_text(headers.get("x-desearch-service")),
        currency=_optional_text(headers.get("x-desearch-currency")),
        source="response_headers",
    )


def _billing_metadata_from_json(raw: object) -> ProviderBillingMetadata | None:
    if not isinstance(raw, Mapping):
        return None
    raw_mapping = cast(Mapping[str, object], raw)
    cost = _optional_float(raw_mapping.get("cost_usd"))
    usage_count = _optional_int(raw_mapping.get("usage_count"))
    if cost is None and usage_count is None:
        return None
    return ProviderBillingMetadata(
        actual_cost_provider="desearch",
        actual_cost_usd=cost,
        usage_count=usage_count,
        service=_optional_text(raw_mapping.get("service")),
        currency=_optional_text(raw_mapping.get("currency")),
        source="response_body",
    )


def _require_provider_billing(
    billing: ProviderBillingMetadata | None,
    *,
    provider: str,
    billable_units: int,
) -> ProviderBillingMetadata:
    if billing is None:
        return ProviderBillingMetadata(
            actual_cost_provider=provider,
            actual_cost_usd=None,
            billable_units=billable_units,
            source="missing_provider_metadata",
        )
    if billing.billable_units is not None:
        return billing
    return replace(billing, billable_units=billable_units)


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        raise ValueError("cost metadata must be finite")
    if numeric < 0:
        raise ValueError("cost metadata must be non-negative")
    return numeric


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    return numeric


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_tweet_ids(posts: list[SearchXResult]) -> list[int]:
    ids: list[int] = []
    for post in posts:
        if post.id is None or not post.id.isdigit():
            continue
        ids.append(int(post.id))
    return ids


def _filter_tweets_for_max_id(
    posts: list[SearchXResult],
    *,
    cursor: int,
) -> tuple[list[SearchXResult], list[int], int]:
    filtered: list[SearchXResult] = []
    filtered_ids: list[int] = []
    ignored_count = 0
    for post in posts:
        if post.id is None or not post.id.isdigit():
            ignored_count += 1
            continue
        tweet_id = int(post.id)
        if tweet_id > cursor:
            ignored_count += 1
            continue
        filtered.append(post)
        filtered_ids.append(tweet_id)
    return filtered, filtered_ids, ignored_count


def _build_web_query(search_queries: tuple[str, ...]) -> str:
    terms = tuple(item.strip() for item in search_queries)
    if not terms or any(not term for term in terms):
        raise ValueError("desearch web search requires non-empty search_queries")
    if len(terms) == 1:
        return terms[0]
    return " OR ".join(f"({term})" for term in terms)


def _extract_desearch_ai_results(raw: object) -> list[SearchAiResult]:
    docs = parse_desearch_ai_response(raw)
    results: list[SearchAiResult] = []
    seen_urls: set[str] = set()

    def add(url: str, *, title: str | None = None, note: str | None = None) -> None:
        normalized = url.strip()
        if not normalized or normalized in seen_urls:
            return
        seen_urls.add(normalized)
        results.append(
            SearchAiResult(
                url=normalized,
                title=title or None,
                note=note or None,
            )
        )

    for item in docs.results or ():
        add(item.link, title=item.title, note=item.snippet)

    return results


__all__ = ["DeSearchClient"]
