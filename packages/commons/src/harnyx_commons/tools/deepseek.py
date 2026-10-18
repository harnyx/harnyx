"""DeepSeek's standalone Search and Fetch JSON APIs."""

from __future__ import annotations

import re

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator

from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.search_http import HttpErrorDecision, JsonSearchProviderClient
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    FetchPageResult,
    SearchWebResult,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)
from harnyx_miner_sdk.tools.search_provider_extra import DeepSeekSearchExtra


class _SearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rerank_score: float = Field(ge=0, le=3, strict=True, allow_inf_nan=False)
    url: str = Field(min_length=1)
    title: str
    snippet: str
    summary: str
    content: str
    site_name: str | None
    site_icon: str | None
    date: AwareDatetime | None
    authority: int = Field(ge=-1, le=4, strict=True)

    @field_validator("date", mode="before")
    @classmethod
    def _validate_date_format(cls, value: object) -> object:
        if value is not None and (
            not isinstance(value, str)
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})", value
            ) is None
        ):
            raise ValueError("DeepSeek result date must be an RFC 3339 timestamp or null")
        return value


class _SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    results: list[_SearchResult]


class _FetchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    content: str


def _classify_error(response: httpx.Response, body: bytes) -> HttpErrorDecision:
    # Fetch 403 means a target security-policy denial, not a revoked API key.
    status = response.status_code
    return HttpErrorDecision(
        retryable=status in {408, 429} or status >= 500,
        error=ToolProviderError(
            "tool provider failed", provider="deepseek", http_status=status,
            billing=ProviderBillingMetadata(
                actual_cost_provider="deepseek", source="response_headers", actual_cost_usd=0.0, billable_units=0,
            ),
            failure_code=(
                ToolProviderFailureCode.AUTHENTICATION_FAILED
                if status == 401 else ToolProviderFailureCode.PROVIDER_FAILED
            ),
        ),
    )


class DeepSeekClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        retry_policy: RetryPolicy | None = None,
        max_concurrent: int | None = None,
        max_response_bytes: int | None = None,
        include_payloads_in_logs: bool = True,
    ) -> None:
        self._http = JsonSearchProviderClient(
            provider="deepseek", base_url=base_url, api_key=api_key, timeout=timeout,
            auth_header="authorization", auth_value_prefix="Bearer ", client=client,
            retry_policy=retry_policy, max_concurrent=max_concurrent,
            max_response_bytes=max_response_bytes, include_payloads_in_logs=include_payloads_in_logs,
            error_classifier=_classify_error,
            retry_uncertain_transport_errors=False,
        )

    async def search_web(self, request: SearchWebSearchRequest) -> SearchProviderResult[SearchWebSearchResponse]:
        if request.num == 0:
            return SearchProviderResult(response=SearchWebSearchResponse(data=[]), billing=_billing(0, "search"))
        extra = request.provider_extra or DeepSeekSearchExtra()
        if not isinstance(extra, DeepSeekSearchExtra):
            raise ValueError("DeepSeek search requires DeepSeekSearchExtra")
        query = request.search_queries[0] if len(request.search_queries) == 1 else " OR ".join(
            f"({term})" for term in request.search_queries
        )
        payload: dict[str, object] = {"query": query, **extra.to_provider_payload()}
        if request.num is not None:
            payload["count"] = request.num
        raw = await self._http.post_json("/v1/tools/search", payload, requested_timeout=request.timeout)
        try:
            parsed = _SearchResponse.model_validate(raw)
        except ValidationError as exc:
            raise ToolProviderError("tool provider response invalid", provider="deepseek") from exc
        return SearchProviderResult(
            response=SearchWebSearchResponse(data=[
                SearchWebResult(link=item.url, title=item.title, snippet=item.content or item.summary or item.snippet)
                for item in parsed.results
            ]),
            billing=_billing(int(bool(parsed.results)), "search"),
        )

    async def fetch_page(self, request: FetchPageRequest) -> SearchProviderResult[FetchPageResponse]:
        raw = await self._http.post_json("/v1/tools/fetch", {"url": request.url}, requested_timeout=request.timeout)
        try:
            parsed = _FetchResponse.model_validate(raw)
        except ValidationError as exc:
            raise ToolProviderError("tool provider response invalid", provider="deepseek") from exc
        content = parsed.content.strip()
        return SearchProviderResult(
            response=FetchPageResponse(data=[
                FetchPageResult(url=request.url, content=content, title=parsed.title or None),
            ] if content else []),
            billing=_billing(int(bool(content)), "fetch"),
        )

    async def aclose(self) -> None:
        await self._http.aclose()


def _billing(units: int, service: str) -> ProviderBillingMetadata:
    return ProviderBillingMetadata(
        actual_cost_provider="deepseek", source="response_results", billable_units=units, service=service,
    )
