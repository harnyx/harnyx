"""Direct Tavily Search and Extract adapter for retrieval-only miner tools."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.search_http import JsonSearchProviderClient
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    FetchPageResult,
    SearchWebResult,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)
from harnyx_miner_sdk.tools.search_provider_extra import TavilyFetchExtra, TavilySearchExtra


class _TavilyUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    credits: int | None = Field(default=None, ge=0)


class _TavilySearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str = Field(min_length=1)
    title: str | None = None
    content: str | None = None


class _TavilySearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    results: list[_TavilySearchResult]
    request_id: str | None = None
    usage: _TavilyUsage | None = None


class _TavilyExtractResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str = Field(min_length=1)
    raw_content: str = Field(min_length=1)


class _TavilyExtractResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    results: list[_TavilyExtractResult]
    request_id: str | None = None
    usage: _TavilyUsage | None = None


class TavilyClient:
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
        self._http = JsonSearchProviderClient(
            provider="tavily",
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            auth_header="authorization",
            auth_value_prefix="Bearer ",
            client=client,
            retry_policy=retry_policy,
            max_concurrent=max_concurrent,
            include_payloads_in_logs=include_payloads_in_logs,
        )

    async def search_web(
        self, request: SearchWebSearchRequest
    ) -> SearchProviderResult[SearchWebSearchResponse]:
        if request.num == 0:
            return SearchProviderResult(
                response=SearchWebSearchResponse(data=[]), billing=_billing(service="search")
            )
        extra = request.provider_extra or TavilySearchExtra()
        if not isinstance(extra, TavilySearchExtra):
            raise ValueError("Tavily search requires TavilySearchExtra")
        payload: dict[str, object] = {
            "query": _combined_query(request.search_queries),
            **extra.to_provider_payload(),
            "auto_parameters": False,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "include_usage": True,
        }
        if request.num is not None:
            payload["max_results"] = min(request.num, 20)
        raw = await self._http.post_json("/search", payload, requested_timeout=request.timeout)
        try:
            parsed = _TavilySearchResponse.model_validate(raw)
        except ValidationError as exc:
            raise ToolProviderError("tool provider response invalid", provider="tavily") from exc
        return SearchProviderResult(
            response=SearchWebSearchResponse(
                data=[
                    SearchWebResult(link=item.url, title=item.title, snippet=item.content)
                    for item in parsed.results
                ]
            ),
            billing=_billing(
                service="search",
                request_id=parsed.request_id,
                credit_count=parsed.usage.credits if parsed.usage is not None else None,
            ),
        )

    async def fetch_page(
        self, request: FetchPageRequest
    ) -> SearchProviderResult[FetchPageResponse]:
        extra = request.provider_extra or TavilyFetchExtra()
        if not isinstance(extra, TavilyFetchExtra):
            raise ValueError("Tavily fetch requires TavilyFetchExtra")
        raw = await self._http.post_json(
            "/extract",
            {
                "urls": [request.url],
                **extra.to_provider_payload(),
                "include_images": False,
                "include_favicon": False,
                "include_usage": True,
            },
            requested_timeout=request.timeout,
        )
        try:
            parsed = _TavilyExtractResponse.model_validate(raw)
        except ValidationError as exc:
            raise ToolProviderError("tool provider response invalid", provider="tavily") from exc
        if len(parsed.results) != 1 or not parsed.results[0].raw_content.strip():
            raise ToolProviderError("tool provider response invalid", provider="tavily")
        item = parsed.results[0]
        return SearchProviderResult(
            response=FetchPageResponse(
                data=[FetchPageResult(url=item.url, content=item.raw_content.strip())]
            ),
            billing=_billing(
                service="extract",
                request_id=parsed.request_id,
                credit_count=parsed.usage.credits if parsed.usage is not None else None,
            ),
        )

    async def aclose(self) -> None:
        await self._http.aclose()


def _combined_query(queries: tuple[str, ...]) -> str:
    return " OR ".join(f"({query})" for query in queries)


def _billing(
    *, service: str, request_id: str | None = None, credit_count: int | None = None
) -> ProviderBillingMetadata:
    return ProviderBillingMetadata(
        actual_cost_provider="tavily",
        source=(
            "response_body"
            if request_id is not None or credit_count is not None
            else "missing_provider_metadata"
        ),
        provider_request_id=request_id,
        usage_count=credit_count,
        service=service,
    )


__all__ = ["TavilyClient"]
