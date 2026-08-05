"""Direct Exa Search and Contents adapter for retrieval-only miner tools."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.llm.cost_settlement import normalized_provider_cost
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
from harnyx_miner_sdk.tools.search_provider_extra import ExaFetchExtra, ExaSearchExtra


class _ExaCost(BaseModel):
    model_config = ConfigDict(extra="ignore")
    total: float | int | str | None = None


class _ExaResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str = Field(min_length=1)
    title: str | None = None
    text: str | None = None


class _ExaResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    results: list[_ExaResult]
    request_id: str | None = Field(default=None, alias="requestId")
    cost_dollars: _ExaCost | None = Field(default=None, alias="costDollars")


class ExaClient:
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
            provider="exa",
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            auth_header="x-api-key",
            auth_value_prefix="",
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
        extra = request.provider_extra or ExaSearchExtra()
        if not isinstance(extra, ExaSearchExtra):
            raise ValueError("Exa search requires ExaSearchExtra")
        payload: dict[str, object] = {
            "query": _combined_query(request.search_queries),
            **extra.to_provider_payload(),
        }
        if request.num is not None:
            payload["numResults"] = min(request.num, 100)
        raw = await self._http.post_json("/search", payload, requested_timeout=request.timeout)
        parsed = _parse_response(raw)
        return SearchProviderResult(
            response=SearchWebSearchResponse(
                data=[SearchWebResult(link=item.url, title=item.title) for item in parsed.results]
            ),
            billing=_billing_from_response(parsed, service="search"),
        )

    async def fetch_page(
        self, request: FetchPageRequest
    ) -> SearchProviderResult[FetchPageResponse]:
        extra = request.provider_extra or ExaFetchExtra()
        if not isinstance(extra, ExaFetchExtra):
            raise ValueError("Exa fetch requires ExaFetchExtra")
        raw = await self._http.post_json(
            "/contents",
            {"urls": [request.url], **extra.to_provider_payload()},
            requested_timeout=request.timeout,
        )
        parsed = _parse_response(raw)
        if len(parsed.results) != 1 or not parsed.results[0].text or not parsed.results[0].text.strip():
            raise ToolProviderError(
                "tool provider response invalid",
                provider="exa",
                billing=_billing_from_response(parsed, service="contents"),
            )
        item = parsed.results[0]
        content = item.text
        if content is None:
            raise AssertionError("validated Exa content unexpectedly missing")
        return SearchProviderResult(
            response=FetchPageResponse(
                data=[FetchPageResult(url=item.url, title=item.title, content=content.strip())]
            ),
            billing=_billing_from_response(parsed, service="contents"),
        )

    async def aclose(self) -> None:
        await self._http.aclose()


def _parse_response(raw: dict[str, Any]) -> _ExaResponse:
    try:
        return _ExaResponse.model_validate(raw)
    except ValidationError as exc:
        raise ToolProviderError("tool provider response invalid", provider="exa") from exc


def _combined_query(queries: tuple[str, ...]) -> str:
    return " OR ".join(f"({query})" for query in queries)


def _billing_from_response(response: _ExaResponse, *, service: str) -> ProviderBillingMetadata:
    raw_cost = response.cost_dollars.total if response.cost_dollars is not None else None
    return _billing(
        service=service,
        request_id=response.request_id,
        cost=normalized_provider_cost(raw_cost, field_name="Exa costDollars.total", strict=False),
    )


def _billing(
    *, service: str, request_id: str | None = None, cost: float | None = None
) -> ProviderBillingMetadata:
    return ProviderBillingMetadata(
        actual_cost_provider="exa",
        actual_cost_usd=cost,
        source="response_body" if request_id is not None or cost is not None else "missing_provider_metadata",
        provider_request_id=request_id,
        service=service,
        currency="USD" if cost is not None else None,
    )


__all__ = ["ExaClient"]
