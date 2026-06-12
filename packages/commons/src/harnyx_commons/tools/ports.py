"""Ports for external tool providers shared across services."""

from __future__ import annotations

from typing import Protocol

from harnyx_commons.tools.provider_billing import SearchProviderResult
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
    SearchXResult,
    SearchXSearchRequest,
    SearchXSearchResponse,
)


class WebSearchProviderPort(Protocol):
    """Shared provider seam for miner-facing web tools."""

    async def search_web(
        self,
        request: SearchWebSearchRequest,
    ) -> SearchProviderResult[SearchWebSearchResponse]: ...

    async def search_ai(
        self,
        request: SearchAiSearchRequest,
    ) -> SearchProviderResult[SearchAiSearchResponse]: ...

    async def fetch_page(
        self,
        request: FetchPageRequest,
    ) -> SearchProviderResult[FetchPageResponse]: ...

    async def aclose(self) -> None: ...


class DeSearchPort(Protocol):
    """Internal DeSearch seam for X-specific helpers."""

    async def search_links_twitter(
        self,
        request: SearchXSearchRequest,
    ) -> SearchXSearchResponse: ...

    async def fetch_twitter_post(self, *, post_id: str) -> SearchXResult | None: ...

__all__ = ["DeSearchPort", "WebSearchProviderPort"]
