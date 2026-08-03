"""Ports for external tool providers shared across services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from harnyx_commons.json_types import JsonObject
from harnyx_commons.tools.embedding_models import EmbedTextRequest, EmbedTextResponse
from harnyx_commons.tools.extraction_models import ExtractPagesRequest, ExtractPagesResponse
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
    """Provider seam for miner-facing web search and page fetching."""

    async def search_web(
        self,
        request: SearchWebSearchRequest,
    ) -> SearchProviderResult[SearchWebSearchResponse]: ...

    async def fetch_page(
        self,
        request: FetchPageRequest,
    ) -> SearchProviderResult[FetchPageResponse]: ...

    async def aclose(self) -> None: ...


class AiSearchProviderPort(Protocol):
    """Provider seam for miner-facing AI search."""

    async def search_ai(
        self,
        request: SearchAiSearchRequest,
    ) -> SearchProviderResult[SearchAiSearchResponse]: ...

    async def aclose(self) -> None: ...


class PageExtractionProviderPort(Protocol):
    """Provider seam for fetching content from a bounded set of known URLs."""

    async def extract_pages(
        self,
        request: ExtractPagesRequest,
    ) -> SearchProviderResult[ExtractPagesResponse]: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EmbeddingProviderResult:
    response: EmbedTextResponse
    actual_cost_usd: float | None
    actual_cost_provider: str
    actual_cost_evidence: JsonObject


class EmbeddingProviderPort(Protocol):
    """Shared provider seam for miner-facing embedding tools."""

    async def embed_text(
        self,
        request: EmbedTextRequest,
    ) -> EmbeddingProviderResult: ...

    async def aclose(self) -> None: ...


class DeSearchPort(Protocol):
    """Internal DeSearch seam for X-specific helpers."""

    async def search_links_twitter(
        self,
        request: SearchXSearchRequest,
    ) -> SearchXSearchResponse: ...

    async def fetch_twitter_post(self, *, post_id: str) -> SearchXResult | None: ...

__all__ = [
    "AiSearchProviderPort",
    "DeSearchPort",
    "EmbeddingProviderPort",
    "EmbeddingProviderResult",
    "PageExtractionProviderPort",
    "WebSearchProviderPort",
]
