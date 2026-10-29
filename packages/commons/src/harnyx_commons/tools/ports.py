"""Ports for external tool providers shared across services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from harnyx_commons.json_types import JsonObject
from harnyx_commons.tools.decision_models import DecisionQueryRequest, DecisionQueryResponse
from harnyx_commons.tools.embedding_models import EmbedTextRequest, EmbedTextResponse
from harnyx_commons.tools.extraction_models import ExtractPagesRequest, ExtractPagesResponse
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.rerank_models import RerankRequest, RerankResponse
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


class FetchPageProviderPort(Protocol):
    """Provider seam for fetching a known page."""

    async def fetch_page(
        self,
        request: FetchPageRequest,
    ) -> SearchProviderResult[FetchPageResponse]: ...

    async def aclose(self) -> None: ...


class WebSearchProviderPort(FetchPageProviderPort, Protocol):
    """Provider seam for web search providers that also fetch pages."""

    async def search_web(self, request: SearchWebSearchRequest) -> SearchProviderResult[SearchWebSearchResponse]: ...


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


@dataclass(frozen=True, slots=True)
class RerankProviderResult:
    response: RerankResponse
    billing: ProviderBillingMetadata


class RerankProviderPort(Protocol):
    async def rerank(self, request: RerankRequest) -> RerankProviderResult: ...

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
    "RerankProviderPort",
    "RerankProviderResult",
    "EmbeddingProviderPort",
    "EmbeddingProviderResult",
    "PageExtractionProviderPort",
    "WebSearchProviderPort",
    "FetchPageProviderPort",
]


@dataclass(frozen=True, slots=True)
class DecisionNativeResponse:
    """Private provider receipt retained by consumers that own grading evidence."""

    body: str
    status_code: int


@dataclass(frozen=True, slots=True)
class DecisionProviderResult:
    response: DecisionQueryResponse
    actual_cost_usd: float | None
    actual_cost_provider: str
    actual_cost_evidence: JsonObject
    reference_cost_usd: float | None = None
    native_response: DecisionNativeResponse | None = None

    @property
    def billing(self) -> ProviderBillingMetadata:
        return ProviderBillingMetadata(
            actual_cost_usd=self.actual_cost_usd,
            actual_cost_provider=self.actual_cost_provider,
            source="response_body",
            usage=self.response.usage,
            reference_cost_usd=self.reference_cost_usd,
            provider_usage=self.actual_cost_evidence,
        )


class DecisionProviderPort(Protocol):
    async def query(self, request: DecisionQueryRequest) -> DecisionProviderResult: ...
    async def aclose(self) -> None: ...
