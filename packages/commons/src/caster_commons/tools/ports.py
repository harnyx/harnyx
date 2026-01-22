"""Ports for external tool providers shared across services."""

from __future__ import annotations

from typing import Protocol

from caster_commons.tools.desearch import (
    DeSearchAiDateFilter,
    DeSearchAiModel,
    DeSearchAiResultType,
    DeSearchAiTool,
)
from caster_commons.tools.search_models import (
    SearchWebSearchRequest,
    SearchWebSearchResponse,
    SearchXResult,
    SearchXSearchRequest,
    SearchXSearchResponse,
)


class DeSearchPort(Protocol):
    """Client abstraction for the DeSearch API."""

    async def search_links_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse: ...

    async def search_links_twitter(
        self,
        request: SearchXSearchRequest,
    ) -> SearchXSearchResponse: ...

    async def fetch_twitter_post(self, *, post_id: str) -> SearchXResult | None: ...

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
    ) -> object: ...


__all__ = ["DeSearchPort"]
