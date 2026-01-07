"""Ports for external tool providers shared across services."""

from __future__ import annotations

from typing import Protocol

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


__all__ = ["DeSearchPort"]
