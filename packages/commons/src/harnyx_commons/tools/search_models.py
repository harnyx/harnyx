"""Provider-agnostic request/response models for search tools.

This module re-exports the miner SDK models so commons/validator/platform share
the exact same schema and typing.
"""

from __future__ import annotations

from harnyx_miner_sdk.tools.search_models import (
    AiSearchProviderName,
    FetchPageRequest,
    FetchPageResponse,
    FetchPageResult,
    SearchAiDateFilter,
    SearchAiResult,
    SearchAiResultType,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchAiTool,
    SearchProviderName,
    SearchWebResult,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
    SearchXExtendedEntities,
    SearchXMediaEntity,
    SearchXResult,
    SearchXSearchRequest,
    SearchXSearchResponse,
    SearchXUser,
)

__all__ = [
    "AiSearchProviderName",
    "SearchAiTool",
    "SearchProviderName",
    "SearchAiDateFilter",
    "SearchAiResultType",
    "SearchAiSearchRequest",
    "SearchAiSearchResponse",
    "SearchAiResult",
    "FetchPageRequest",
    "FetchPageResponse",
    "FetchPageResult",
    "SearchWebSearchRequest",
    "SearchWebSearchResponse",
    "SearchWebResult",
    "SearchXSearchRequest",
    "SearchXSearchResponse",
    "SearchXResult",
    "SearchXMediaEntity",
    "SearchXExtendedEntities",
    "SearchXUser",
]
