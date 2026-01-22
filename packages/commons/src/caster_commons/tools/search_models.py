"""Provider-agnostic request/response models for search tools.

This module re-exports the miner SDK models so commons/validator/platform share
the exact same schema and typing.
"""

from __future__ import annotations

from caster_miner_sdk.tools.search_models import (
    SearchAiDateFilter,
    SearchAiResult,
    SearchAiResultType,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchAiTool,
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
    "SearchAiTool",
    "SearchAiDateFilter",
    "SearchAiResultType",
    "SearchAiSearchRequest",
    "SearchAiSearchResponse",
    "SearchAiResult",
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
