"""Provider-agnostic request/response models for search tools.

This module re-exports the miner SDK models so commons/validator/platform share
the exact same schema and typing.
"""

from __future__ import annotations

from harnyx_miner_sdk.tools.search_models import (
    GetRepoFileRequest,
    GetRepoFileResponse,
    GetRepoFileResult,
    SearchAiDateFilter,
    SearchAiResult,
    SearchAiResultType,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchAiTool,
    SearchRepoResult,
    SearchRepoSearchRequest,
    SearchRepoSearchResponse,
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
    "SearchRepoSearchRequest",
    "SearchRepoSearchResponse",
    "SearchRepoResult",
    "GetRepoFileRequest",
    "GetRepoFileResponse",
    "GetRepoFileResult",
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
