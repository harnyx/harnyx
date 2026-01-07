"""Provider-agnostic request/response models for search tools.

This module re-exports the miner SDK models so commons/validator/platform share
the exact same schema and typing.
"""

from __future__ import annotations

from caster_miner_sdk.tools.search_models import (
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

