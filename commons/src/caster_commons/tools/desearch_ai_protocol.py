"""Typed boundary for DeSearch AI search responses.

This module exists to keep raw external response parsing (dict/JSON) isolated behind
Pydantic `TypeAdapter` validation, so downstream code can work with typed objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ConfigDict, TypeAdapter

from caster_commons.tools.search_models import SearchXResult


@dataclass(frozen=True, slots=True)
class DeSearchAiDocsLinkResult:
    link: str
    title: str | None = None
    snippet: str | None = None

    __pydantic_config__ = ConfigDict(extra="ignore")


@dataclass(frozen=True, slots=True)
class DeSearchAiDocsResponse:
    search: list[DeSearchAiDocsLinkResult] | None = None
    wikipedia_search: list[DeSearchAiDocsLinkResult] | None = None
    youtube_search: list[DeSearchAiDocsLinkResult] | None = None
    arxiv_search: list[DeSearchAiDocsLinkResult] | None = None
    reddit_search: list[DeSearchAiDocsLinkResult] | None = None
    hacker_news_search: list[DeSearchAiDocsLinkResult] | None = None

    tweets: list[SearchXResult] | None = None
    miner_link_scores: dict[str, str] | None = None
    completion: str | None = None
    text: str | None = None

    __pydantic_config__ = ConfigDict(extra="ignore")


DESEARCH_AI_DOCS_RESPONSE_ADAPTER = TypeAdapter(DeSearchAiDocsResponse)


def parse_desearch_ai_response(raw: object) -> DeSearchAiDocsResponse:
    if not isinstance(raw, dict):
        raise TypeError("desearch ai search response must be a JSON object")
    if not _looks_like_docs_response(raw):
        raise ValueError("unexpected desearch ai search response shape (missing docs keys like 'tweets' or '*_search')")
    return DESEARCH_AI_DOCS_RESPONSE_ADAPTER.validate_python(raw)


def _looks_like_docs_response(raw: dict[object, object]) -> bool:
    for key in (
        "tweets",
        "search",
        "wikipedia_search",
        "youtube_search",
        "arxiv_search",
        "reddit_search",
        "hacker_news_search",
        "miner_link_scores",
        "completion",
        "text",
    ):
        if key in raw:
            return True
    return False


__all__ = [
    "DESEARCH_AI_DOCS_RESPONSE_ADAPTER",
    "DeSearchAiDocsLinkResult",
    "DeSearchAiDocsResponse",
    "parse_desearch_ai_response",
]
