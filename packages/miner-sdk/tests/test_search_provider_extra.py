from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.tools.search_models import FetchPageRequest, SearchWebSearchRequest
from harnyx_miner_sdk.tools.search_provider_extra import (
    ExaSearchExtra,
    FirecrawlFetchExtra,
    TavilyFetchExtra,
)


def test_exa_search_extra_is_selected_by_provider() -> None:
    request = SearchWebSearchRequest.model_validate(
        {
            "provider": "exa",
            "search_queries": ["harnyx"],
            "provider_extra": {"type": "fast", "include_domains": ["harnyx.ai"]},
        }
    )

    assert isinstance(request.provider_extra, ExaSearchExtra)
    assert request.provider_extra.to_provider_payload() == {
        "type": "fast",
        "includeDomains": ["harnyx.ai"],
    }


def test_tavily_fetch_extra_requires_query_for_chunks() -> None:
    with pytest.raises(ValidationError, match="requires query"):
        FetchPageRequest.model_validate(
            {
                "provider": "tavily",
                "url": "https://example.com",
                "provider_extra": {"chunks_per_source": 2},
            }
        )

    request = FetchPageRequest.model_validate(
        {
            "provider": "tavily",
            "url": "https://example.com",
            "provider_extra": {"query": "relevant section", "chunks_per_source": 2},
        }
    )
    assert isinstance(request.provider_extra, TavilyFetchExtra)


@pytest.mark.parametrize(
    ("provider", "provider_extra", "error"),
    [
        ("exa", {"type": "deep"}, "auto.*instant.*fast"),
        ("exa", {"output_schema": {"type": "string"}}, "[Ee]xtra inputs"),
        ("tavily", {"include_answer": True}, "[Ee]xtra inputs"),
        ("tavily", {"search_depth": "fast", "safe_search": True}, "cannot be combined"),
        ("tavily", {"search_depth": "ultra-fast", "safe_search": True}, "cannot be combined"),
        ("firecrawl", {"scrape_options": {"formats": ["summary"]}}, "[Ee]xtra inputs"),
        ("parallel", {"session_id": "cross-call-state"}, "[Ee]xtra inputs"),
    ],
)
def test_search_extra_rejects_generated_output_state_and_unsupported_controls(
    provider: str, provider_extra: object, error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        SearchWebSearchRequest.model_validate(
            {
                "provider": provider,
                "search_queries": ["harnyx"],
                "provider_extra": provider_extra,
            }
        )


def test_exa_company_filters_reject_documented_incompatible_fields() -> None:
    with pytest.raises(ValidationError, match="do not support"):
        SearchWebSearchRequest.model_validate(
            {
                "provider": "exa",
                "search_queries": ["harnyx"],
                "provider_extra": {
                    "category": "company",
                    "start_published_date": "2026-01-01T00:00:00Z",
                },
            }
        )


def test_exa_defaults_to_non_deep_auto_search() -> None:
    request = SearchWebSearchRequest.model_validate(
        {
            "provider": "exa",
            "search_queries": ["harnyx"],
        }
    )

    assert request.provider_extra is None
    assert ExaSearchExtra().to_provider_payload()["type"] == "auto"


@pytest.mark.parametrize("search_depth", ("basic", "fast", "advanced", "ultra-fast"))
def test_tavily_accepts_ordinary_search_depths(search_depth: str) -> None:
    request = SearchWebSearchRequest.model_validate(
        {
            "provider": "tavily",
            "search_queries": ["harnyx"],
            "provider_extra": {"search_depth": search_depth},
        }
    )

    assert request.provider_extra is not None
    assert request.provider_extra.to_provider_payload()["search_depth"] == search_depth


def test_tavily_ultra_fast_rejects_chunk_controls_it_cannot_apply() -> None:
    with pytest.raises(ValidationError, match="chunks_per_source cannot be combined"):
        SearchWebSearchRequest.model_validate(
            {
                "provider": "tavily",
                "search_queries": ["harnyx"],
                "provider_extra": {"search_depth": "ultra-fast", "chunks_per_source": 2},
            }
        )


@pytest.mark.parametrize("time_range", ("day", "week", "month", "year", "d", "w", "m", "y"))
def test_tavily_accepts_documented_time_ranges(time_range: str) -> None:
    request = SearchWebSearchRequest.model_validate(
        {
            "provider": "tavily",
            "search_queries": ["harnyx"],
            "provider_extra": {"time_range": time_range},
        }
    )

    assert request.provider_extra is not None
    assert request.provider_extra.to_provider_payload()["time_range"] == time_range


def test_parallel_accepts_turbo_search_mode() -> None:
    request = SearchWebSearchRequest.model_validate(
        {
            "provider": "parallel",
            "search_queries": ["harnyx"],
            "provider_extra": {"mode": "turbo"},
        }
    )

    assert request.provider_extra is not None
    assert request.provider_extra.to_provider_payload()["mode"] == "turbo"


def test_exa_fetch_rejects_disabling_text_content() -> None:
    with pytest.raises(ValidationError, match="Input should be True"):
        FetchPageRequest.model_validate(
            {
                "provider": "exa",
                "url": "https://example.com",
                "provider_extra": {"text": False},
            }
        )


def test_firecrawl_fetch_rejects_disabling_pdf_parsing() -> None:
    with pytest.raises(ValidationError, match="Input should be True"):
        FetchPageRequest.model_validate(
            {
                "provider": "firecrawl",
                "url": "https://example.com/document.pdf",
                "provider_extra": {"parse_pdf": False},
            }
        )


def test_firecrawl_fetch_accepts_documented_enhanced_proxy() -> None:
    request = FetchPageRequest.model_validate(
        {
            "provider": "firecrawl",
            "url": "https://example.com",
            "provider_extra": {"proxy": "enhanced"},
        }
    )

    assert request.provider_extra is not None
    assert request.provider_extra.to_provider_payload()["proxy"] == "enhanced"


@pytest.mark.parametrize(
    "formats",
    [
        ["rawHtml"],
        ["markdown", "rawHtml"],
        ["rawHtml", "markdown"],
    ],
)
def test_firecrawl_fetch_preserves_provider_formats(formats: list[str]) -> None:
    request = FetchPageRequest.model_validate(
        {
            "provider": "firecrawl",
            "url": "https://example.com",
            "provider_extra": {"formats": formats},
        }
    )

    assert isinstance(request.provider_extra, FirecrawlFetchExtra)
    assert request.provider_extra.to_provider_payload()["formats"] == formats


@pytest.mark.parametrize("formats", [[], ["html"], ["markdown", "html"]])
def test_firecrawl_fetch_rejects_empty_or_unsupported_formats(formats: list[str]) -> None:
    with pytest.raises(ValidationError):
        FetchPageRequest.model_validate(
            {
                "provider": "firecrawl",
                "url": "https://example.com",
                "provider_extra": {"formats": formats},
            }
        )
