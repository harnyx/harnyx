"""Strict provider-specific controls for miner web search and page fetch calls."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class _SearchExtra(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    def to_provider_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, mode="json", by_alias=True)


class DeSearchSearchExtra(_SearchExtra):
    start: int | None = Field(default=None, ge=0)


class DeSearchFetchExtra(_SearchExtra):
    format: Literal["text", "html"] = "text"
    js: bool | None = None
    wait: int | None = Field(default=None, ge=0)


class ParallelSourcePolicy(_SearchExtra):
    include_domains: tuple[str, ...] | None = None
    exclude_domains: tuple[str, ...] | None = None
    after_date: date | None = None

    @model_validator(mode="after")
    def _validate_domains(self) -> ParallelSourcePolicy:
        if self.include_domains and self.exclude_domains:
            raise ValueError("Parallel include_domains and exclude_domains cannot be combined")
        return self


class ParallelFetchPolicy(_SearchExtra):
    max_age_seconds: int | None = Field(default=None, ge=0)
    disable_cache_fallback: bool | None = None


class ParallelExcerptSettings(_SearchExtra):
    max_chars_per_result: int | None = Field(default=None, gt=0)


class ParallelSearchExtra(_SearchExtra):
    mode: Literal["turbo", "basic", "advanced"] | None = None
    max_chars_total: int | None = Field(default=None, gt=0)
    source_policy: ParallelSourcePolicy | None = None
    fetch_policy: ParallelFetchPolicy | None = None
    excerpt_settings: ParallelExcerptSettings | None = None
    location: str | None = Field(default=None, min_length=2, max_length=2)


class ParallelFetchExtra(_SearchExtra):
    objective: str | None = Field(default=None, min_length=1)
    max_chars_total: int | None = Field(default=None, gt=0)
    fetch_policy: ParallelFetchPolicy | None = None
    excerpt_settings: ParallelExcerptSettings | None = None
    full_content: bool | None = None


class FirecrawlCategory(_SearchExtra):
    type: Literal["github", "research", "pdf"]


class FirecrawlSearchExtra(_SearchExtra):
    categories: tuple[FirecrawlCategory | Literal["github", "research", "pdf"], ...] | None = None
    include_domains: tuple[str, ...] | None = Field(default=None, alias="includeDomains")
    exclude_domains: tuple[str, ...] | None = Field(default=None, alias="excludeDomains")
    tbs: str | None = Field(default=None, min_length=1)
    location: str | None = Field(default=None, min_length=1)
    country: str | None = Field(default=None, min_length=2, max_length=2)
    ignore_invalid_urls: bool | None = Field(default=None, alias="ignoreInvalidURLs")
    privacy_mode: Literal["anon"] | None = None

    @model_validator(mode="after")
    def _validate_domains(self) -> FirecrawlSearchExtra:
        if self.include_domains and self.exclude_domains:
            raise ValueError("Firecrawl include_domains and exclude_domains cannot be combined")
        return self

    def to_provider_payload(self) -> dict[str, Any]:
        payload = super().to_provider_payload()
        privacy_mode = payload.pop("privacy_mode", None)
        if privacy_mode is not None:
            payload["enterprise"] = [privacy_mode]
        return payload


class FirecrawlLocation(_SearchExtra):
    country: str | None = Field(default=None, min_length=2, max_length=2)
    languages: tuple[str, ...] | None = None


class FirecrawlFetchExtra(_SearchExtra):
    formats: tuple[Literal["markdown", "rawHtml"], ...] = Field(
        default=("markdown",),
        min_length=1,
    )
    only_main_content: bool | None = Field(default=None, alias="onlyMainContent")
    include_tags: tuple[str, ...] | None = Field(default=None, alias="includeTags")
    exclude_tags: tuple[str, ...] | None = Field(default=None, alias="excludeTags")
    max_age: int | None = Field(default=None, alias="maxAge", ge=0)
    min_age: int | None = Field(default=None, alias="minAge", ge=0)
    wait_for: int | None = Field(default=None, alias="waitFor", ge=0)
    mobile: bool | None = None
    parse_pdf: Literal[True] | None = Field(default=None, alias="parsers")
    location: FirecrawlLocation | None = None
    remove_base64_images: bool | None = Field(default=None, alias="removeBase64Images")
    block_ads: bool | None = Field(default=None, alias="blockAds")
    proxy: Literal["basic", "enhanced", "auto"] | None = None
    store_in_cache: bool | None = Field(default=None, alias="storeInCache")

    def to_provider_payload(self) -> dict[str, Any]:
        payload = super().to_provider_payload()
        parse_pdf = payload.pop("parsers", None)
        if parse_pdf is not None:
            payload["parsers"] = ["pdf"]
        return payload


class ExaSearchExtra(_SearchExtra):
    type: Literal["auto", "instant", "fast"] = "auto"
    category: Literal["company", "publication", "news", "personal site", "financial report", "people"] | None = None
    include_domains: tuple[str, ...] | None = Field(default=None, alias="includeDomains", max_length=1200)
    exclude_domains: tuple[str, ...] | None = Field(default=None, alias="excludeDomains", max_length=1200)
    start_published_date: datetime | None = Field(default=None, alias="startPublishedDate")
    end_published_date: datetime | None = Field(default=None, alias="endPublishedDate")
    user_location: str | None = Field(default=None, alias="userLocation", min_length=2, max_length=2)
    moderation: bool | None = None

    @model_validator(mode="after")
    def _validate_category_filters(self) -> ExaSearchExtra:
        if self.category in {"company", "people"} and (
            self.start_published_date is not None
            or self.end_published_date is not None
            or self.exclude_domains is not None
        ):
            raise ValueError("Exa company/people searches do not support published dates or exclude_domains")
        return self


class ExaTextOptions(_SearchExtra):
    max_characters: int | None = Field(default=None, alias="maxCharacters", gt=0)
    verbosity: Literal["compact", "standard", "full"] | None = None
    include_html_tags: bool | None = Field(default=None, alias="includeHtmlTags")


class ExaFetchExtra(_SearchExtra):
    text: ExaTextOptions | Literal[True] = True
    max_age_hours: int | None = Field(default=None, alias="maxAgeHours", ge=-1, le=720)
    livecrawl_timeout: int | None = Field(default=None, alias="livecrawlTimeout", gt=0, le=90_000)


class TavilySearchExtra(_SearchExtra):
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "fast"
    chunks_per_source: int | None = Field(default=None, ge=1, le=3)
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year", "d", "w", "m", "y"] | None = None
    start_date: date | None = None
    end_date: date | None = None
    include_domains: tuple[str, ...] | None = Field(default=None, max_length=300)
    exclude_domains: tuple[str, ...] | None = Field(default=None, max_length=150)
    country: str | None = Field(default=None, min_length=1)
    include_favicon: bool | None = None
    exact_match: bool | None = None
    safe_search: bool | None = None

    @model_validator(mode="after")
    def _validate_combinations(self) -> TavilySearchExtra:
        if self.country is not None and self.topic != "general":
            raise ValueError("Tavily country is only supported for the general topic")
        if self.safe_search is True and self.search_depth in {"fast", "ultra-fast"}:
            raise ValueError("Tavily safe_search cannot be combined with fast or ultra-fast search_depth")
        if self.chunks_per_source is not None and self.search_depth == "ultra-fast":
            raise ValueError("Tavily chunks_per_source cannot be combined with ultra-fast search_depth")
        return self


class TavilyFetchExtra(_SearchExtra):
    query: str | None = Field(default=None, min_length=1)
    chunks_per_source: int | None = Field(default=None, ge=1, le=5)
    extract_depth: Literal["basic", "advanced"] = "basic"
    format: Literal["text", "markdown"] = "markdown"

    @model_validator(mode="after")
    def _validate_chunks(self) -> TavilyFetchExtra:
        if self.chunks_per_source is not None and self.query is None:
            raise ValueError("Tavily fetch chunks_per_source requires query")
        return self


SearchWebProviderExtra: TypeAlias = (
    DeSearchSearchExtra | ParallelSearchExtra | FirecrawlSearchExtra | ExaSearchExtra | TavilySearchExtra
)
FetchPageProviderExtra: TypeAlias = (
    DeSearchFetchExtra | ParallelFetchExtra | FirecrawlFetchExtra | ExaFetchExtra | TavilyFetchExtra
)

_SEARCH_MODELS = {
    "desearch": TypeAdapter(DeSearchSearchExtra),
    "parallel": TypeAdapter(ParallelSearchExtra),
    "firecrawl": TypeAdapter(FirecrawlSearchExtra),
    "exa": TypeAdapter(ExaSearchExtra),
    "tavily": TypeAdapter(TavilySearchExtra),
}
_FETCH_MODELS = {
    "desearch": TypeAdapter(DeSearchFetchExtra),
    "parallel": TypeAdapter(ParallelFetchExtra),
    "firecrawl": TypeAdapter(FirecrawlFetchExtra),
    "exa": TypeAdapter(ExaFetchExtra),
    "tavily": TypeAdapter(TavilyFetchExtra),
}


def validate_search_provider_extra(
    *, operation: Literal["search_web", "fetch_page"], provider: str, provider_extra: object
) -> SearchWebProviderExtra | FetchPageProviderExtra | None:
    if provider_extra is None:
        return None
    models = _SEARCH_MODELS if operation == "search_web" else _FETCH_MODELS
    try:
        adapter = models[provider]
    except KeyError as exc:
        raise ValueError(f"provider_extra is not supported for provider {provider!r}") from exc
    return adapter.validate_python(provider_extra)


__all__ = [
    "DeSearchFetchExtra", "DeSearchSearchExtra", "ExaFetchExtra", "ExaSearchExtra",
    "FetchPageProviderExtra", "FirecrawlFetchExtra", "FirecrawlSearchExtra",
    "ParallelFetchExtra", "ParallelSearchExtra", "SearchWebProviderExtra",
    "TavilyFetchExtra", "TavilySearchExtra", "validate_search_provider_extra",
]
