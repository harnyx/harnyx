"""Provider-agnostic request/response models for miner-facing web tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from harnyx_miner_sdk.tools.types import ToolInvocationTimeout


class SearchWebSearchRequest(BaseModel):
    """Query parameters for the `search_web` tool."""

    model_config = ConfigDict(extra="forbid")

    search_queries: tuple[str, ...] = Field(min_length=1)
    num: int | None = Field(default=None, ge=0)
    timeout: ToolInvocationTimeout | None = None

    @field_validator("search_queries", mode="before")
    @classmethod
    def _normalize_search_queries(cls, value: object) -> object:
        if isinstance(value, str):
            return (value,)
        return value

    @field_validator("search_queries")
    @classmethod
    def _validate_search_queries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if not normalized or any(not item for item in normalized):
            raise ValueError("search_queries must contain non-empty keywords")
        return normalized

    def to_query_params(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude={"timeout"})


class SearchWebResult(BaseModel):
    """Single web search result item."""

    link: str
    snippet: str | None = None
    title: str | None = None


class SearchWebSearchResponse(BaseModel):
    """Response payload for the `search_web` tool."""

    data: list[SearchWebResult] = Field(default_factory=list)
    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None


class SearchXSearchRequest(BaseModel):
    """Query parameters for the `search_x` tool."""

    model_config = ConfigDict(extra="forbid")

    query: str
    count: int | None = None
    lang: str | None = None
    sort: Literal["Top", "Latest"] | None = None
    user: str | None = None
    start_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    verified: bool | None = None
    blue_verified: bool | None = None
    is_quote: bool | None = None
    is_video: bool | None = None
    is_image: bool | None = None
    min_retweets: int | None = None
    min_replies: int | None = None
    min_likes: int | None = None

    def to_query_params(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class SearchXUser(BaseModel):
    """Author details for an X result."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    username: str | None = None
    id: str | None = None
    display_name: str | None = Field(default=None, alias="name")
    profile_image_url: str | None = Field(
        default=None, validation_alias=AliasChoices("profile_image_url_https", "profile_image_url")
    )
    followers_count: int | None = None
    verified: bool | None = None
    is_blue_verified: bool | None = None
    url: str | None = None


class SearchXMediaEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    media_url_https: str | None = None
    media_url: str | None = None
    expanded_url: str | None = None


class SearchXExtendedEntities(BaseModel):
    model_config = ConfigDict(extra="ignore")

    media: list[SearchXMediaEntity] = Field(default_factory=list)


class SearchXResult(BaseModel):
    """Single X (Twitter) search result item."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    url: str | None = None
    text: str
    user: SearchXUser
    created_at: str | None = None
    lang: str | None = None
    like_count: int | None = None
    retweet_count: int | None = None
    reply_count: int | None = None
    quote_count: int | None = None
    view_count: int | None = None
    bookmark_count: int | None = None
    conversation_id: str | None = None
    in_reply_to_status_id: str | None = None
    quoted_status_id: str | None = None
    is_quote_tweet: bool | None = None
    media: list[SearchXMediaEntity] | None = None
    extended_entities: SearchXExtendedEntities | None = None


class SearchXSearchResponse(BaseModel):
    """Response payload for the `search_x` tool."""

    data: list[SearchXResult] = Field(default_factory=list)
    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None


SearchAiTool = Literal[
    "web",
    "hackernews",
    "reddit",
    "wikipedia",
    "youtube",
    "twitter",
    "arxiv",
]

SearchAiDateFilter = Literal[
    "PAST_24_HOURS",
    "PAST_2_DAYS",
    "PAST_WEEK",
    "PAST_2_WEEKS",
    "PAST_MONTH",
    "PAST_2_MONTHS",
    "PAST_YEAR",
    "PAST_2_YEARS",
]

SearchAiResultType = Literal[
    "ONLY_LINKS",
    "LINKS_WITH_FINAL_SUMMARY",
]


class SearchAiSearchRequest(BaseModel):
    """Query parameters for the `search_ai` tool."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1)
    count: int = Field(default=10, ge=10, le=200)
    timeout: ToolInvocationTimeout | None = None


class SearchAiResult(BaseModel):
    """Single AI search result item."""

    url: str = Field(min_length=1)
    note: str | None = None
    title: str | None = None


class SearchAiSearchResponse(BaseModel):
    """Response payload for the `search_ai` tool."""

    data: list[SearchAiResult] = Field(default_factory=list)
    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None


class FetchPageRequest(BaseModel):
    """Query parameters for the `fetch_page` tool."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    timeout: ToolInvocationTimeout | None = None


class FetchPageResult(BaseModel):
    """Single fetched page item."""

    url: str = Field(min_length=1)
    content: str = Field(min_length=1)
    title: str | None = None


class FetchPageResponse(BaseModel):
    """Response payload for the `fetch_page` tool."""

    data: list[FetchPageResult] = Field(default_factory=list)
    attempts: int | None = None
    retry_reasons: tuple[str, ...] | None = None


__all__ = [
    "ToolInvocationTimeout",
    "SearchWebSearchRequest",
    "SearchWebSearchResponse",
    "SearchWebResult",
    "SearchXSearchRequest",
    "SearchXSearchResponse",
    "SearchXResult",
    "SearchXMediaEntity",
    "SearchXExtendedEntities",
    "SearchXUser",
    "SearchAiTool",
    "SearchAiDateFilter",
    "SearchAiResultType",
    "SearchAiSearchRequest",
    "SearchAiSearchResponse",
    "SearchAiResult",
    "FetchPageRequest",
    "FetchPageResponse",
    "FetchPageResult",
]
