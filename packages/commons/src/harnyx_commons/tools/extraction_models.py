"""Provider-independent contracts for extracting content from known web pages."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field, field_validator

from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG


class ExtractPagesRequest(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    urls: tuple[str, ...] = Field(min_length=1, max_length=20)
    objective: str | None = Field(default=None, min_length=1)
    max_chars_per_result: int | None = Field(default=None, gt=0)
    max_age_seconds: int | None = Field(default=None, ge=600)
    disable_cache_fallback: bool | None = None
    client_model: str | None = Field(default=None, min_length=1)

    @field_validator("urls", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("urls")
    @classmethod
    def _canonicalize_and_deduplicate_urls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(canonicalize_extraction_url(url) for url in value))


class ExtractedPage(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    url: str = Field(min_length=1)
    title: str | None = None
    publish_date: str | None = None
    excerpts: tuple[str, ...] = ()
    content: str | None = None

    @field_validator("excerpts", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class PageExtractionError(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    url: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    http_status_code: int | None = None
    content: str | None = None


class PageExtractionWarning(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    warning_type: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ExtractPagesResponse(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    pages: tuple[ExtractedPage, ...] = ()
    errors: tuple[PageExtractionError, ...] = ()
    warnings: tuple[PageExtractionWarning, ...] = ()
    session_id: str | None = None
    attempts: int | None = Field(default=None, ge=1)
    retry_reasons: tuple[str, ...] | None = None

    @field_validator("pages", "errors", "warnings", "retry_reasons", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


def canonicalize_extraction_url(url: str) -> str:
    """Return the canonical HTTP URL identity shared by extraction callers and adapters."""
    try:
        parsed = httpx.URL(url.strip())
    except httpx.InvalidURL as exc:
        raise ValueError("extraction URL was invalid") from exc
    if not parsed.is_absolute_url or parsed.scheme not in {"http", "https"}:
        raise ValueError("extraction URL must be an absolute HTTP URL")
    if parsed.userinfo:
        raise ValueError("extraction URL must not contain credentials")
    return str(parsed.copy_with(path=parsed.path or "/", fragment=None))


__all__ = [
    "ExtractPagesRequest",
    "ExtractPagesResponse",
    "ExtractedPage",
    "PageExtractionError",
    "PageExtractionWarning",
    "canonicalize_extraction_url",
]
