"""Request/response models for platform-owned repo-search callbacks."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchRepoSearchRequest(BaseModel):
    """Query parameters for the platform repo-search callback."""

    model_config = ConfigDict(extra="forbid")

    repo_url: str
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    query: str
    path_glob: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class SearchRepoResult(BaseModel):
    """Single repository search result item."""

    model_config = ConfigDict(extra="ignore")

    path: str
    bm25: float | None = None
    url: str
    excerpt: str | None = None
    title: str | None = None


class SearchRepoSearchResponse(BaseModel):
    """Response payload for the platform repo-search callback."""

    data: list[SearchRepoResult] = Field(default_factory=list)


class GetRepoFileRequest(BaseModel):
    """Query parameters for the platform repo file callback."""

    model_config = ConfigDict(extra="forbid")

    repo_url: str
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class GetRepoFileResult(BaseModel):
    """Single repository file response item."""

    model_config = ConfigDict(extra="ignore")

    path: str
    url: str
    text: str
    excerpt: str | None = None
    title: str | None = None


class GetRepoFileResponse(BaseModel):
    """Response payload for the platform repo file callback."""

    data: list[GetRepoFileResult] = Field(default_factory=list)


__all__ = [
    "SearchRepoSearchRequest",
    "SearchRepoSearchResponse",
    "SearchRepoResult",
    "GetRepoFileRequest",
    "GetRepoFileResponse",
    "GetRepoFileResult",
]
