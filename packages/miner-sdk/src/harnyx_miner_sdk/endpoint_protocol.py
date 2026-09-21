"""Version-one wire contract for registered miner endpoint execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Self
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)

from harnyx_miner_sdk.json_types import JsonObject, JsonValue
from harnyx_miner_sdk.query import Query, Response

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, json_schema_mode_override="validation")
ENDPOINT_CALLBACK_CONTEXT_HEADER = "X-Harnyx-Callback-Context"

_HEX_64 = r"^[0-9a-f]{64}$"
_CALLBACK_URL = TypeAdapter(HttpUrl)


def _validated_callback_url(value: str, *, allowed_schemes: frozenset[str]) -> str:
    normalized = value.rstrip("/")
    parsed = urlsplit(normalized)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("callback URL contains an invalid port") from exc
    if (
        parsed.scheme not in allowed_schemes
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        schemes = " or ".join(sorted(scheme.upper() for scheme in allowed_schemes))
        raise ValueError(f"callback URL must use {schemes} without credentials, query or fragment")
    _CALLBACK_URL.validate_python(normalized)
    try:
        transport_url = httpx.URL(normalized)
    except httpx.InvalidURL as exc:
        raise ValueError("invalid callback URL") from exc
    if not transport_url.host:
        raise ValueError("callback URL requires a host")
    return normalized


def validate_endpoint_callback_url(value: str) -> str:
    """Validate the HTTP transport URL carried by the signed endpoint protocol."""

    return _validated_callback_url(value, allowed_schemes=frozenset({"http", "https"}))


class EndpointMinerStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class EndpointSearchTool(StrEnum):
    SEARCH_WEB = "search_web"
    FETCH_PAGE = "fetch_page"


class EndpointDurableTerminalResult(StrEnum):
    PERSISTED = "persisted"
    CLOSED = "closed"


class EndpointDelegation(BaseModel):
    """Original Platform-signed assignment authority, retained across validator retries."""

    model_config = _STRICT
    platform_hotkey: str = Field(min_length=1)
    body_utf8: str = Field(min_length=1, max_length=12_000)
    signature_hex: str = Field(min_length=128, max_length=128, pattern=r"^[0-9a-f]+$")


class EndpointAssignment(BaseModel):
    model_config = _STRICT

    assignment_id: UUID
    query: Query
    query_digest: str = Field(pattern=_HEX_64)
    expected_hotkey: str = Field(min_length=1)
    callback_url: str = Field(min_length=1, max_length=2000)
    search_url: str = Field(min_length=1, max_length=2000)
    endpoint_url: str = Field(min_length=1, max_length=2000)
    callback_context: str | None = Field(default=None, max_length=24_000)
    nonce: str = Field(min_length=32, max_length=128)
    expires_at: datetime

    @field_validator("search_url", "endpoint_url")
    @classmethod
    def validate_https_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("endpoint and search URLs must use HTTPS")
        return value.rstrip("/")

    @field_validator("callback_url")
    @classmethod
    def normalize_callback_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("callback_context")
    @classmethod
    def validate_context(cls, value: str | None) -> str | None:
        if value is not None and (value != value.strip() or any(ord(c) < 32 or ord(c) > 126 for c in value)):
            raise ValueError("callback context must be an unchanged ASCII HTTP header value")
        return value

    @field_validator("expires_at")
    @classmethod
    def validate_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expiry must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_query_binding(self) -> Self:
        if self.query_digest != query_digest(self.query):
            raise ValueError("query digest does not match query")
        validate_endpoint_callback_url(self.callback_url)
        return self


class EndpointCallback(BaseModel):
    model_config = _STRICT

    assignment_id: UUID
    query_digest: str = Field(pattern=_HEX_64)
    nonce: str = Field(min_length=32, max_length=128)
    expires_at: datetime
    response: Response

    @field_validator("expires_at")
    @classmethod
    def validate_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expiry must include a timezone")
        return value


class EndpointStatusResponse(BaseModel):
    model_config = _STRICT

    state: EndpointMinerStatus


class EndpointAssignmentAcknowledgement(BaseModel):
    model_config = _STRICT

    accepted: bool = True


class EndpointSearchRequest(BaseModel):
    model_config = _STRICT

    receipt_id: str = Field(min_length=1, max_length=256, pattern=r"^[^\x00]+$")
    provider: str = Field(min_length=1, max_length=64)
    tool: EndpointSearchTool
    args: list[JsonValue] = Field(default_factory=list, max_length=32)
    kwargs: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_search_input(self) -> Self:
        # Reject invalid input without changing the request's keys or values.
        pending: list[JsonValue] = [self.args, self.kwargs]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if any("\x00" in key for key in value):
                    raise ValueError("search object keys must not contain NUL")
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, float) and not isfinite(value):
                raise ValueError("search numbers must be finite")
        return self


class EndpointSearchResult(BaseModel):
    model_config = _STRICT

    result_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    note: str | None = None
    title: str | None = None


class EndpointSearchResponse(BaseModel):
    model_config = _STRICT

    receipt_id: str = Field(min_length=1, max_length=256)
    response: JsonObject
    results: tuple[EndpointSearchResult, ...]


class EndpointCallbackAcknowledgement(BaseModel):
    model_config = _STRICT

    durable_terminal_result: EndpointDurableTerminalResult


def query_digest(query: Query) -> str:
    """Hash the stable public query representation used by both peers."""

    payload = json.dumps(
        query.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ENDPOINT_CALLBACK_CONTEXT_HEADER",
    "EndpointAssignment",
    "EndpointDelegation",
    "EndpointAssignmentAcknowledgement",
    "EndpointCallback",
    "EndpointCallbackAcknowledgement",
    "EndpointDurableTerminalResult",
    "EndpointMinerStatus",
    "EndpointSearchRequest",
    "EndpointSearchResponse",
    "EndpointSearchResult",
    "EndpointSearchTool",
    "EndpointStatusResponse",
    "query_digest",
    "validate_endpoint_callback_url",
]
