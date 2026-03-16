"""Query request/response contracts for miners."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_MINER_SDK_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    str_strip_whitespace=True,
)

class Query(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    text: str = Field(min_length=1)


class Response(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    text: str = Field(min_length=1)


__all__ = ["Query", "Response"]
