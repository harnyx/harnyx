"""Shared type aliases for JSON-compatible values.

These types allow us to keep `pydantic` quarantined to boundary layers while still
modeling JSON payloads precisely throughout the domain and application layers.
"""

from __future__ import annotations

from typing import TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonArray: TypeAlias = list[JsonValue]

LogValue: TypeAlias = JsonPrimitive
LogFields: TypeAlias = dict[str, LogValue]

__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "JsonObject",
    "JsonArray",
    "LogValue",
    "LogFields",
]

