"""Shared type aliases for JSON-compatible values.

These types allow us to keep `pydantic` quarantined to boundary layers while still
modeling JSON payloads precisely throughout the domain and application layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import TypeAliasType

JsonPrimitive: TypeAlias = str | int | float | bool | None

if TYPE_CHECKING:
    JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
    JsonObject: TypeAlias = dict[str, JsonValue]
    JsonArray: TypeAlias = list[JsonValue]
else:
    JsonValue = TypeAliasType(
        "JsonValue",
        JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"],
    )
    JsonObject = TypeAliasType("JsonObject", dict[str, JsonValue])
    JsonArray = TypeAliasType("JsonArray", list[JsonValue])

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
