"""Helper for coercing tool responses into JSON-friendly structures."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from typing import ParamSpec, TypeVar, cast

from pydantic import BaseModel

from caster_commons.json_types import JsonValue


def _normalize_payload(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_payload(asdict(value))
    if isinstance(value, BaseModel):
        return _normalize_payload(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {key: _normalize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    return str(value)


P = ParamSpec("P")
T = TypeVar("T")


def normalize_response(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        result = await func(*args, **kwargs)
        normalized = _normalize_payload(result)
        return cast(T, normalized)

    return wrapper


__all__ = ["normalize_response"]
