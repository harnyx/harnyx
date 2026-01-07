"""Entrypoint registration helpers used by sandboxed miner agents."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class RegisteredEntrypoint:
    """Metadata describing a registered entrypoint."""

    name: str
    callable: Callable[..., Any]


class EntrypointRegistry:
    """In-memory registry of agent entrypoints."""

    def __init__(self) -> None:
        self._entrypoints: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[P, R]) -> Callable[P, R]:
        if name in self._entrypoints:
            raise ValueError(f"entrypoint {name!r} is already registered")
        _assert_entrypoint_signature(func)
        self._entrypoints[name] = func
        return func

    def get(self, name: str) -> Callable[..., Any]:
        return self._entrypoints[name]

    def exists(self, name: str) -> bool:
        return name in self._entrypoints

    def clear(self) -> None:
        self._entrypoints.clear()

    def iter(self) -> Iterable[RegisteredEntrypoint]:
        for name, func in self._entrypoints.items():
            yield RegisteredEntrypoint(name=name, callable=func)


_ENTRYPOINT_REGISTRY = EntrypointRegistry()


def entrypoint(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that registers a callable as a miner entrypoint."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        entrypoint_name = name or func.__name__
        _ENTRYPOINT_REGISTRY.register(entrypoint_name, func)
        return func

    return decorator


def _assert_entrypoint_signature(func: Callable[..., Any]) -> None:
    if not inspect.iscoroutinefunction(func):
        raise TypeError("entrypoints must be declared with 'async def'")

    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    if len(params) != 1:
        raise TypeError("entrypoints must accept exactly one parameter: 'request'")
    request_param = params[0]
    if request_param.name != "request":
        raise TypeError("entrypoints must accept a single parameter named 'request'")
    if request_param.kind is inspect.Parameter.POSITIONAL_ONLY:
        raise TypeError("entrypoint request parameter must be passable as a keyword argument")
    if request_param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        raise TypeError("entrypoints must not accept *args or **kwargs")


def get_entrypoint(name: str) -> Callable[..., Any]:
    return _ENTRYPOINT_REGISTRY.get(name)


def entrypoint_exists(name: str) -> bool:
    return _ENTRYPOINT_REGISTRY.exists(name)


def iter_entrypoints() -> Iterable[RegisteredEntrypoint]:
    return _ENTRYPOINT_REGISTRY.iter()


def clear_entrypoints() -> None:
    _ENTRYPOINT_REGISTRY.clear()


def get_entrypoint_registry() -> EntrypointRegistry:
    return _ENTRYPOINT_REGISTRY


__all__ = [
    "entrypoint",
    "get_entrypoint",
    "entrypoint_exists",
    "iter_entrypoints",
    "clear_entrypoints",
    "get_entrypoint_registry",
    "EntrypointRegistry",
    "RegisteredEntrypoint",
]
