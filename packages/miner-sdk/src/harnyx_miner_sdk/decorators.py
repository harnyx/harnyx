"""Entrypoint registration helpers used by sandboxed miner agents."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, ParamSpec, TypeVar, cast, get_type_hints, overload

from pydantic import TypeAdapter

from harnyx_miner_sdk.query import Query
from harnyx_miner_sdk.query import Response as QueryResponse

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
        self._entrypoints[name] = _compile_entrypoint(name, func)
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


@overload
def entrypoint(name: None = None) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def entrypoint(
    name: Literal["query"],
) -> Callable[
    [Callable[[Query], Awaitable[QueryResponse]]],
    Callable[[Query], Awaitable[QueryResponse]],
]: ...


@overload
def entrypoint(name: str) -> Any: ...


def entrypoint(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that registers a callable as a miner entrypoint."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        entrypoint_name = name or cast(Any, func).__name__
        _ENTRYPOINT_REGISTRY.register(entrypoint_name, func)
        return func

    return decorator


@overload
def get_entrypoint(
    name: Literal["query"],
) -> Callable[[object], Awaitable[QueryResponse]]: ...


@overload
def get_entrypoint(name: str) -> Callable[..., Any]: ...


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


@dataclass(frozen=True, slots=True)
class _CompiledEntrypointSpec:
    parameter_name: str
    request_adapter: TypeAdapter[Any]
    response_adapter: TypeAdapter[Any]


def _compile_entrypoint(name: str, func: Callable[..., Any]) -> Callable[[object], Awaitable[Any]]:
    spec = _build_entrypoint_spec(name, func)

    async def invoke(request: object) -> Any:
        parsed_request = spec.request_adapter.validate_python(request)
        result = await cast(Callable[..., Awaitable[Any]], func)(**{spec.parameter_name: parsed_request})
        return spec.response_adapter.validate_python(result)

    return invoke


def _build_entrypoint_spec(name: str, func: Callable[..., Any]) -> _CompiledEntrypointSpec:
    parameter_name = _entrypoint_parameter_name(_assert_entrypoint_signature(func))
    request_type, response_type = _entrypoint_types(name, func, parameter_name)
    if name == "query":
        _assert_query_contract(request_type, response_type)
    return _CompiledEntrypointSpec(
        parameter_name=parameter_name,
        request_adapter=TypeAdapter(request_type),
        response_adapter=TypeAdapter(response_type),
    )


def _assert_entrypoint_signature(func: Callable[..., Any]) -> inspect.Signature:
    if not inspect.iscoroutinefunction(func):
        raise TypeError("entrypoints must be declared with 'async def'")

    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    if len(parameters) != 1:
        raise TypeError("entrypoints must accept exactly one parameter")
    parameter = parameters[0]
    if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        raise TypeError("entrypoint parameter must be passable as a keyword argument")
    if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        raise TypeError("entrypoints must not accept *args or **kwargs")
    return signature


def _entrypoint_parameter_name(signature: inspect.Signature) -> str:
    return next(iter(signature.parameters.values())).name


def _entrypoint_types(
    name: str,
    func: Callable[..., Any],
    parameter_name: str,
) -> tuple[Any, Any]:
    type_hints = get_type_hints(func)
    request_type = _require_type_hint(type_hints.get(parameter_name), f"entrypoint {name!r} parameter")
    response_type = _require_type_hint(type_hints.get("return"), f"entrypoint {name!r} return")
    return request_type, response_type


def _assert_query_contract(request_type: Any, response_type: Any) -> None:
    if request_type is not Query:
        raise TypeError("query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query")
    if response_type is not QueryResponse:
        raise TypeError("query entrypoint return type must be harnyx_miner_sdk.query.Response")


def _require_type_hint(annotation: Any, label: str) -> Any:
    if annotation is None:
        raise TypeError(f"{label} must be annotated")
    return annotation


__all__ = [
    "EntrypointRegistry",
    "RegisteredEntrypoint",
    "clear_entrypoints",
    "entrypoint",
    "entrypoint_exists",
    "get_entrypoint",
    "get_entrypoint_registry",
    "iter_entrypoints",
]
