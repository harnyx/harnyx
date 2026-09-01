"""Entrypoint registration helpers used by sandboxed miner agents."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, ParamSpec, TypeVar, cast, get_type_hints, overload

from pydantic import TypeAdapter

from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.query import Query
from harnyx_miner_sdk.query import Response as QueryResponse

P = ParamSpec("P")
R = TypeVar("R")
QueryEntrypoint = (
    Callable[[Query], Awaitable[QueryResponse]]
    | Callable[[Query, ContextSnapshot], Awaitable[QueryResponse]]
)
Q = TypeVar("Q", bound=QueryEntrypoint)


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
) -> Callable[[Q], Q]: ...


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
) -> Callable[[object, object], Awaitable[QueryResponse]]: ...


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
    request_parameter_name: str
    request_adapter: TypeAdapter[Any]
    response_adapter: TypeAdapter[Any]
    context_parameter_name: str | None = None
    context_adapter: TypeAdapter[Any] | None = None


def _compile_entrypoint(name: str, func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    spec = _build_entrypoint_spec(name, func)

    if name != "query":
        async def invoke(request: object) -> Any:
            parsed_request = spec.request_adapter.validate_python(request)
            result = await cast(Callable[..., Awaitable[Any]], func)(
                **{spec.request_parameter_name: parsed_request}
            )
            return spec.response_adapter.validate_python(result)

        return invoke

    context_parameter_name = spec.context_parameter_name
    context_adapter = spec.context_adapter
    if context_adapter is None:  # pragma: no cover - construction invariant
        raise RuntimeError("query entrypoint context adapter is missing")

    if context_parameter_name is None:
        async def invoke_legacy_query(request: object, context: object) -> Any:
            parsed_request = spec.request_adapter.validate_python(request)
            context_adapter.validate_python(context)
            result = await cast(Callable[..., Awaitable[Any]], func)(
                **{spec.request_parameter_name: parsed_request}
            )
            return spec.response_adapter.validate_python(result)

        return invoke_legacy_query

    async def invoke_query(request: object, context: object) -> Any:
        parsed_request = spec.request_adapter.validate_python(request)
        parsed_context = context_adapter.validate_python(context)
        result = await cast(Callable[..., Awaitable[Any]], func)(
            **{
                spec.request_parameter_name: parsed_request,
                context_parameter_name: parsed_context,
            }
        )
        return spec.response_adapter.validate_python(result)

    return invoke_query


def _build_entrypoint_spec(name: str, func: Callable[..., Any]) -> _CompiledEntrypointSpec:
    parameters = _entrypoint_parameters(name, _assert_entrypoint_signature(func))
    request_parameter = parameters[0]
    context_parameter = parameters[1] if name == "query" and len(parameters) == 2 else None
    request_type, context_type, response_type = _entrypoint_types(
        name,
        func,
        request_parameter.name,
        None if context_parameter is None else context_parameter.name,
    )
    if name == "query":
        _assert_query_contract(request_type, context_type, response_type)
    return _CompiledEntrypointSpec(
        request_parameter_name=request_parameter.name,
        request_adapter=TypeAdapter(request_type),
        response_adapter=TypeAdapter(response_type),
        context_parameter_name=None if context_parameter is None else context_parameter.name,
        context_adapter=TypeAdapter(ContextSnapshot) if name == "query" else None,
    )


def _assert_entrypoint_signature(func: Callable[..., Any]) -> inspect.Signature:
    if not inspect.iscoroutinefunction(func):
        raise TypeError("entrypoints must be declared with 'async def'")

    return inspect.signature(func)


def _entrypoint_parameters(name: str, signature: inspect.Signature) -> tuple[inspect.Parameter, ...]:
    parameters = tuple(signature.parameters.values())
    if name == "query":
        if len(parameters) not in {1, 2}:
            raise TypeError("query entrypoints must accept one or two parameters")
    elif len(parameters) != 1:
        raise TypeError("entrypoints must accept exactly one parameter")
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError("entrypoint parameters must be passable as keyword arguments")
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError("entrypoints must not accept *args or **kwargs")
    return parameters


def _entrypoint_types(
    name: str,
    func: Callable[..., Any],
    request_parameter_name: str,
    context_parameter_name: str | None,
) -> tuple[Any, Any | None, Any]:
    type_hints = get_type_hints(func)
    request_type = _require_type_hint(
        type_hints.get(request_parameter_name),
        f"entrypoint {name!r} parameter",
    )
    context_type = (
        None
        if context_parameter_name is None
        else _require_type_hint(
            type_hints.get(context_parameter_name),
            f"entrypoint {name!r} context",
        )
    )
    response_type = _require_type_hint(type_hints.get("return"), f"entrypoint {name!r} return")
    return request_type, context_type, response_type


def _assert_query_contract(request_type: Any, context_type: Any | None, response_type: Any) -> None:
    if request_type is not Query:
        raise TypeError("query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query")
    if context_type is not None and context_type is not ContextSnapshot:
        raise TypeError(
            "query entrypoint context must be annotated as harnyx_miner_sdk.context.ContextSnapshot"
        )
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
