"""Compatibility shim forwarding entrypoint helpers to caster-miner SDK."""

from caster_miner_sdk.decorators import (
    EntrypointRegistry,
    RegisteredEntrypoint,
    clear_entrypoints,
    entrypoint,
    entrypoint_exists,
    get_entrypoint,
    get_entrypoint_registry,
    iter_entrypoints,
)

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
