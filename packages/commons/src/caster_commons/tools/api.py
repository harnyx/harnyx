"""Compatibility shim forwarding miner-facing tool helpers to caster-miner SDK."""

from caster_miner_sdk.api import (
    LlmChatResult,
    TestToolResponse,
    ToolCallResponse,
    llm_chat,
    search_web,
    search_x,
    test_tool,
)
from caster_miner_sdk.decorators import (
    clear_entrypoints,
    entrypoint,
    entrypoint_exists,
    get_entrypoint,
    get_entrypoint_registry,
    iter_entrypoints,
)

__all__ = [
    "clear_entrypoints",
    "entrypoint",
    "entrypoint_exists",
    "get_entrypoint",
    "get_entrypoint_registry",
    "iter_entrypoints",
    "llm_chat",
    "search_x",
    "search_web",
    "test_tool",
    "LlmChatResult",
    "ToolCallResponse",
    "TestToolResponse",
]
