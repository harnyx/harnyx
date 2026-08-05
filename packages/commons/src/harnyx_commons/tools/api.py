"""Compatibility shim forwarding miner-facing tool helpers to harnyx-miner SDK."""

from harnyx_miner_sdk.api import (
    EmbedTextResponse,
    LlmChatResult,
    TestToolResponse,
    ToolCallResponse,
    embed_text,
    fetch_page,
    llm_chat,
    search_web,
    test_tool,
    tooling_info,
)
from harnyx_miner_sdk.decorators import (
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
    "embed_text",
    "fetch_page",
    "llm_chat",
    "search_web",
    "test_tool",
    "tooling_info",
    "EmbedTextResponse",
    "LlmChatResult",
    "ToolCallResponse",
    "TestToolResponse",
]
