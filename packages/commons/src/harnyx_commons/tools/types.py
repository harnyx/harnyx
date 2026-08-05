"""Host-side tool types, including names retained for internal records."""

from __future__ import annotations

from typing import Literal, TypeGuard, cast

from harnyx_miner_sdk.tools.types import (
    EMBEDDING_TOOLS,
    LLM_TOOLS,
    MINER_TOOL_NAMES,
    EmbeddingToolName,
    LlmToolName,
    MinerToolName,
    ToolInvocationTimeout,
    is_embedding_tool,
)

ToolName = Literal[
    "search_web",
    "search_ai",
    "fetch_page",
    "embed_text",
    "llm_chat",
    "test_tool",
    "tooling_info",
]
SearchToolName = Literal["search_web", "search_ai", "fetch_page"]

TOOL_NAMES: set[ToolName] = {
    "search_web",
    "search_ai",
    "fetch_page",
    "embed_text",
    "llm_chat",
    "test_tool",
    "tooling_info",
}
SEARCH_TOOLS: set[SearchToolName] = {"search_web", "search_ai", "fetch_page"}


def parse_tool_name(raw: str) -> ToolName:
    """Parse a host-side tool string into a canonical ToolName or raise."""
    value = raw.strip()
    if value not in TOOL_NAMES:
        raise ValueError(f"unsupported tool {value!r}")
    return cast(ToolName, value)


def is_search_tool(name: str) -> TypeGuard[SearchToolName]:
    return name in SEARCH_TOOLS


def is_citation_source(name: str) -> bool:
    return is_search_tool(name)


__all__ = [
    "ToolInvocationTimeout",
    "MinerToolName",
    "ToolName",
    "SearchToolName",
    "EmbeddingToolName",
    "LlmToolName",
    "MINER_TOOL_NAMES",
    "TOOL_NAMES",
    "SEARCH_TOOLS",
    "EMBEDDING_TOOLS",
    "LLM_TOOLS",
    "parse_tool_name",
    "is_search_tool",
    "is_embedding_tool",
    "is_citation_source",
]
