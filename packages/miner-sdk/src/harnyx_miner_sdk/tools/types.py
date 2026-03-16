"""Shared tool type definitions."""

from __future__ import annotations

from typing import Literal, TypeGuard, cast

ToolName = Literal[
    "search_web",
    "search_x",
    "search_ai",
    "search_repo",
    "get_repo_file",
    "llm_chat",
    "search_items",
    "test_tool",
    "tooling_info",
]
SearchToolName = Literal["search_web", "search_x", "search_ai", "search_repo", "get_repo_file"]
LlmToolName = Literal["llm_chat"]

TOOL_NAMES: set[ToolName] = {
    "search_web",
    "search_x",
    "search_ai",
    "search_repo",
    "get_repo_file",
    "llm_chat",
    "search_items",
    "test_tool",
    "tooling_info",
}
SEARCH_TOOLS: set[SearchToolName] = {"search_web", "search_x", "search_ai", "search_repo", "get_repo_file"}
LLM_TOOLS: set[LlmToolName] = {"llm_chat"}


def parse_tool_name(raw: str) -> ToolName:
    """Parse an external tool string into a canonical ToolName or raise."""
    value = raw.strip()
    if value not in TOOL_NAMES:
        raise ValueError(f"unsupported tool {value!r}")
    return cast(ToolName, value)


def is_search_tool(name: str) -> TypeGuard[SearchToolName]:
    return name in SEARCH_TOOLS


def is_citation_source(name: str) -> bool:
    # Today, only search tools can be cited.
    return is_search_tool(name)


__all__ = [
    "ToolName",
    "SearchToolName",
    "LlmToolName",
    "TOOL_NAMES",
    "SEARCH_TOOLS",
    "LLM_TOOLS",
    "parse_tool_name",
    "is_search_tool",
    "is_citation_source",
]
