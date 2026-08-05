"""Shared tool type definitions."""

from __future__ import annotations

from typing import Annotated, Literal, TypeGuard, cast

from pydantic import Field, StrictFloat

ToolInvocationTimeout = Annotated[StrictFloat, Field(gt=0, allow_inf_nan=False)]

MinerToolName = Literal[
    "search_web",
    "fetch_page",
    "embed_text",
    "llm_chat",
    "test_tool",
    "tooling_info",
]
ToolName = MinerToolName
SearchToolName = Literal["search_web", "fetch_page"]
EmbeddingToolName = Literal["embed_text"]
LlmToolName = Literal["llm_chat"]

MINER_TOOL_NAMES: set[MinerToolName] = {
    "search_web",
    "fetch_page",
    "embed_text",
    "llm_chat",
    "test_tool",
    "tooling_info",
}
TOOL_NAMES: set[ToolName] = set(MINER_TOOL_NAMES)
SEARCH_TOOLS: set[SearchToolName] = {"search_web", "fetch_page"}
EMBEDDING_TOOLS: set[EmbeddingToolName] = {"embed_text"}
LLM_TOOLS: set[LlmToolName] = {"llm_chat"}


def parse_tool_name(raw: str) -> ToolName:
    """Parse an external tool string into a canonical ToolName or raise."""
    value = raw.strip()
    if value not in TOOL_NAMES:
        raise ValueError(f"unsupported tool {value!r}")
    return cast(ToolName, value)


def is_search_tool(name: str) -> TypeGuard[SearchToolName]:
    return name in SEARCH_TOOLS


def is_embedding_tool(name: str) -> TypeGuard[EmbeddingToolName]:
    return name in EMBEDDING_TOOLS


def is_citation_source(name: str) -> bool:
    # Today, only search tools can be cited.
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
