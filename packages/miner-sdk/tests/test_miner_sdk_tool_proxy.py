from __future__ import annotations

import pytest
from pydantic import ValidationError

import harnyx_miner_sdk.api as miner_api
from harnyx_miner_sdk.tools.http_models import ToolExecuteRequestDTO
from harnyx_miner_sdk.tools.proxy import DEFAULT_TOOL_PROXY_TIMEOUT_SECONDS
from harnyx_miner_sdk.tools.search_models import FetchPageRequest, SearchAiSearchRequest, SearchWebSearchRequest
from harnyx_miner_sdk.tools.types import SEARCH_TOOLS, TOOL_NAMES, parse_tool_name


def test_default_tool_proxy_timeout_remains_120_seconds() -> None:
    assert DEFAULT_TOOL_PROXY_TIMEOUT_SECONDS == pytest.approx(120.0)


def test_search_ai_is_not_available_to_miners() -> None:
    assert not hasattr(miner_api, "search_ai")
    assert "search_ai" not in TOOL_NAMES
    assert "search_ai" not in SEARCH_TOOLS

    with pytest.raises(ValidationError):
        ToolExecuteRequestDTO(tool="search_ai")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unsupported tool 'search_ai'"):
        parse_tool_name("search_ai")


def test_firecrawl_is_available_only_for_ordinary_web_tools() -> None:
    assert SearchWebSearchRequest(
        provider="firecrawl",
        search_queries=("harnyx",),
    ).provider == "firecrawl"
    assert FetchPageRequest(provider="firecrawl", url="https://example.com").provider == "firecrawl"

    with pytest.raises(ValidationError):
        SearchAiSearchRequest(provider="firecrawl", prompt="harnyx")  # type: ignore[arg-type]

    assert SearchAiSearchRequest(provider="parallel", prompt="harnyx").provider == "parallel"


def test_firecrawl_search_rejects_combined_query_over_provider_limit() -> None:
    with pytest.raises(ValidationError, match="500 characters"):
        SearchWebSearchRequest(provider="firecrawl", search_queries=("x" * 499,))
