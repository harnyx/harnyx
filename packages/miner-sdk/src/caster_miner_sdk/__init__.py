"""Compatibility shim exporting the legacy miner SDK import root."""

from __future__ import annotations

import importlib
import sys

from harnyx_miner_sdk import *  # noqa: F403

_ALIASED_SUBMODULES = (
    "api",
    "decorators",
    "json_types",
    "llm",
    "query",
    "sandbox_headers",
    "verdict",
    "_internal",
    "_internal.tool_invoker",
    "context",
    "context.snapshot",
    "tools",
    "tools.http_models",
    "tools.proxy",
    "tools.search_models",
    "tools.types",
)

for _submodule in _ALIASED_SUBMODULES:
    sys.modules[f"{__name__}.{_submodule}"] = importlib.import_module(
        f"harnyx_miner_sdk.{_submodule}"
    )
