from __future__ import annotations

import argparse
import asyncio
import json
import runpy
from collections.abc import Sequence
from pathlib import Path

from harnyx_miner_sdk._internal.tool_invoker import bind_tool_invoker
from harnyx_miner_sdk.decorators import clear_entrypoints, entrypoint_exists, get_entrypoint
from harnyx_miner_sdk.query import Query, Response

_DEFAULT_QUERY_TEXT = "Harnyx Subnet validators manage sandboxed miners."


def _load_agent(agent_path: Path) -> None:
    clear_entrypoints()
    runpy.run_path(str(agent_path))
    if not entrypoint_exists("query"):
        raise RuntimeError("agent did not register entrypoint 'query'")


def _build_request(*, request_path: Path | None, query_text: str | None) -> dict[str, object]:
    if request_path is not None:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
        payload = Query.model_validate(raw)
        return payload.model_dump()

    payload = Query(text=query_text or _DEFAULT_QUERY_TEXT)
    return payload.model_dump()


def _serialize_response(response: Response) -> str:
    return json.dumps(response.model_dump(mode="json"))


def _existing_path(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _optional_existing_path(raw_path: str | None, *, label: str) -> Path | None:
    if raw_path is None:
        return None
    return _existing_path(raw_path, label=label)


async def _invoke_query_entrypoint(request_payload: dict[str, object]) -> Response:
    try:
        from harnyx_commons.tools.local_dev_host import create_local_tool_host
    except ImportError as exc:  # pragma: no cover - only hit outside the mono-workspace
        raise RuntimeError("harnyx-miner-dev requires harnyx-commons (install the full workspace)") from exc

    invoker = create_local_tool_host()
    entrypoint = get_entrypoint("query")
    try:
        with bind_tool_invoker(invoker):
            return await entrypoint(request_payload)
    finally:
        await invoker.aclose()


async def _amain(argv: Sequence[str] | None) -> None:
    parser = argparse.ArgumentParser(description="Run a miner agent locally with real tool calls.")
    parser.add_argument("--agent-path", required=True, help="Path to the miner agent file.")
    parser.add_argument("--request-json", help="Path to a Query JSON file.")
    parser.add_argument("--query-text", help="Override query text for the default request payload.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    agent_path = _existing_path(args.agent_path, label="agent path")
    request_path = _optional_existing_path(args.request_json, label="request json")
    _load_agent(agent_path)
    request_payload = _build_request(request_path=request_path, query_text=args.query_text)
    result = await _invoke_query_entrypoint(request_payload)
    print(_serialize_response(result))


def main(argv: Sequence[str] | None = None) -> None:
    try:
        asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["main"]
