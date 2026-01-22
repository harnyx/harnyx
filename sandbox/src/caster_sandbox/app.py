"""FastAPI sandbox runtime for executing miner entrypoints."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import runpy
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from caster_sandbox.sandbox.harness import SandboxHarness
from caster_sandbox.tools.proxy import ToolProxy

logger = logging.getLogger("caster_sandbox")


def _token_header() -> str:
    raw = (os.getenv("CASTER_TOKEN_HEADER") or "").strip()
    return raw or "x-caster-token"


def _tool_factory(config: Mapping[str, object] | None, headers: Mapping[str, str]) -> ToolProxy | None:
    raw_base_url = (config or {}).get("base_url")
    if raw_base_url is not None and not isinstance(raw_base_url, str):
        raise ValueError("tool proxy config.base_url must be a string")
    base_url = raw_base_url or os.getenv("CASTER_VALIDATOR_URL")

    raw_token = (config or {}).get("token")
    if raw_token is not None and not isinstance(raw_token, str):
        raise ValueError("tool proxy config.token must be a string")
    token_header = _token_header()
    token = raw_token or headers.get(token_header)
    session_id = headers.get("x-caster-session-id")
    if not base_url or not token or not session_id:
        logger.warning(
            "tool proxy missing configuration",
            extra={"base_url": base_url, "token_present": bool(token), "session_id": session_id},
        )
        return None
    return ToolProxy(
        base_url=base_url,
        token=token,
        session_id=session_id,
        token_header=token_header,
    )


sandbox_harness: SandboxHarness | None = None
_agent_loaded = False


def _load_agent_from_env() -> None:
    global _agent_loaded
    if _agent_loaded:
        return

    agent_path = os.getenv("CASTER_AGENT_PATH")
    agent_module = os.getenv("CASTER_AGENT_MODULE")

    try:
        if agent_path:
            path = Path(agent_path)
            if not path.exists():
                logger.warning("agent path %s is not present inside sandbox", agent_path)
            else:
                runpy.run_path(str(path))
                logger.info("loaded agent from path %s", path)
                _agent_loaded = True
        elif agent_module:
            importlib.import_module(agent_module)
            logger.info("loaded agent module %s", agent_module)
            _agent_loaded = True
    except Exception as exc:  # pragma: no cover - defensive logging for sandbox startup
        logger.exception("failed to load agent", exc_info=exc)
        raise


if sandbox_harness is None:
    sandbox_harness = SandboxHarness(tool_factory=_tool_factory, preload=_load_agent_from_env)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    logger.info("caster-sandbox starting up")
    yield
    logger.info("caster-sandbox shutting down")


app = FastAPI(title="Caster Sandbox", version="0.1.0", lifespan=lifespan)
app.include_router(sandbox_harness.create_router(), prefix="/entry")


@app.get("/healthz", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Caster sandbox runtime.")
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI app with uvicorn.")
    parser.add_argument(
        "--host",
        default=os.getenv("SANDBOX_HOST", "127.0.0.1"),
        help="Host interface when serving the app.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SANDBOX_PORT", "8000")),
        help="Port when serving the app.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.serve:
        import uvicorn

        logger.info("starting uvicorn on %s:%s", args.host, args.port)
        uvicorn.run("caster_sandbox.app:app", host=args.host, port=args.port, log_level="info")
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()

