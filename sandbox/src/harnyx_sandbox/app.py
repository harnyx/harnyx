"""FastAPI sandbox runtime for executing miner entrypoints."""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import traceback
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from harnyx_miner_sdk.sandbox_headers import (
    read_host_container_url_header,
    read_platform_token_header,
    read_session_id_header,
)
from harnyx_miner_sdk.tools.proxy import PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS
from harnyx_sandbox.sandbox.harness import (
    SandboxHarness,
    SandboxPreloadFailure,
)
from harnyx_sandbox.tools.proxy import ToolProxy

logger = logging.getLogger("harnyx_sandbox")

PLATFORM_TOKEN_SCHEME = APIKeyHeader(name="x-platform-token", scheme_name="PlatformToken", auto_error=False)


async def require_tool_token(_request: Request, token: str | None = Security(PLATFORM_TOKEN_SCHEME)) -> str:
    if not token:
        raise HTTPException(status_code=401, detail="missing x-platform-token header")
    return token


def _tool_factory(config: Mapping[str, object] | None, headers: Mapping[str, str]) -> ToolProxy | None:
    if config:
        raise ValueError("tool proxy config is not supported; use request headers")

    base_url = read_host_container_url_header(headers)
    token = read_platform_token_header(headers)
    session_id = read_session_id_header(headers)
    if not session_id or not base_url or not token:
        return None
    return ToolProxy(
        base_url=base_url,
        token=token,
        session_id=session_id,
        timeout=PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS,
    )


sandbox_harness: SandboxHarness | None = None
_agent_loaded = False


def _load_agent_from_env() -> SandboxPreloadFailure | None:
    global _agent_loaded
    if _agent_loaded:
        return None

    preload_failure_type = SandboxPreloadFailure

    def make_preload_infrastructure_failure(message: str, exception: str) -> SandboxPreloadFailure:
        return preload_failure_type(
            code="PreloadInfrastructureFailed",
            error=message,
            exception=exception,
        )

    raw_agent_path = os.getenv("AGENT_PATH")
    agent_path = raw_agent_path.strip() if raw_agent_path is not None else ""
    raw_agent_module = os.getenv("AGENT_MODULE")
    agent_module = raw_agent_module.strip() if raw_agent_module is not None else ""
    if agent_module:
        return make_preload_infrastructure_failure(
            "AGENT_MODULE is not supported; use AGENT_PATH",
            "ValueError",
        )
    if not agent_path:
        return make_preload_infrastructure_failure("AGENT_PATH is required", "ValueError")

    path = Path(agent_path)
    if not path.exists():
        return make_preload_infrastructure_failure(
            "agent path is not present inside sandbox",
            "FileNotFoundError",
        )
    mounted_root_text = str(path.parent.resolve())
    mounted_root_prefix = f"{mounted_root_text}{os.sep}"
    extract_tb = traceback.extract_tb
    log_preload_infrastructure_failure = logger.warning
    log_preload_success = logger.info
    log_preload_exception = logger.exception

    def is_mounted_root_path(filename: str) -> bool:
        return filename == mounted_root_text or filename.startswith(mounted_root_prefix)

    def is_loader_owned_os_error(exc: OSError) -> bool:
        if exc.filename is None:
            return False
        if not is_mounted_root_path(exc.filename):
            return False
        for frame in extract_tb(exc.__traceback__):
            filename = frame.filename
            if not filename or filename.startswith("<"):
                continue
            if is_mounted_root_path(filename):
                return False
        return True

    try:
        try:
            runpy.run_path(str(path))
        except OSError as exc:
            if is_loader_owned_os_error(exc):
                log_preload_infrastructure_failure("sandbox preload infrastructure failed", exc_info=exc)
                return make_preload_infrastructure_failure(
                    "failed to read mounted agent path",
                    exc.__class__.__name__,
                )
            raise
        log_preload_success("loaded agent from path %s", path)
        _agent_loaded = True
        return None
    except Exception as exc:  # pragma: no cover - defensive logging for sandbox startup
        log_preload_exception("failed to load agent", exc_info=exc)
        raise


if sandbox_harness is None:
    sandbox_harness = SandboxHarness(tool_factory=_tool_factory, preload=_load_agent_from_env)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    logger.info("harnyx-sandbox starting up")
    yield
    logger.info("harnyx-sandbox shutting down")


app = FastAPI(title="Harnyx Sandbox", version="0.1.0", lifespan=lifespan)
app.include_router(
    sandbox_harness.create_router(),
    prefix="/entry",
    dependencies=[Depends(require_tool_token)],
)


@app.get("/healthz", tags=["health"], description="Sandbox health check.")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Harnyx sandbox runtime.")
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
        uvicorn.run("harnyx_sandbox.app:app", host=args.host, port=args.port, log_level="info")
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
