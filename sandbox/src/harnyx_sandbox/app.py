"""FastAPI sandbox runtime for executing miner entrypoints."""

from __future__ import annotations

import argparse
import logging
import os
import secrets
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from harnyx_miner_sdk.sandbox_headers import (
    read_host_container_url_header,
    read_platform_token_header,
    read_session_id_header,
)
from harnyx_miner_sdk.sandbox_protocol import AdmissionRequest, SandboxAdmission
from harnyx_miner_sdk.tools.proxy import PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS
from harnyx_sandbox.sandbox.harness import (
    SandboxCleanupUnconfirmedError,
    SandboxHarness,
)
from harnyx_sandbox.tools.proxy import ToolProxy

logger = logging.getLogger("harnyx_sandbox")

PLATFORM_TOKEN_SCHEME = APIKeyHeader(name="x-platform-token", scheme_name="PlatformToken", auto_error=False)


CONTROL_TOKEN_SCHEME = APIKeyHeader(name="x-sandbox-control-token", scheme_name="SandboxControl", auto_error=False)


async def require_control_token(request: Request, token: str | None = Security(CONTROL_TOKEN_SCHEME)) -> None:
    expected = getattr(request.app.state, "control_token", None)
    if not expected or not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid sandbox control credential")


async def require_tool_token(_request: Request, token: str | None = Security(PLATFORM_TOKEN_SCHEME)) -> str:
    if not token:
        raise HTTPException(status_code=401, detail="missing x-platform-token header")
    return token


def _tool_factory(
    config: Mapping[str, object] | None, headers: Mapping[str, str], *, client: httpx.AsyncClient | None = None
) -> ToolProxy | None:
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
        client=client,
    )


sandbox_harness = SandboxHarness(tool_factory=_tool_factory)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    token = os.environ.pop("SANDBOX_CONTROL_TOKEN", "")
    if not token:
        raise RuntimeError("SANDBOX_CONTROL_TOKEN is required")
    app.state.control_token = token
    logger.info("harnyx-sandbox starting up")
    try:
        yield
    finally:
        await sandbox_harness.close()
    logger.info("harnyx-sandbox shutting down")


app = FastAPI(title="Harnyx Sandbox", version="0.1.0", lifespan=lifespan)


@app.exception_handler(SandboxCleanupUnconfirmedError)
async def cleanup_unconfirmed(_request: Request, _exc: SandboxCleanupUnconfirmedError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": {"code": "SandboxCleanupUnconfirmed"}})


app.include_router(
    sandbox_harness.create_router(),
    prefix="/entry",
    dependencies=[Depends(require_control_token), Depends(require_tool_token)],
)


@app.post("/admission", dependencies=[Depends(require_control_token), Depends(require_tool_token)])
async def reserve_admission(body: AdmissionRequest, request: Request) -> SandboxAdmission:
    return await sandbox_harness.reserve(body, request.headers)


@app.post("/admission/release", dependencies=[Depends(require_control_token), Depends(require_tool_token)])
async def release_admission(body: SandboxAdmission) -> dict[str, bool]:
    sandbox_harness.release(body)
    return {"ok": True}


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
