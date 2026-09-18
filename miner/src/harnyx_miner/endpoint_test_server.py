"""Runnable in-memory endpoint that exercises the signed version-one protocol."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

import bittensor as bt
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response as HttpResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from harnyx_commons.bittensor import VerificationError, build_canonical_request, verify_signed_request
from harnyx_commons.config.platform_api import PlatformApiSettings
from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_commons.endpoint_execution import ENDPOINT_REPORT_FORWARDING_ALLOWANCE
from harnyx_miner.tooling import require_tooling_url
from harnyx_miner.validator_eligibility import ValidatorEligibility
from harnyx_miner_sdk.endpoint_protocol import (
    ENDPOINT_CALLBACK_CONTEXT_HEADER,
    EndpointAssignment,
    EndpointAssignmentAcknowledgement,
    EndpointCallback,
    EndpointCallbackAcknowledgement,
    EndpointDurableTerminalResult,
    EndpointMinerStatus,
    EndpointSearchRequest,
    EndpointSearchResponse,
    EndpointSearchTool,
    EndpointStatusResponse,
)
from harnyx_miner_sdk.query import CitationRef, Response

_MAX_ASSIGNMENT_BODY_BYTES = 1_000_000
logger = logging.getLogger(__name__)


class _OwnershipChallenge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    purpose: Literal["harnyx.miner_endpoint_ownership.v1"]
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    hotkey: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=2000)
    block_at_registration: int = Field(ge=0)
    expires_at: AwareDatetime


async def _read_bounded_body(request: Request, *, max_bytes: int = _MAX_ASSIGNMENT_BODY_BYTES) -> bytes:
    content_length = request.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = 0
        if declared_length > max_bytes:
            raise HTTPException(status_code=413, detail=f"request body exceeds {max_bytes} bytes")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail=f"request body exceeds {max_bytes} bytes")
        body.extend(chunk)
    return bytes(body)


@dataclass(slots=True)
class EndpointTestServerState:
    signers: dict[UUID, set[str]] = field(default_factory=dict, repr=False)
    assignments: dict[UUID, EndpointAssignment] = field(default_factory=dict, repr=False)
    statuses: dict[UUID, EndpointMinerStatus] = field(default_factory=dict, repr=False)
    callbacks: dict[UUID, EndpointCallback] = field(default_factory=dict, repr=False)
    callback_attempts: dict[UUID, int] = field(default_factory=dict)
    active_tasks: dict[UUID, asyncio.Task[None]] = field(default_factory=dict, repr=False)
    destinations: dict[UUID, dict[str, EndpointAssignment]] = field(default_factory=dict, repr=False)
    delivery_tasks: dict[tuple[UUID, str], asyncio.Task[None]] = field(default_factory=dict, repr=False)
    acknowledged_destinations: set[tuple[UUID, str]] = field(default_factory=set, repr=False)
    acknowledged: set[UUID] = field(default_factory=set, repr=False)

    def clear(self) -> None:
        parent_owned = set(self.active_tasks)
        for task in self.active_tasks.values():
            task.cancel()
        self.active_tasks.clear()
        self.assignments.clear()
        self.signers.clear()
        self.statuses.clear()
        self.callbacks.clear()
        self.callback_attempts.clear()
        self.acknowledged.clear()
        self.acknowledged_destinations.clear()
        self.destinations.clear()
        for key, task in self.delivery_tasks.items():
            if key[0] not in parent_owned:
                task.cancel()
        self.delivery_tasks.clear()


def _canonical_endpoint_url(endpoint_url: str) -> str:
    if len(endpoint_url.encode()) > 2000 or any(ord(char) < 33 or ord(char) == 127 for char in endpoint_url):
        raise ValueError("endpoint URL is invalid")
    url = httpx.URL(endpoint_url)
    if url.scheme != "https" or not url.host or url.userinfo or any(char in endpoint_url for char in "?#\\"):
        raise ValueError("endpoint must be an HTTPS base URL without credentials, query or fragment")
    return str(url).rstrip("/")


async def _require_eligible(validator_eligibility: Callable[[str], Awaitable[bool]], signer: str) -> None:
    try:
        eligible = await validator_eligibility(signer)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="validator eligibility is unavailable") from exc
    if not eligible:
        raise HTTPException(status_code=403, detail="validator requires subnet registration and permit")


def create_endpoint_test_app(
    *,
    validator_eligibility: Callable[[str], Awaitable[bool]],
    platform_base_url: str | None = None,
    miner_hotkey: bt.Keypair,
    endpoint_url: str,
    block_at_registration: int,
    state: EndpointTestServerState | None = None,
    provider: str | None = None,
    provider_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    callback_retry_seconds: float = 0.05,
    callback_retry_max_seconds: float = 10,
    callback_attempt_timeout_seconds: float = 5,
    answerer: Callable[[EndpointAssignment], Awaitable[Response]] | None = None,
) -> FastAPI:
    """Build a registerable endpoint; provider inputs enable its search/callback loop."""

    canonical_url = _canonical_endpoint_url(endpoint_url)
    if block_at_registration < 0:
        raise ValueError("registration block must be nonnegative")
    # Strip literal trailing URL slashes before decoding; %2F is part of the registered base.
    base_path = httpx.URL(canonical_url).path if httpx.URL(canonical_url).raw_path != b"/" else ""

    server_state = state or EndpointTestServerState()
    http_client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
    owns_client = client is None
    tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if owns_client:
                await http_client.aclose()

    app = FastAPI(title="Harnyx miner endpoint protocol test server", lifespan=lifespan, root_path=base_path)
    app.state.endpoint_background_tasks = tasks

    @app.middleware("http")
    async def require_registered_base(
        request: Request,
        call_next: Callable[[Request], Awaitable[HttpResponse]],
    ) -> HttpResponse:
        # Keep URL data out of route templates and require the externally signed prefix.
        if not request.scope["path"].startswith(base_path + "/"):
            return HttpResponse(status_code=404)
        return await call_next(request)

    def schedule(assignment: EndpointAssignment, work: Callable[[], Coroutine[Any, Any, None]]) -> None:
        identity = assignment.assignment_id
        if identity in server_state.active_tasks:
            return
        task = asyncio.create_task(work())
        server_state.active_tasks[identity] = task
        tasks.add(task)

        def completed(done: asyncio.Task[None]) -> None:
            tasks.discard(done)
            if not done.cancelled() and done.exception() is not None:
                # Exception messages/tracebacks can contain the transient provider key.
                operation = "callback delivery" if identity in server_state.callbacks else "search"
                logger.warning("endpoint %s failed", operation, extra={"assignment_id": str(identity)})
            if server_state.active_tasks.get(identity) is done:
                server_state.active_tasks.pop(identity)
                if identity not in server_state.callbacks:
                    server_state.statuses.pop(identity, None)

        task.add_done_callback(completed)

    def schedule_delivery(assignment: EndpointAssignment) -> None:
        key = (assignment.assignment_id, assignment.callback_url)
        if key in server_state.acknowledged_destinations:
            return
        task = server_state.delivery_tasks.get(key)
        if task is not None and not task.done():
            return
        callback = server_state.callbacks[assignment.assignment_id]
        task = asyncio.create_task(
            _deliver_callback(
                assignment=assignment,
                callback=callback,
                miner_hotkey=miner_hotkey,
                state=server_state,
                client=http_client,
                retry_seconds=callback_retry_seconds,
                retry_max_seconds=callback_retry_max_seconds,
                attempt_timeout_seconds=callback_attempt_timeout_seconds,
            )
        )
        server_state.delivery_tasks[key] = task
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    @app.post("/verify")
    async def prove_ownership(request: Request) -> dict[str, str]:
        body = await _read_bounded_body(request, max_bytes=8192)
        try:
            challenge = _OwnershipChallenge.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail="invalid ownership challenge") from exc
        if (
            challenge.hotkey != miner_hotkey.ss58_address
            or challenge.url != canonical_url
            or challenge.block_at_registration != block_at_registration
            or challenge.expires_at <= datetime.now(UTC)
        ):
            raise HTTPException(
                status_code=403,
                detail="ownership challenge does not match this endpoint or has expired",
            )
        path = request.scope["raw_path"].decode("ascii")
        signature = miner_hotkey.sign(build_canonical_request("POST", path, body)).hex()
        return {"signature": signature}

    @app.post("/v1/endpoint-assignments")
    async def accept_assignment(request: Request) -> HttpResponse:
        path = request.scope["raw_path"].decode("ascii")
        body = await _read_bounded_body(request)
        try:
            assignment = EndpointAssignment.model_validate_json(body, strict=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid endpoint assignment") from exc
        signer = _verify_or_401(method="POST", path=path, body=body, authorization=request.headers.get("Authorization"))
        if assignment.expected_hotkey != miner_hotkey.ss58_address or assignment.endpoint_url != canonical_url:
            raise HTTPException(status_code=403, detail="assignment targets another miner endpoint")
        if assignment.expires_at <= datetime.now(UTC):
            raise HTTPException(status_code=409, detail="assignment has expired")
        if answerer is None and assignment.query.output_schema is not None:
            raise HTTPException(status_code=422, detail="this example supports text queries only")
        if signer not in server_state.signers.get(assignment.assignment_id, set()):
            await _require_eligible(validator_eligibility, signer)
        # Recheck and retain atomically after remote I/O; no await in this state transition.
        existing = server_state.assignments.get(assignment.assignment_id)
        if existing is not None and existing.model_dump(
            exclude={"callback_url", "callback_context"}
        ) != assignment.model_dump(exclude={"callback_url", "callback_context"}):
            raise HTTPException(status_code=409, detail="assignment identity conflicts")
        # Do not let another signed request replace an already retained callback's context.
        destinations = server_state.destinations.get(assignment.assignment_id, {})
        destination = destinations.get(assignment.callback_url)
        if destination is not None and destination.callback_context != assignment.callback_context:
            raise HTTPException(status_code=409, detail="callback destination context conflicts")
        if assignment.expires_at <= datetime.now(UTC):
            raise HTTPException(status_code=409, detail="assignment has expired")
        if existing is None:
            server_state.assignments[assignment.assignment_id] = assignment
            server_state.statuses[assignment.assignment_id] = EndpointMinerStatus.RUNNING
        server_state.signers.setdefault(assignment.assignment_id, set()).add(signer)
        server_state.destinations.setdefault(assignment.assignment_id, {})[assignment.callback_url] = assignment
        if assignment.assignment_id in server_state.callbacks:
            schedule_delivery(assignment)
        if existing is None and (answerer is not None or (provider is not None and provider_key is not None)):
            schedule(
                assignment,
                lambda: _execute_assignment(
                    assignment=assignment,
                    platform_base_url=platform_base_url,
                    provider=provider,
                    provider_key=provider_key,
                    miner_hotkey=miner_hotkey,
                    state=server_state,
                    client=http_client,
                    callback_retry_seconds=callback_retry_seconds,
                    callback_retry_max_seconds=callback_retry_max_seconds,
                    callback_attempt_timeout_seconds=callback_attempt_timeout_seconds,
                    answerer=answerer,
                ),
            )
        return _signed_json(
            method="POST",
            path=path,
            payload=EndpointAssignmentAcknowledgement(),
            hotkey=miner_hotkey,
        )

    @app.get("/v1/endpoint-assignments/{assignment_id}/status")
    async def assignment_status(assignment_id: UUID, request: Request) -> HttpResponse:
        path = request.scope["raw_path"].decode("ascii")
        signer = _verify_or_401(method="GET", path=path, body=b"", authorization=request.headers.get("Authorization"))
        if assignment_id not in server_state.assignments:
            await _require_eligible(validator_eligibility, signer)
        # An assignment may have been admitted while eligibility was being read.
        if assignment_id in server_state.assignments and signer not in server_state.signers.get(assignment_id, set()):
            raise HTTPException(status_code=403, detail="validator has not been admitted to this assignment")
        status = server_state.statuses.get(assignment_id)
        callback = server_state.callbacks.get(assignment_id)
        assignment = server_state.assignments.get(assignment_id)
        if status is None and assignment is not None:
            raise HTTPException(status_code=503, detail="assignment execution is unavailable")
        if (
            callback is not None
            and assignment is not None
            and assignment_id not in server_state.acknowledged
            and datetime.now(UTC) < assignment.expires_at
        ):
            for destination in server_state.destinations.get(assignment_id, {}).values():
                schedule_delivery(destination)
        return _signed_json(
            method="GET",
            path=path,
            payload=EndpointStatusResponse(state=status or EndpointMinerStatus.UNKNOWN),
            hotkey=miner_hotkey,
        )

    return app


async def _execute_assignment(
    *,
    platform_base_url: str | None = None,
    assignment: EndpointAssignment,
    provider: str | None,
    provider_key: str | None,
    miner_hotkey: bt.Keypair,
    state: EndpointTestServerState,
    client: httpx.AsyncClient,
    callback_retry_seconds: float,
    callback_retry_max_seconds: float,
    callback_attempt_timeout_seconds: float,
    answerer: Callable[[EndpointAssignment], Awaitable[Response]] | None = None,
) -> None:
    remaining = (assignment.expires_at - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise TimeoutError("assignment expired before answering")
    async with asyncio.timeout(remaining):
        if answerer is not None:
            response = await answerer(assignment)
        else:
            assert provider is not None and provider_key is not None
            response = await _search_snippet(
                assignment, provider, provider_key, miner_hotkey, client, platform_base_url
            )
    if state.assignments.get(assignment.assignment_id) is not assignment:
        return
    callback = EndpointCallback(
        assignment_id=assignment.assignment_id,
        query_digest=assignment.query_digest,
        nonce=assignment.nonce,
        expires_at=assignment.expires_at,
        response=response,
    )
    state.callbacks[assignment.assignment_id] = callback
    state.statuses[assignment.assignment_id] = EndpointMinerStatus.COMPLETED
    async with asyncio.TaskGroup() as deliveries:
        for destination in tuple(
            state.destinations.get(assignment.assignment_id, {assignment.callback_url: assignment}).values()
        ):
            key = (assignment.assignment_id, destination.callback_url)
            task = deliveries.create_task(
                _deliver_callback(
                    assignment=destination,
                    callback=callback,
                    miner_hotkey=miner_hotkey,
                    state=state,
                    client=client,
                    retry_seconds=callback_retry_seconds,
                    retry_max_seconds=callback_retry_max_seconds,
                    attempt_timeout_seconds=callback_attempt_timeout_seconds,
                )
            )
            state.delivery_tasks[key] = task


async def _search_snippet(
    assignment: EndpointAssignment,
    provider: str,
    provider_key: str,
    miner_hotkey: bt.Keypair,
    client: httpx.AsyncClient,
    platform_base_url: str | None,
) -> Response:
    require_tooling_url(assignment.search_url, assignment.assignment_id, platform_base_url)
    receipt_id = f"endpoint-test-{assignment.assignment_id}"
    search = EndpointSearchRequest(
        receipt_id=receipt_id,
        provider=provider,
        tool=EndpointSearchTool.SEARCH_WEB,
        kwargs={"provider": provider, "search_queries": [assignment.query.text]},
    )
    search_url = assignment.search_url
    search_response = await _post_signed_json(
        client=client,
        url=search_url,
        path=httpx.URL(search_url).raw_path.decode("ascii"),
        payload=search,
        hotkey=miner_hotkey,
        headers={"X-Provider-Api-Key": provider_key},
        deadline_at=assignment.expires_at,
    )
    search_response.raise_for_status()
    result = EndpointSearchResponse.model_validate_json(search_response.content, strict=True)
    first = result.results[0] if result.results else None
    source_text = first.note if first is not None and first.note and first.note.strip() else None
    citations = [CitationRef(receipt_id=receipt_id, result_id=first.result_id)] if first and source_text else None
    answer = source_text or first.title if first else "No search results were returned."
    return Response(text=answer or "The search result did not include a summary.", citations=citations)


async def _deliver_callback(
    *,
    assignment: EndpointAssignment,
    callback: EndpointCallback,
    miner_hotkey: bt.Keypair,
    state: EndpointTestServerState,
    client: httpx.AsyncClient,
    retry_seconds: float,
    retry_max_seconds: float,
    attempt_timeout_seconds: float,
) -> None:
    path = httpx.URL(assignment.callback_url).raw_path.decode("ascii")
    owner = state.assignments.get(assignment.assignment_id)
    cutoff = assignment.expires_at + ENDPOINT_REPORT_FORWARDING_ALLOWANCE
    retry_delay = min(retry_seconds, retry_max_seconds)
    while (
        owner is not None and state.assignments.get(assignment.assignment_id) is owner and datetime.now(UTC) <= cutoff
    ):
        state.callback_attempts[assignment.assignment_id] = state.callback_attempts.get(assignment.assignment_id, 0) + 1
        try:
            response = await _post_signed_json(
                client=client,
                url=assignment.callback_url,
                path=path,
                payload=callback,
                hotkey=miner_hotkey,
                deadline_at=min(cutoff, datetime.now(UTC) + timedelta(seconds=attempt_timeout_seconds)),
                headers=(
                    {ENDPOINT_CALLBACK_CONTEXT_HEADER: assignment.callback_context}
                    if assignment.callback_context is not None
                    else {}
                ),
            )
            if response.status_code == 200:
                acknowledgement = EndpointCallbackAcknowledgement.model_validate_json(response.content, strict=True)
                if acknowledgement.durable_terminal_result in {
                    EndpointDurableTerminalResult.PERSISTED,
                    EndpointDurableTerminalResult.CLOSED,
                }:
                    if state.assignments.get(assignment.assignment_id) is owner:
                        state.acknowledged.add(assignment.assignment_id)
                        state.acknowledged_destinations.add((assignment.assignment_id, assignment.callback_url))
                    return
        except (TimeoutError, httpx.HTTPError, ValueError):
            pass
        remaining = (cutoff - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            return
        await asyncio.sleep(min(retry_delay, remaining))
        retry_delay = min(retry_delay * 2, retry_max_seconds)


async def _post_signed_json(
    *,
    client: httpx.AsyncClient,
    url: str,
    path: str,
    payload: object,
    hotkey: bt.Keypair,
    headers: dict[str, str] | None = None,
    deadline_at: datetime | None = None,
) -> httpx.Response:
    body = payload.model_dump_json().encode("utf-8")  # type: ignore[attr-defined]
    signature = hotkey.sign(build_canonical_request("POST", path, body)).hex()
    authorization = f'Bittensor ss58="{hotkey.ss58_address}",sig="{signature}"'
    signed_headers = {**(headers or {}), "Authorization": authorization}
    if deadline_at is None:
        return await client.post(url, content=body, headers=signed_headers, follow_redirects=False)
    remaining = (deadline_at - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise TimeoutError("assignment expired before signed request")
    async with asyncio.timeout(remaining):
        return await client.post(url, content=body, headers=signed_headers, timeout=remaining, follow_redirects=False)


def _verify_or_401(
    *,
    method: str,
    path: str,
    body: bytes,
    authorization: str | None,
) -> str:
    try:
        return verify_signed_request(
            method=method,
            path_qs=path,
            body=body,
            authorization_header=authorization,
        ).ss58
    except VerificationError as exc:
        raise HTTPException(status_code=401, detail=exc.message) from exc


def _signed_json(*, method: str, path: str, payload: object, hotkey: bt.Keypair) -> HttpResponse:
    body = payload.model_dump_json().encode("utf-8")  # type: ignore[attr-defined]
    signature = hotkey.sign(build_canonical_request(method, path, body)).hex()
    authorization = f'Bittensor ss58="{hotkey.ss58_address}",sig="{signature}"'
    return HttpResponse(content=body, media_type="application/json", headers={"Authorization": authorization})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the signed Harnyx miner endpoint protocol test server")
    parser.add_argument("--miner-hotkey-uri", required=True)
    parser.add_argument("--endpoint-url", required=True, help="public HTTPS base URL submitted for registration")
    parser.add_argument(
        "--registration-block",
        required=True,
        type=int,
        help="the hotkey's current BlockAtRegistration",
    )
    parser.add_argument("--provider")
    parser.add_argument(
        "--provider-key-env",
        help="environment variable containing the transient provider key; the key is never retained in server state",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--ssl-certfile", required=True)
    parser.add_argument("--ssl-keyfile", required=True)
    args = parser.parse_args()
    provider_key = os.environ[args.provider_key_env] if args.provider_key_env else None
    if (args.provider is None) != (provider_key is None):
        parser.error("--provider and --provider-key-env must be supplied together")
    app = create_endpoint_test_app(
        validator_eligibility=ValidatorEligibility(SubtensorSettings()),
        platform_base_url=PlatformApiSettings().platform_base_url,
        miner_hotkey=bt.Keypair.create_from_uri(args.miner_hotkey_uri),
        endpoint_url=args.endpoint_url,
        block_at_registration=args.registration_block,
        provider=args.provider,
        provider_key=provider_key,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


__all__ = ["EndpointTestServerState", "create_endpoint_test_app", "main"]
