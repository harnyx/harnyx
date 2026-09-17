"""Signed HTTPS client for registered miner endpoint assignments."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import bittensor as bt
import httpx
from pydantic import ValidationError

from harnyx_commons.bittensor import VerificationError, build_canonical_request, verify_signed_request
from harnyx_commons.endpoint_execution import ENDPOINT_DELEGATION_HEADER, delegation_header
from harnyx_commons.task_ownership import await_owned_task
from harnyx_miner_sdk.endpoint_protocol import (
    EndpointAssignment,
    EndpointAssignmentAcknowledgement,
    EndpointDelegation,
    EndpointStatusResponse,
)
from harnyx_validator.application.endpoint_execution import EndpointStatusUnavailableError, MinerRequestAttempt

_ASSIGNMENT_PATH = "/v1/endpoint-assignments"
_MAX_RESPONSE_BYTES = 8 * 1024


class _EndpointRequestNotSentError(RuntimeError):
    """The deadline expired before transport was entered."""


@dataclass(slots=True)
class SignedMinerEndpointClient:
    validator_hotkey: bt.Keypair
    client: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            transport=httpx.AsyncHTTPTransport(retries=0),
        )
    )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def send_assignment(
        self,
        *,
        endpoint_url: str,
        expected_hotkey: str,
        assignment: EndpointAssignment,
        stop_at: float,
    ) -> MinerRequestAttempt:
        target = _endpoint_target(endpoint_url, _ASSIGNMENT_PATH)
        attempt = MinerRequestAttempt()
        path = target.raw_path.decode("ascii")
        remaining = (assignment.expires_at - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            return attempt
        body = assignment.model_dump_json().encode("utf-8")
        try:
            response = await self._request(
                method="POST",
                url=str(target),
                path=path,
                body=body,
                deadline_at=assignment.expires_at,
                attempt=attempt,
                delegation=assignment.delegation,
                stop_at=stop_at,
            )
        except (
            _EndpointRequestNotSentError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.InvalidURL,
            httpx.UnsupportedProtocol,
        ):
            return attempt
        except (httpx.HTTPError, TimeoutError, OSError, RuntimeError):
            return attempt
        if response.status_code != 200:
            return attempt
        try:
            verify_signed_request(
                method="POST",
                path_qs=path,
                body=response.content,
                authorization_header=response.headers.get("Authorization"),
                allowed_ss58=(expected_hotkey,),
            )
            EndpointAssignmentAcknowledgement.model_validate_json(response.content, strict=True)
        except (ValidationError, VerificationError):
            return attempt
        return attempt

    async def get_status(
        self,
        *,
        endpoint_url: str,
        expected_hotkey: str,
        assignment_id: UUID,
        deadline_at: datetime,
        stop_at: float,
        delegation: EndpointDelegation,
    ) -> EndpointStatusResponse:
        suffix = f"/v1/endpoint-assignments/{assignment_id}/status"
        target = _endpoint_target(endpoint_url, suffix)
        path = target.raw_path.decode("ascii")
        try:
            response = await self._request(
                method="GET",
                url=str(target),
                path=path,
                body=b"",
                deadline_at=deadline_at,
                delegation=delegation,
                stop_at=stop_at,
            )
            if response.status_code != 200:
                raise RuntimeError(f"miner endpoint status returned HTTP {response.status_code}")
            verify_signed_request(
                method="GET",
                path_qs=path,
                body=response.content,
                authorization_header=response.headers.get("Authorization"),
                allowed_ss58=(expected_hotkey,),
            )
            return EndpointStatusResponse.model_validate_json(response.content, strict=True)
        except (httpx.HTTPError, TimeoutError, OSError, RuntimeError, VerificationError, ValidationError) as exc:
            raise EndpointStatusUnavailableError("miner endpoint status unavailable") from exc

    async def _request(
        self,
        *,
        method: str,
        url: str,
        path: str,
        body: bytes,
        deadline_at: datetime,
        stop_at: float,
        delegation: EndpointDelegation,
        attempt: MinerRequestAttempt | None = None,
    ) -> httpx.Response:
        loop = asyncio.get_running_loop()
        remaining = (deadline_at - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise _EndpointRequestNotSentError("deadline expired before transport")
        expires = min(stop_at, loop.time() + min(remaining, 10.0))
        async with asyncio.timeout_at(expires):
            signature = self.validator_hotkey.sign(build_canonical_request(method, path, body)).hex()
            authorization = f'Bittensor ss58="{self.validator_hotkey.ss58_address}",sig="{signature}"'
            remaining = expires - loop.time()
            if remaining <= 0:
                raise _EndpointRequestNotSentError("deadline expired before transport")
            if attempt is not None:
                attempt.attempted_at = datetime.now(UTC)
            async with self.client.stream(
                method,
                url,
                content=body,
                headers={
                    "Authorization": authorization,
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                    ENDPOINT_DELEGATION_HEADER: delegation_header(delegation),
                },
                timeout=min(remaining, 10.0),
            ) as streamed:
                try:
                    # Only identity-encoded bytes are accepted; decoding is unnecessary.
                    if streamed.headers.get("Content-Encoding", "identity").strip().lower() != "identity":
                        raise RuntimeError("miner endpoint response must use identity encoding")
                    raw = bytearray()
                    # Iterators on Response close implicitly at EOF, outside our owned cleanup.
                    assert isinstance(streamed.stream, httpx.AsyncByteStream)
                    async for chunk in streamed.stream:
                        if len(raw) + len(chunk) > _MAX_RESPONSE_BYTES:
                            raise RuntimeError("miner endpoint response exceeds 8 KiB")
                        raw.extend(chunk)
                    response = httpx.Response(
                        status_code=streamed.status_code,
                        headers=streamed.headers,
                        content=bytes(raw),
                        request=streamed.request,
                    )
                finally:
                    # A total timeout followed by parent cancellation must still finish closing.
                    await await_owned_task(asyncio.create_task(streamed.aclose()))
            if loop.time() >= expires:
                raise TimeoutError("assignment deadline expired")
        return response


def _require_https_base(endpoint_url: str) -> httpx.URL:
    parsed = httpx.URL(endpoint_url)
    if parsed.scheme != "https" or not parsed.host or parsed.query or parsed.fragment or parsed.userinfo:
        raise ValueError("miner endpoint must be an HTTPS base URL")
    return parsed


def _endpoint_target(endpoint_url: str, suffix: str) -> httpx.URL:
    base = _require_https_base(endpoint_url)
    return httpx.URL(str(base).rstrip("/") + suffix)


__all__ = ["SignedMinerEndpointClient"]
