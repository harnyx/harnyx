"""Assignment-scoped validator execution contracts and timing policy."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta
from typing import Literal, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from harnyx_commons.bittensor import build_canonical_request, verify_signed_request
from harnyx_commons.json_types import JsonObject
from harnyx_miner_sdk.endpoint_protocol import EndpointAssignment, EndpointDelegation, query_digest
from harnyx_miner_sdk.query import Query

ENDPOINT_START_ALLOWANCE = timedelta(seconds=60)
ENDPOINT_REPORT_FORWARDING_ALLOWANCE = timedelta(seconds=60)
ENDPOINT_CONTROL_TIMEOUT_SECONDS = 10.0
ENDPOINT_CALLBACK_MAX_BYTES = 1_000_000
ENDPOINT_REPORT_MAX_BYTES = 2_000_000
ENDPOINT_FAILURE_MAX_BYTES = 16 * 1024
DELEGATION_PATH = "/v1/endpoint-execution/delegation"
EXECUTION_PATH = "/validator/endpoint-assignments"
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class EndpointAuthority(BaseModel):
    model_config = _STRICT

    assignment_id: UUID
    validator_hotkey: str = Field(min_length=1)
    miner_hotkey: str = Field(min_length=1)
    query_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    nonce: str = Field(min_length=32, max_length=128)
    endpoint_url: str = Field(min_length=1, max_length=2000)
    callback_url: str = Field(min_length=1, max_length=2000)
    search_url: str = Field(min_length=1, max_length=2000)
    first_scheduled_at: AwareDatetime
    execution_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    attempt_number: int = Field(ge=1)
    started_at: AwareDatetime | None = None
    deadline_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        for value in (self.endpoint_url, self.callback_url, self.search_url):
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("execution URLs must be HTTPS without credentials, query or fragment")
        if (self.started_at is None) != (self.deadline_at is None):
            raise ValueError("start and deadline must be supplied together")
        if self.started_at is not None and self.deadline_at != self.started_at + timedelta(
            seconds=self.execution_timeout_seconds
        ):
            raise ValueError("deadline must preserve the original response window")
        return self

    @property
    def report_cutoff(self) -> datetime:
        if self.deadline_at is None:
            raise ValueError("assignment has not started")
        return self.deadline_at + ENDPOINT_REPORT_FORWARDING_ALLOWANCE

    def assignment(self, query: Query, delegation: EndpointDelegation) -> EndpointAssignment:
        if self.deadline_at is None or query_digest(query) != self.query_digest:
            raise ValueError("assignment requires persisted timing and matching query")
        return EndpointAssignment(
            assignment_id=self.assignment_id,
            query=query,
            query_digest=self.query_digest,
            expected_hotkey=self.miner_hotkey,
            callback_url=self.callback_url,
            search_url=self.search_url,
            endpoint_url=self.endpoint_url,
            callback_context=delegation_header(delegation),
            nonce=self.nonce,
            expires_at=self.deadline_at,
        )


def verify_delegation(delegation: EndpointDelegation, platform_hotkey: str) -> EndpointAuthority:
    if delegation.platform_hotkey != platform_hotkey:
        raise ValueError("delegation signer does not match trusted Platform key")
    verify_signed_request(
        method="POST",
        path_qs=DELEGATION_PATH,
        body=delegation.body_utf8.encode(),
        authorization_header=f'Bittensor ss58="{platform_hotkey}",sig="{delegation.signature_hex}"',
        allowed_ss58=(platform_hotkey,),
    )
    return EndpointAuthority.model_validate_json(delegation.body_utf8, strict=True)


def delegation_signing_bytes(authority: EndpointAuthority) -> bytes:
    return build_canonical_request("POST", DELEGATION_PATH, authority.model_dump_json().encode())


def delegation_header(delegation: EndpointDelegation) -> str:
    return base64.b64encode(delegation.model_dump_json().encode()).decode("ascii")


def parse_delegation_header(value: str | None) -> EndpointDelegation:
    if value is None or len(value) > 24_000:
        raise ValueError("missing or oversized assignment delegation")
    return EndpointDelegation.model_validate_json(base64.b64decode(value, validate=True), strict=True)


class EndpointExecutionWork(BaseModel):
    model_config = _STRICT
    query: Query
    delegation: EndpointDelegation


class EndpointProgress(BaseModel):
    model_config = _STRICT
    assignment_id: UUID
    state: Literal["queued", "active", "reporting", "saved"]


class EndpointStartReport(BaseModel):
    model_config = _STRICT
    delegation: EndpointDelegation
    proposed_start: AwareDatetime


class EndpointResponseReport(BaseModel):
    model_config = _STRICT
    delegation: EndpointDelegation
    received_at: AwareDatetime
    callback_base64: str = Field(max_length=1_333_336)
    signature_hex: str = Field(pattern=r"^[0-9a-f]{128}$")
    signed_callback_path: str = Field(min_length=1, max_length=2000, pattern=r"^/")

    def callback_bytes(self) -> bytes:
        body = base64.b64decode(self.callback_base64, validate=True)
        if len(body) > ENDPOINT_CALLBACK_MAX_BYTES:
            raise ValueError("callback exceeds size limit")
        return body


class EndpointFailureReport(BaseModel):
    model_config = _STRICT
    delegation: EndpointDelegation
    attempted_at: AwareDatetime
    observed_through: AwareDatetime
    outcome: Literal["attempted_no_valid_answer"] = "attempted_no_valid_answer"


class EndpointFailureAcknowledgement(BaseModel):
    model_config = _STRICT
    recorded: bool


class EndpointAssignmentSnapshot(BaseModel):
    model_config = _STRICT
    work: EndpointExecutionWork
    status: Literal["created", "awaiting_callback", "succeeded", "endpoint_failed", "platform_void"]
    response_json: JsonObject | None = None
