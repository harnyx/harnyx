"""Protect callback transport policy across signed authority and miner assignment boundaries."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.endpoint_execution import EndpointAuthority
from harnyx_miner_sdk.endpoint_protocol import EndpointDelegation, query_digest
from harnyx_miner_sdk.query import Query


def _authority_payload(callback_url: str) -> tuple[dict[str, object], Query]:
    query = Query(text="Find the primary source")
    return (
        {
            "assignment_id": uuid4(),
            "validator_hotkey": "validator",
            "miner_hotkey": "miner",
            "query_digest": query_digest(query),
            "nonce": "a" * 64,
            "endpoint_url": "https://miner.example",
            "callback_url": callback_url,
            "search_url": "https://platform.example/search",
            "first_scheduled_at": datetime.now(UTC),
            "execution_timeout_seconds": 30.0,
            "attempt_number": 1,
        },
        query,
    )


def test_authority_and_assignment_preserve_an_http_callback() -> None:
    assignment_id = uuid4()
    callback_url = f"http://validator:8100/validator/endpoint-assignments/{assignment_id}/callback"
    payload, query = _authority_payload(callback_url)
    payload["assignment_id"] = assignment_id
    payload["started_at"] = datetime.now(UTC)
    payload["execution_timeout_seconds"] = 1.0
    payload["deadline_at"] = payload["started_at"] + timedelta(seconds=1)

    authority = EndpointAuthority.model_validate(payload, strict=True)
    assignment = authority.assignment(
        query,
        EndpointDelegation(platform_hotkey="platform", body_utf8="body", signature_hex="a" * 128),
    )

    assert assignment.callback_url == callback_url


def test_authority_keeps_endpoint_and_search_urls_https_only_with_http_callback() -> None:
    assignment_id = uuid4()
    payload, _ = _authority_payload(
        f"http://validator:8100/validator/endpoint-assignments/{assignment_id}/callback"
    )
    payload["assignment_id"] = assignment_id
    for field in ("endpoint_url", "search_url"):
        insecure = payload | {field: "http://internal.example/path"}
        with pytest.raises(ValidationError, match="execution URLs must be HTTPS"):
            EndpointAuthority.model_validate(insecure, strict=True)
