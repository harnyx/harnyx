"""Protect the public registered-endpoint wire contract."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.endpoint_protocol import (
    EndpointAssignment,
    EndpointCallback,
    EndpointCallbackAcknowledgement,
    EndpointDurableTerminalResult,
    EndpointMinerStatus,
    EndpointSearchRequest,
    EndpointSearchTool,
    EndpointStatusResponse,
    query_digest,
)
from harnyx_miner_sdk.query import CitationRef, Query, Response


def _assignment() -> EndpointAssignment:
    query = Query(text="Find the primary source")
    return EndpointAssignment(
        assignment_id=uuid4(),
        query=query,
        query_digest=query_digest(query),
        expected_hotkey="5ExpectedMinerHotkey",
        callback_url="https://platform.example/v1/endpoint-assignments/callback",
        nonce="a" * 64,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )


def test_assignment_round_trip_preserves_query_and_binding() -> None:
    assignment = _assignment()

    decoded = EndpointAssignment.model_validate_json(assignment.model_dump_json())

    assert decoded == assignment
    assert decoded.query_digest == query_digest(decoded.query)


def test_query_digest_is_stable_and_sensitive_to_public_query_fields() -> None:
    query = Query(text="same", fast=True, output_schema={"type": "string"})

    assert query_digest(query) == query_digest(Query.model_validate(query.model_dump(mode="json")))
    assert query_digest(query) != query_digest(query.model_copy(update={"fast": False}))


def test_callback_preserves_response_citation_refs() -> None:
    assignment = _assignment()
    response = Response(
        text="Answer",
        citations=[CitationRef(receipt_id="receipt-1", result_id="result-1")],
    )

    callback = EndpointCallback(
        assignment_id=assignment.assignment_id,
        query_digest=assignment.query_digest,
        nonce=assignment.nonce,
        expires_at=assignment.expires_at,
        response=response,
    )

    assert EndpointCallback.model_validate_json(callback.model_dump_json()).response == response


def test_protocol_rejects_extra_fields_and_insecure_callback_urls() -> None:
    payload = _assignment().model_dump(mode="json")
    payload["internal_query_id"] = str(uuid4())
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EndpointAssignment.model_validate(payload)

    payload.pop("internal_query_id")
    payload["callback_url"] = "http://platform.example/callback"
    with pytest.raises(ValidationError, match="HTTPS"):
        EndpointAssignment.model_validate(payload)


def test_status_search_and_acknowledgement_are_bounded() -> None:
    status = EndpointStatusResponse(state=EndpointMinerStatus.UNKNOWN)
    request = EndpointSearchRequest(
        receipt_id="receipt-1",
        provider="tavily",
        tool=EndpointSearchTool.SEARCH_WEB,
        args=[],
        kwargs={"query": "primary source"},
    )
    ack = EndpointCallbackAcknowledgement(durable_terminal_result=EndpointDurableTerminalResult.PERSISTED)

    assert status.state is EndpointMinerStatus.UNKNOWN
    assert request.tool is EndpointSearchTool.SEARCH_WEB
    assert ack.durable_terminal_result.value == "persisted"


@pytest.mark.parametrize("tool", ["execute_python", "search_ai"])
def test_search_rejects_tools_unavailable_to_miners(tool):
    payload = {"receipt_id": "receipt-1", "provider": "desearch", "tool": tool, "kwargs": {"prompt": "query"}}
    with pytest.raises(ValidationError, match="tool"):
        EndpointSearchRequest.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize("field", ["args", "kwargs"])
@pytest.mark.parametrize("key", ["bad\x00key", "bad\ud800key"])
def test_search_rejects_invalid_nested_object_keys(field, key):
    nested = {"provider": "parallel", "nested": [{key: "value"}]}
    payload = {
        "receipt_id": "receipt",
        "provider": "parallel",
        "tool": "search_web",
        field: [nested] if field == "args" else nested,
    }
    with pytest.raises(ValidationError):
        EndpointSearchRequest.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize("field", ["args", "kwargs"])
@pytest.mark.parametrize(
    "number", [float("nan"), float("inf"), -float("inf")], ids=["nan", "infinity", "negative-infinity"]
)
def test_search_rejects_nested_non_finite_numbers(field, number):
    nested = {"nested": [{"number": number}]}
    payload = {
        "receipt_id": "receipt",
        "provider": "parallel",
        "tool": "search_web",
        field: [nested] if field == "args" else nested,
    }
    with pytest.raises(ValidationError, match="finite"):
        EndpointSearchRequest.model_validate_json(json.dumps(payload), strict=True)


def test_search_preserves_valid_unicode_keys_and_string_values():
    kwargs = {"证据/é\n": [{"": "text\x00value"}], "emoji😀": True}
    payload = {"receipt_id": "receipt", "provider": "parallel", "tool": "search_web", "kwargs": kwargs}
    request = EndpointSearchRequest.model_validate_json(json.dumps(payload), strict=True)
    assert request.kwargs == kwargs


def test_search_preserves_finite_numbers_and_non_numeric_values():
    values = [0, -42, 1.5, -0.0, 1.7976931348623157e308, True, None, "NaN", "Infinity", "-Infinity"]
    payload = {
        "receipt_id": "receipt",
        "provider": "parallel",
        "tool": "search_web",
        "args": values,
        "kwargs": {"nested": values},
    }
    request = EndpointSearchRequest.model_validate_json(json.dumps(payload), strict=True)
    assert request.args == values
    assert request.kwargs == {"nested": values}


@pytest.mark.parametrize("receipt", ["\x00", "\x00prefix", "in\x00side", "suffix\x00"])
def test_search_rejects_receipt_ids_that_cannot_be_persisted(receipt: str) -> None:
    with pytest.raises(ValidationError, match="receipt_id"):
        EndpointSearchRequest.model_validate_json(
            json.dumps({"receipt_id": receipt, "provider": "parallel", "tool": "search_web"})
        )


@pytest.mark.parametrize("receipt", ["évidence/证据", " line\nbreak\t ", "r" * 256])
def test_search_preserves_valid_receipt_identity(receipt: str) -> None:
    request = EndpointSearchRequest.model_validate_json(
        json.dumps({"receipt_id": receipt, "provider": "parallel", "tool": "search_web"})
    )
    assert request.receipt_id == receipt
