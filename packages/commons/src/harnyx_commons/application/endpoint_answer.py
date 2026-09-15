"""Verify original miner evidence before identity-blind quality judging."""

from harnyx_commons.application.miner_response_hydration import hydrate_miner_response_payload
from harnyx_commons.bittensor import verify_signed_request
from harnyx_commons.domain.miner_task import Query, Response
from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolResultPolicy
from harnyx_commons.endpoint_answer import EndpointAnswer
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_miner_sdk.endpoint_protocol import EndpointCallback, query_digest


def verified_endpoint_answer(answer: EndpointAnswer, query: Query) -> Response:
    body = answer.callback_body_utf8.encode("utf-8")
    verify_signed_request(
        method="POST",
        path_qs=answer.signed_callback_path,
        body=body,
        authorization_header=f'Bittensor ss58="{answer.expected_hotkey}",sig="{answer.signature_hex}"',
        allowed_ss58=(answer.expected_hotkey,),
    )
    callback = EndpointCallback.model_validate_json(body, strict=True)
    if callback.assignment_id != answer.assignment_id or callback.query_digest != query_digest(query):
        raise ValueError("signed answer does not match assigned comparison query/assignment")
    receipt_log = InMemoryReceiptLog()
    for receipt in answer.receipt_logs:
        if receipt.assignment_id != answer.assignment_id:
            raise ValueError("receipt does not belong to successful assignment work")
        receipt_log.record(
            ToolCall(
                receipt_id=receipt.receipt_id,
                session_id=receipt.assignment_id,
                uid=0,
                tool=receipt.tool,
                issued_at=receipt.issued_at,
                outcome=ToolCallOutcome.OK,
                details=ToolCallDetails(
                    request_hash="platform-verified",
                    results=receipt.results,
                    result_policy=ToolResultPolicy.REFERENCEABLE,
                ),
            )
        )
    response = hydrate_miner_response_payload(
        callback.response.model_dump(mode="json"),
        query=query,
        session_id=answer.assignment_id,
        receipt_log=receipt_log,
    )
    if response.citations and any(citation is None for citation in response.citations):
        raise ValueError("signed answer citation does not resolve through assignment receipts")
    return response
