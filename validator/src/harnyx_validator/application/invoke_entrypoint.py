"""Use case for invoking the miner query entrypoint under concurrency control."""

from __future__ import annotations

import json
import re
from uuid import UUID

from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from harnyx_commons.application.miner_response_hydration import (
    MinerResponsePayloadError,
    hydrate_miner_response_payload,
)
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.ports.session_registry import SessionRegistryPort
from harnyx_commons.application.ports.token_registry import TokenRegistryPort
from harnyx_commons.domain.session import Session, SessionStatus
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.sandbox.client import SandboxClient, SandboxInvokeError
from harnyx_commons.tools.dto import session_budget_snapshot
from harnyx_commons.tools.http_serialization import serialize_tool_budget
from harnyx_validator.application.dto.evaluation import EntrypointInvocationRequest, EntrypointInvocationResult

QUERY_ENTRYPOINT = "query"


class SandboxInvocationError(RuntimeError):
    """Raised when a sandbox entrypoint fails to execute."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        detail_code: str | None,
        detail_exception: str | None,
        detail_error: str | None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail_code = detail_code
        self.detail_exception = detail_exception
        self.detail_error = detail_error


class MinerResponseValidationError(RuntimeError):
    """Raised when miner output violates the response contract."""

    def __init__(self, message: str, *, rejected_response: str | None = None) -> None:
        super().__init__(message)
        self.rejected_response = rejected_response


_GENERIC_REJECTION = "miner returned invalid response payload"
_HYDRATION_REASONS = frozenset(
    {
        "legacy query response must use text",
        "structured query response must use output",
        "response text must not be blank",
        "response note must not be blank",
        "response must include exactly one answer field",
        "response text must not be null",
        "value must be finite JSON",
        "response output exceeds 80000 compact JSON characters",
        "response citations exceed 400 materialized evidence segments",
        "response citations exceed 120000 materialized source-text characters",
        "citation slice start must be non-negative",
        "citation slice end must be greater than start",
        "inline citation position is out of range",
        "cited result has no source text",
        "citation slice exceeds source text length",
        "citation slice must contain at least 100 characters",
    }
)
_PYDANTIC_REASONS = {
    "missing": "required field is missing",
    "extra_forbidden": "unexpected response field",
    "string_type": "must be a string",
    "string_too_short": "string is shorter than the allowed minimum",
    "string_too_long": "string exceeds the allowed maximum length",
    "list_type": "must be a list",
    "dict_type": "must be an object",
    "model_type": "must be an object",
    "int_type": "must be an integer",
    "greater_than_equal": "value is below the allowed minimum",
    "greater_than": "value must exceed the allowed minimum",
    "too_long": "exceeds the allowed maximum size",
}
_SCHEMA_REASONS = {
    "type": "value has the wrong type",
    "required": "required field is missing",
    "additionalProperties": "unexpected field",
    "minLength": "string is shorter than the allowed minimum",
    "maxLength": "string exceeds the allowed maximum length",
    "minItems": "too few items",
    "maxItems": "too many items",
    "minimum": "value is below the allowed minimum",
    "maximum": "value exceeds the allowed maximum",
    "enum": "value is not an allowed option",
    "const": "value does not match the required value",
    "pattern": "string does not match the required pattern",
}


def _known_hydration_reason(message: str) -> str | None:
    if message in _HYDRATION_REASONS:
        return message
    if re.fullmatch(r"inline citation \[\[[0-9]{1,3}\]\] points to an unresolved citation", message):
        return "inline citation points to an unresolved citation"
    return None


def _public_rejection_reason(exc: Exception) -> str:
    """Describe validation metadata without publishing library input/schema dumps."""
    cause: BaseException | None = exc
    visited: set[int] = set()
    for _ in range(16):
        if cause is None or id(cause) in visited:
            break
        visited.add(id(cause))
        if isinstance(cause, JsonSchemaValidationError):
            # Schema property names are declared by the task, unlike instance paths
            # whose arbitrary keys can contain submitted answer text.
            schema_path = iter(cause.absolute_schema_path)
            fields = []
            for part in schema_path:
                if part == "properties":
                    fields.append(str(next(schema_path)))
            location = "output" + "".join(f"[{json.dumps(field, ensure_ascii=True)}]" for field in fields)
            reason = _SCHEMA_REASONS.get(cause.validator, "value does not match the output schema")
            return f"{location[:256]}: {reason}"[:512]
        if isinstance(cause, ValidationError):
            for error in cause.errors(include_url=False, include_input=False):
                context = error.get("ctx", {})
                known = _known_hydration_reason(str(context.get("error", "")))
                if known is not None:
                    return known
                reason = _PYDANTIC_REASONS.get(error["type"])
                if reason is not None:
                    safe_fields = {
                        "text",
                        "output",
                        "note",
                        "citations",
                        "receipt_id",
                        "result_id",
                        "slices",
                        "start",
                        "end",
                    }
                    location = ".".join(
                        str(part) for part in error["loc"] if part in safe_fields or isinstance(part, int)
                    )
                    return f"{location[:256] or 'response'}: {reason}"[:512]
        if isinstance(cause, MinerResponsePayloadError | ValueError):
            known = _known_hydration_reason(str(cause))
            if known is not None:
                return known
        cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
    return _GENERIC_REJECTION


class EntrypointInvoker:
    """Coordinates entrypoint invocation with token concurrency enforcement."""

    def __init__(
        self,
        session_registry: SessionRegistryPort,
        sandbox_client: SandboxClient,
        token_registry: TokenRegistryPort,
        receipt_log: ReceiptLogPort,
    ) -> None:
        self._sessions = session_registry
        self._sandbox = sandbox_client
        self._tokens = token_registry
        self._receipts = receipt_log

    async def invoke(self, request: EntrypointInvocationRequest) -> EntrypointInvocationResult:
        """Invoke the requested entrypoint after validating the session token."""
        session = self._load_session(request.session_id)
        self._validate_session(session, request)
        payload = await self._invoke_query(request=request, session=session)
        self._raise_if_session_exhausted(session.session_id)
        receipts = tuple(self._receipts.for_session(session.session_id))
        try:
            hydrated_response = hydrate_miner_response_payload(
                payload,
                query=request.query,
                session_id=session.session_id,
                receipt_log=self._receipts,
            )
        except (MinerResponsePayloadError, ValidationError) as exc:
            raise MinerResponseValidationError(
                _public_rejection_reason(exc),
                rejected_response=json.dumps(payload, ensure_ascii=True),
            ) from exc
        return EntrypointInvocationResult(
            response=hydrated_response,
            tool_receipts=receipts,
        )

    async def _invoke_query(
        self,
        *,
        request: EntrypointInvocationRequest,
        session: Session,
    ) -> object:
        token = request.token
        cost_budget = serialize_tool_budget(session_budget_snapshot(session))
        try:
            return await self._sandbox.invoke(
                QUERY_ENTRYPOINT,
                payload=request.query.model_dump(mode="json"),
                context={
                    "cost_budget": cost_budget.model_dump(mode="json"),
                    "time_budget": {
                        "limit_seconds": request.execution_time_limit_seconds,
                    },
                },
                token=token,
                session_id=session.session_id,
            )
        except SandboxInvokeError as exc:
            self._raise_if_session_exhausted(session.session_id, cause=exc)
            identifier = f"session={session.session_id} uid={request.uid} entrypoint={QUERY_ENTRYPOINT}"
            message = f"sandbox invocation failed ({identifier}): {exc}"
            raise SandboxInvocationError(
                message,
                status_code=exc.status_code,
                detail_code=exc.detail_code,
                detail_exception=exc.detail_exception,
                detail_error=exc.detail_error,
            ) from exc
        except Exception as exc:
            self._raise_if_session_exhausted(session.session_id, cause=exc)
            identifier = f"session={session.session_id} uid={request.uid} entrypoint={QUERY_ENTRYPOINT}"
            message = f"sandbox invocation failed ({identifier}): {exc}"
            raise SandboxInvocationError(
                message,
                status_code=0,
                detail_code=None,
                detail_exception=exc.__class__.__name__,
                detail_error=str(exc),
            ) from exc

    def _load_session(self, session_id: UUID) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found")
        if session.status is not SessionStatus.ACTIVE:
            raise RuntimeError(f"session {session_id} is not active")
        return session

    def _validate_session(self, session: Session, request: EntrypointInvocationRequest) -> None:
        if session.uid != request.uid:
            raise PermissionError("session UID does not match invocation UID")
        if not self._tokens.verify(session.session_id, request.token):
            raise PermissionError("invalid session token presented for entrypoint invocation")

    def _raise_if_session_exhausted(
        self,
        session_id: UUID,
        *,
        cause: Exception | None = None,
    ) -> None:
        session = self._load_post_invoke_session(session_id)
        if session.status is not SessionStatus.EXHAUSTED:
            return
        message = f"session {session_id} exhausted during entrypoint invocation"
        if cause is None:
            raise SessionBudgetExhaustedError(message)
        raise SessionBudgetExhaustedError(message) from cause

    def _load_post_invoke_session(self, session_id: UUID) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found after entrypoint invocation")
        return session

__all__ = [
    "EntrypointInvoker",
    "MinerResponseValidationError",
    "SandboxClient",
    "SandboxInvocationError",
]
