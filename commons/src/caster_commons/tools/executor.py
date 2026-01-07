"""Use case for executing sandbox tools."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.application.ports.session_registry import SessionRegistryPort
from caster_commons.application.ports.token_registry import TokenRegistryPort
from caster_commons.domain.session import Session, SessionStatus
from caster_commons.domain.tool_call import (
    ReceiptMetadata,
    SearchToolResult,
    ToolCall,
    ToolCallOutcome,
    ToolResult,
    ToolResultPolicy,
)
from caster_commons.json_types import JsonObject, JsonValue
from caster_commons.llm.pricing import ToolModelName, parse_tool_model, price_llm, price_search
from caster_commons.llm.schema import LlmResponse
from caster_commons.tools.dto import ToolBudgetSnapshot, ToolInvocationRequest, ToolInvocationResult
from caster_commons.tools.types import LLM_TOOLS, SearchToolName, ToolName, is_citation_source, is_search_tool
from caster_commons.tools.usage_tracker import ToolCallUsage, UsageTracker


class ToolInvoker(Protocol):
    """Adapter responsible for invoking the actual tool implementation."""

    async def invoke(
        self,
        tool_name: ToolName,
        *,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> JsonObject:
        """Call the tool and return its response payload."""


tool_logger = logging.getLogger("caster_commons.tools")

_TOOLS_WITHOUT_USAGE: set[ToolName] = {"test_tool"}

_SEARCH_RESULT_FIELDS: dict[SearchToolName, tuple[str, str, str]] = {
    "search_web": ("link", "snippet", "title"),
    "search_x": ("url", "text", "title"),
}


@dataclass(frozen=True)
class _ExecutionResult:
    receipt: ToolCall
    response_payload: JsonObject
    results: tuple[ToolResult, ...]
    llm_tokens: int
    usage_details: ToolCallUsage | None
    budget: ToolBudgetSnapshot


class ToolExecutor:
    """Coordinates budget enforcement and receipt recording for tool calls."""

    def __init__(
        self,
        session_registry: SessionRegistryPort,
        receipt_log: ReceiptLogPort,
        usage_tracker: UsageTracker,
        tool_invoker: ToolInvoker,
        *,
        token_registry: TokenRegistryPort,
        clock: Callable[[], datetime],
    ) -> None:
        self._sessions = session_registry
        self._receipts = receipt_log
        self._usage_tracker = usage_tracker
        self._tool_invoker = tool_invoker
        self._tokens = token_registry
        self._clock = clock

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        """Execute a tool call on behalf of the supplied session."""
        session = self._load_session(request.session_id)
        log_context = _build_tool_log_context(request, session)
        tool_logger.info("tool call started", extra={**log_context, "event": "tool_call_start"})

        try:
            result = await self._execute_and_record_async(session, request)
        except Exception as exc:
            self._log_failure(log_context, exc)
            raise

        self._log_success(log_context, result)
        return ToolInvocationResult(
            receipt=result.receipt,
            response_payload=result.response_payload,
            budget=result.budget,
            usage=result.usage_details,
        )

    async def _invoke_tool_async(self, request: ToolInvocationRequest) -> JsonObject:
        return await self._tool_invoker.invoke(
            request.tool,
            args=request.args,
            kwargs=request.kwargs,
        )

    def _extract_usage(
        self,
        request: ToolInvocationRequest,
        response_payload: object,
    ) -> tuple[int, ToolCallUsage | None, float | None]:
        name = request.tool
        if name in LLM_TOOLS:
            return _extract_llm_usage(request, response_payload)
        if is_search_tool(name):
            if not isinstance(response_payload, Mapping):
                raise ValueError("search tool response must be a mapping")
            return 0, None, price_search(name)
        if name in _TOOLS_WITHOUT_USAGE:
            return 0, None, None
        raise LookupError(f"unsupported tool {request.tool!r}")

    def _record_usage(
        self,
        session: Session,
        request: ToolInvocationRequest,
        llm_tokens: int,
        usage_details: ToolCallUsage | None,
        call_cost: float | None,
    ) -> Session:
        return self._usage_tracker.record_tool_call(
            session,
            tool_name=request.tool,
            llm_tokens=llm_tokens,
            usage=usage_details if usage_details is not None else None,
            cost_usd=call_cost,
        )

    async def _execute_and_record_async(
        self,
        session: Session,
        request: ToolInvocationRequest,
    ) -> _ExecutionResult:
        self._validate_token(session.session_id, request.token)

        response_payload = await self._invoke_tool_async(request)
        llm_tokens, usage_details, call_cost = self._extract_usage(request, response_payload)
        updated_session = self._record_usage(session, request, llm_tokens, usage_details, call_cost)
        budget_limit = self._usage_tracker.limit_usd
        budget_snapshot = ToolBudgetSnapshot(
            session_budget_usd=budget_limit,
            session_used_budget_usd=updated_session.usage.total_cost_usd,
            session_remaining_budget_usd=budget_limit - updated_session.usage.total_cost_usd,
        )
        results, result_policy = self._build_results(request, response_payload)
        receipt = self._build_receipt(
            request,
            updated_session,
            response_payload,
            results,
            result_policy,
            cost_usd=call_cost,
        )
        self._receipts.record(receipt)
        self._sessions.update(updated_session)

        return _ExecutionResult(
            receipt=receipt,
            response_payload=response_payload,
            results=results,
            llm_tokens=llm_tokens,
            usage_details=usage_details,
            budget=budget_snapshot,
        )

    def _log_success(self, log_context: dict[str, object], result: _ExecutionResult) -> None:
        response_preview = _summarize_value(result.response_payload, limit=500)
        results_preview = _summarize_value(result.results, limit=200)

        tool_logger.info(
            "tool call completed: response_preview=%s results_preview=%s",
            response_preview,
            results_preview,
            extra={
                **log_context,
                "event": "tool_call_success",
                "receipt_id": result.receipt.receipt_id,
                "llm_tokens": result.llm_tokens,
                "usage": asdict(result.usage_details) if result.usage_details else None,
                "response_preview": response_preview,
                "results_preview": results_preview,
            },
        )

    def _log_failure(self, log_context: dict[str, object], exc: Exception) -> None:
        tool_logger.exception(
            "tool call failed",
            extra={
                **log_context,
                "event": "tool_call_error",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            },
        )

    def _build_results(
        self,
        request: ToolInvocationRequest,
        response_payload: object,
    ) -> tuple[tuple[ToolResult, ...], ToolResultPolicy]:
        result_policy = _resolve_result_policy(request.tool)
        results = _build_tool_results(request.tool, response_payload, result_policy)
        return results, result_policy

    def _load_session(self, session_id: UUID) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found")
        if session.status is not SessionStatus.ACTIVE:
            raise RuntimeError(f"session {session_id} is not active")
        now = self._clock()
        if now > session.expires_at:
            raise RuntimeError(
                f"session {session_id} expired at {session.expires_at.isoformat()}",
            )
        return session

    def _validate_token(self, session_id: UUID, presented: str) -> None:
        if not self._tokens.verify(session_id, presented):
            raise PermissionError("invalid session token presented for tool execution")

    def _build_receipt(
        self,
        request: ToolInvocationRequest,
        session: Session,
        response_payload: object,
        results: tuple[ToolResult, ...],
        result_policy: ToolResultPolicy,
        cost_usd: float | None = None,
    ) -> ToolCall:
        issued_at = self._clock()
        normalized_response: JsonValue | None = _normalize_payload(response_payload)
        extra: dict[str, str] = {"issued_at": issued_at.isoformat()}
        if cost_usd is not None:
            extra["cost_usd"] = f"{cost_usd:.6f}"
        return ToolCall(
            receipt_id=str(uuid4()),
            session_id=session.session_id,
            uid=session.uid,
            tool=request.tool,
            issued_at=issued_at,
            outcome=ToolCallOutcome.OK,
            metadata=ReceiptMetadata(
                request_hash=_hash_payload(
                    {
                        "args": list(request.args),
                        "kwargs": dict(request.kwargs),
                    },
                ),
                response_hash=_hash_payload(response_payload),
                response_payload=normalized_response,
                results=results,
                result_policy=result_policy,
                cost_usd=cost_usd,
                extra=extra,
            ),
        )


def _normalize_payload(value: object) -> JsonValue | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalize_payload(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_payload(item) for item in value]
    return str(value)


def _hash_payload(payload: object) -> str:
    import json
    from hashlib import sha256

    normalized = _normalize_payload(payload)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(serialized).hexdigest()


def _resolve_result_policy(tool_name: ToolName) -> ToolResultPolicy:
    if is_citation_source(tool_name):
        return ToolResultPolicy.REFERENCEABLE
    return ToolResultPolicy.LOG_ONLY


def _build_tool_results(
    tool_name: ToolName,
    payload: object,
    policy: ToolResultPolicy,
) -> tuple[ToolResult, ...]:
    if policy is ToolResultPolicy.REFERENCEABLE:
        if not is_search_tool(tool_name):
            raise ValueError(f"REFERENCEABLE result policy not supported for tool {tool_name!r}")
        return _build_search_results(tool_name, payload)
    return _build_log_only_results(payload)


def _build_search_results(tool_name: SearchToolName, payload: object) -> tuple[ToolResult, ...]:
    if not isinstance(payload, Mapping):
        return ()

    data = payload.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        return ()
    results: list[SearchToolResult] = []

    for entry in data:
        if not isinstance(entry, Mapping):
            continue

        url_key, note_key, title_key = _SEARCH_RESULT_FIELDS[tool_name]
        url = _coerce_str(entry.get(url_key))
        note = _coerce_str(entry.get(note_key))
        title = _coerce_str(entry.get(title_key))

        if not url:
            continue

        results.append(
            SearchToolResult(
                index=len(results),
                result_id=uuid4().hex,
                url=url,
                note=note,
                title=title,
            ),
        )

    return tuple(results)


def _build_log_only_results(payload: object) -> tuple[ToolResult, ...]:
    normalized = _normalize_payload(payload)
    return (
        ToolResult(
            index=0,
            result_id=uuid4().hex,
            raw=normalized,
        ),
    )


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _extract_llm_usage(
    request: ToolInvocationRequest,
    payload: Mapping[str, object | None] | Sequence[object] | object,
) -> tuple[int, ToolCallUsage | None, float | None]:
    if request.tool not in LLM_TOOLS:
        raise ValueError(f"expected llm tool request, got {request.tool!r}")

    if not isinstance(payload, Mapping):
        raise ValueError("llm tool response must be a mapping")

    provider = "chutes"  # billing reference provider

    model_raw = request.kwargs.get("model")
    if not isinstance(model_raw, str) or not model_raw:
        raise ValueError("llm tool request must include a 'model' kwarg")
    model: ToolModelName = parse_tool_model(model_raw)

    llm_response = LlmResponse.from_payload(payload)
    usage_obj = llm_response.usage
    if usage_obj is None:
        keys = ", ".join(str(key) for key in sorted(payload.keys())) or "none"
        raise ValueError(
            "llm tool response missing 'usage' field "
            f"(payload keys: {keys})",
        )

    prompt = usage_obj.prompt_tokens
    completion = usage_obj.completion_tokens
    total = usage_obj.total_tokens

    if prompt is None and completion is None and total is None:
        return 0, None, None

    usage_details = ToolCallUsage(
        provider=provider,
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )
    resolved_total = total if total is not None else (prompt or 0) + (completion or 0)
    call_cost = price_llm(model, usage_obj)
    return resolved_total, usage_details, call_cost


SENSITIVE_KEY_SUBSTRINGS = (
    "token",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "password",
)


def _build_tool_log_context(request: ToolInvocationRequest, session: Session) -> dict[str, object]:
    return {
        "tool_name": request.tool,
        "session_id": str(session.session_id),
        "uid": session.uid,
        "tool_args": _summarize_args(request.args),
        "tool_kwargs": _sanitize_kwargs(request.kwargs),
    }


def _summarize_args(args: Sequence[object]) -> tuple[str, ...]:
    return tuple(_summarize_value(arg) for arg in args)


def _sanitize_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in kwargs.items():
        lowered = key.lower()
        if any(fragment in lowered for fragment in SENSITIVE_KEY_SUBSTRINGS):
            sanitized[key] = "<redacted>"
        else:
            sanitized[key] = _summarize_value(value)
    return sanitized


def _summarize_value(value: object, *, limit: int = 200) -> str:
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - repr should rarely fail
        text = f"<unrepresentable {type(value).__name__}>"
    return text if len(text) <= limit else text[:limit] + "…"


__all__ = ["ToolExecutor", "ToolInvoker", "ToolCallUsage"]
