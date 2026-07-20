"""Use case for executing sandbox tools."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.ports.session_registry import SessionRegistryPort
from harnyx_commons.application.ports.token_registry import TokenRegistryPort
from harnyx_commons.domain.session import ProviderCredentialSource, Session, SessionFailureCode, SessionStatus
from harnyx_commons.domain.tool_call import (
    SearchToolResult,
    StartedToolCall,
    ToolCall,
    ToolCallOutcome,
    ToolExecutionFacts,
    ToolResult,
    ToolResultPolicy,
)
from harnyx_commons.errors import (
    BudgetExceededError,
    ToolInvocationTimeoutError,
    ToolProviderError,
    ToolProviderFailureCode,
)
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.schema import LlmResponse
from harnyx_commons.tools.dto import (
    ToolBudgetSnapshot,
    ToolInvocationRequest,
    ToolInvocationResult,
    tool_payload_for_invocation,
)
from harnyx_commons.tools.token_semaphore import ToolConcurrencyLimiter
from harnyx_commons.tools.types import (
    EMBEDDING_TOOLS,
    LLM_TOOLS,
    SEARCH_TOOLS,
    SearchToolName,
    ToolName,
    is_citation_source,
    is_search_tool,
)
from harnyx_commons.tools.usage_tracker import ToolCallUsage, UsageTracker


class ToolInvoker(Protocol):
    """Adapter responsible for invoking the actual tool implementation."""

    async def invoke(
        self,
        tool_name: ToolName,
        *,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
        context: ToolInvocationContext | None = None,
    ) -> object:
        """Call the tool and return its response payload."""


class _ToolExecutionPort(Protocol):
    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        """Execute a validated tool invocation."""


tool_logger = logging.getLogger("harnyx_commons.tools")

_TOOLS_WITHOUT_USAGE: set[ToolName] = {
    "test_tool",
    "tooling_info",
}

_PROVIDER_BACKED_TOOL_NAMES: frozenset[ToolName] = frozenset({*LLM_TOOLS, *SEARCH_TOOLS, *EMBEDDING_TOOLS})

_SEARCH_RESULT_FIELDS: dict[SearchToolName, tuple[str, str, str]] = {
    "search_web": ("link", "snippet", "title"),
    "search_ai": ("url", "note", "title"),
    "fetch_page": ("url", "content", "title"),
}


@dataclass(frozen=True)
class _ExecutionResult:
    receipt: ToolCall
    response_payload: JsonObject
    results: tuple[ToolResult, ...]
    llm_tokens: int
    usage_details: ToolCallUsage | None
    budget: ToolBudgetSnapshot


@dataclass(frozen=True, slots=True)
class ToolInvocationOutput:
    """Internal tool result that keeps public payload separate from execution facts."""

    public_payload: JsonObject
    execution: ToolExecutionFacts | None = None
    actual_cost_usd: float | None = None
    actual_cost_provider: str | None = None
    actual_cost_evidence: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """Receipt/session metadata available to tool invoker adapters."""

    receipt_id: str
    session_id: UUID
    active_attempt: int
    uid: int
    miner_hotkey_ss58: str | None = None
    provider_credential_source: ProviderCredentialSource = ProviderCredentialSource.MINER


ToolCallObserver = Callable[[Session, ToolCall], Awaitable[None]]


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
        tool_call_observer: ToolCallObserver | None = None,
    ) -> None:
        self._sessions = session_registry
        self._receipts = receipt_log
        self._usage_tracker = usage_tracker
        self._tool_invoker = tool_invoker
        self._tokens = token_registry
        self._clock = clock
        self._tool_call_observer = tool_call_observer

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        """Execute a tool call on behalf of the supplied session."""
        session = self._load_session(request.session_id)
        log_context = _build_tool_log_context(request, session)
        request_payload = _normalize_payload(
            {
                "args": list(request.args),
                "kwargs": dict(request.kwargs),
            }
        )
        debug_call_id = str(uuid4())
        tool_logger.info("tool call started", extra={**log_context, "event": "tool_call_start"})
        _debug_tool_event(
            "miner_tool_call.started",
            lambda: _tool_call_debug_data(
                call_id=debug_call_id,
                session=session,
                request=request,
                request_payload=_debug_payload(session, request_payload),
                response_payload=None,
                usage=None,
                cost_usd=None,
                budget=None,
                elapsed_ms=None,
                error=None,
            ),
        )

        try:
            result = await self._execute_and_record_async(
                session,
                request,
                debug_call_id=debug_call_id,
                request_payload=request_payload,
            )
        except Exception as exc:
            self._log_failure(log_context, exc)
            _debug_tool_event(
                "miner_tool_call.failed",
                lambda exc=exc: _tool_call_debug_data(
                    call_id=debug_call_id,
                    session=session,
                    request=request,
                    request_payload=_debug_payload(session, request_payload),
                    response_payload=None,
                    usage=None,
                    cost_usd=None,
                    budget=None,
                    elapsed_ms=None,
                    error=_debug_error_for_session(session, exc),
                ),
            )
            raise

        self._log_success(log_context, result)
        return ToolInvocationResult(
            receipt=result.receipt,
            response_payload=result.response_payload,
            budget=result.budget,
            usage=result.usage_details,
        )

    async def _invoke_tool_async(
        self,
        request: ToolInvocationRequest,
    ) -> JsonObject:
        return (await self._invoke_tool_output_async(request, context=None)).public_payload

    async def _invoke_tool_output_async(
        self,
        request: ToolInvocationRequest,
        *,
        context: ToolInvocationContext | None,
    ) -> ToolInvocationOutput:
        return _normalize_invocation_output(
            await self._tool_invoker.invoke(
                request.tool,
                args=request.args,
                kwargs=request.kwargs,
                context=context,
            )
        )

    def _extract_usage(
        self,
        request: ToolInvocationRequest,
        response_payload: object,
        results: tuple[ToolResult, ...],
    ) -> tuple[int, ToolCallUsage | None]:
        name = request.tool
        if name in LLM_TOOLS:
            return _extract_llm_usage(request, response_payload)
        if is_search_tool(name):
            if not isinstance(response_payload, Mapping):
                raise ValueError("search tool response must be a mapping")
            return 0, None
        if name in EMBEDDING_TOOLS:
            if not isinstance(response_payload, Mapping):
                raise ValueError("embedding tool response must be a mapping")
            return 0, None
        if name in _TOOLS_WITHOUT_USAGE:
            return 0, None
        raise LookupError(f"unsupported tool {request.tool!r}")

    def _record_usage(
        self,
        session: Session,
        request: ToolInvocationRequest,
        llm_tokens: int,
        usage_details: ToolCallUsage | None,
        settled_cost_usd: float | None,
        actual_cost_usd: float | None,
        actual_cost_provider: str | None,
    ) -> Session:
        return self._usage_tracker.record_tool_call(
            session,
            tool_name=request.tool,
            llm_tokens=llm_tokens,
            usage=usage_details if usage_details is not None else None,
            cost_usd=settled_cost_usd,
            actual_cost_usd=actual_cost_usd,
            actual_cost_provider=actual_cost_provider,
        )

    async def _execute_and_record_async(
        self,
        session: Session,
        request: ToolInvocationRequest,
        *,
        debug_call_id: str,
        request_payload: JsonValue | None,
    ) -> _ExecutionResult:
        started_at = self._clock()
        self._validate_token(session.session_id, request.token)
        receipt_id = str(uuid4())
        issued_at = self._clock()
        started_call = StartedToolCall(
            receipt_id=receipt_id,
            session_id=session.session_id,
            session_active_attempt=session.active_attempt,
            uid=session.uid,
            tool=request.tool,
            issued_at=issued_at,
            request_payload=request_payload,
            result_policy=_resolve_result_policy(request.tool),
            execution=ToolExecutionFacts(started_at=started_at),
        )
        self._receipts.start_pending_receipt(
            started_call=started_call,
        )
        invocation_context = ToolInvocationContext(
            receipt_id=receipt_id,
            session_id=session.session_id,
            active_attempt=session.active_attempt,
            uid=session.uid,
            miner_hotkey_ss58=session.miner_hotkey_ss58,
            provider_credential_source=session.provider_credential_source,
        )
        try:
            result = await self._execute_pending_receipt_async(
                session,
                request,
                started_call=started_call,
                started_at=started_at,
                debug_call_id=debug_call_id,
                request_payload=request_payload,
                invocation_context=invocation_context,
            )
        except asyncio.CancelledError as exc:
            await self._try_materialize_failed_pending_receipt(
                session=session,
                started_call=started_call,
                started_at=started_at,
                exc=exc,
            )
            raise
        except Exception as exc:
            await self._try_materialize_failed_pending_receipt(
                session=session,
                started_call=started_call,
                started_at=started_at,
                exc=exc,
            )
            raise
        return result

    async def _execute_pending_receipt_async(
        self,
        session: Session,
        request: ToolInvocationRequest,
        *,
        started_call: StartedToolCall,
        started_at: datetime,
        debug_call_id: str,
        request_payload: JsonValue | None,
        invocation_context: ToolInvocationContext,
    ) -> _ExecutionResult:
        invocation_output = await self._invoke_tool_output_async(request, context=invocation_context)
        finished_at = self._clock()
        results, result_policy = self._build_results(request, invocation_output.public_payload)
        llm_tokens, usage_details = self._extract_usage(
            request,
            invocation_output.public_payload,
            results,
        )
        settled_cost_usd = _settled_success_cost(
            request.tool,
            actual_cost_usd=invocation_output.actual_cost_usd,
            actual_cost_provider=invocation_output.actual_cost_provider,
            actual_cost_evidence=invocation_output.actual_cost_evidence,
        )
        cost_unavailable = settled_cost_usd is None and request.tool in _PROVIDER_BACKED_TOOL_NAMES
        receipt = started_call.materialize(
            outcome=ToolCallOutcome.OK,
            response_payload=invocation_output.public_payload,
            results=results,
            result_policy=result_policy,
            cost_usd=settled_cost_usd,
            actual_cost_usd=settled_cost_usd,
            actual_cost_provider=invocation_output.actual_cost_provider,
            extra=(
                {"actual_cost_settlement_source": "unavailable"}
                if cost_unavailable
                else None
            ),
            execution=_merge_execution_facts(
                invocation_output.execution,
                started_at=started_at,
                finished_at=finished_at,
            ),
        )
        completion = self._receipts.complete_pending_receipt(
            receipt,
            settle_usage=lambda: self._settle_usage(
                session_id=session.session_id,
                request=request,
                llm_tokens=llm_tokens,
                usage_details=usage_details,
                settled_cost_usd=settled_cost_usd,
                actual_cost_usd=settled_cost_usd,
                actual_cost_provider=invocation_output.actual_cost_provider,
            ),
        )
        if completion is None:
            raise RuntimeError("tool completion arrived after pending receipt was abandoned")
        updated_session, should_raise_budget_exhausted = completion
        await self._observe_tool_call(updated_session, receipt)
        budget_snapshot = _build_budget_snapshot(updated_session)
        result = _ExecutionResult(
            receipt=receipt,
            response_payload=invocation_output.public_payload,
            results=results,
            llm_tokens=llm_tokens,
            usage_details=usage_details,
            budget=budget_snapshot,
        )
        _debug_tool_event(
            "miner_tool_call.completed",
            lambda: _tool_call_debug_data(
                call_id=debug_call_id,
                session=updated_session,
                request=request,
                request_payload=_debug_payload(session, request_payload),
                response_payload=_debug_payload(session, result.response_payload),
                usage=asdict(result.usage_details) if result.usage_details else None,
                cost_usd=result.receipt.details.cost_usd,
                actual_cost_evidence=invocation_output.actual_cost_evidence,
                budget=asdict(result.budget),
                elapsed_ms=_elapsed_ms_from_receipt(result.receipt),
                error=None,
            ),
        )
        if should_raise_budget_exhausted:
            raise BudgetExceededError(f"session {session.session_id} exhausted during tool accounting")

        return result

    async def _try_materialize_failed_pending_receipt(
        self,
        *,
        session: Session,
        started_call: StartedToolCall,
        started_at: datetime,
        exc: BaseException,
    ) -> None:
        finished_at = self._clock()
        error_extra = _failed_receipt_error_extra(
            exc,
            provider_credential_source=session.provider_credential_source,
        )
        failed_receipt = started_call.materialize(
            outcome=_tool_failure_outcome(exc),
            response_payload=None,
            results=(),
            cost_usd=None,
            extra=error_extra,
            execution=ToolExecutionFacts(
                elapsed_ms=_elapsed_ms_between(started_at, finished_at),
                started_at=started_at,
                finished_at=finished_at,
            ),
        )
        completion = self._receipts.complete_pending_receipt(
            failed_receipt,
            settle_usage=lambda: (session, False),
        )
        if completion is None:
            return
        updated_session, _ = completion
        await self._observe_tool_call(updated_session, failed_receipt)
        if (
            session.provider_credential_source is ProviderCredentialSource.PLATFORM
            and isinstance(exc, ToolProviderError)
            and exc.failure_code
            in {
                ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE,
                ToolProviderFailureCode.AUTHENTICATION_FAILED,
            }
        ):
            failure_code = (
                SessionFailureCode.PROVIDER_CREDENTIAL_UNAVAILABLE
                if exc.failure_code is ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE
                else SessionFailureCode.PROVIDER_AUTHENTICATION_FAILED
            )
            self._sessions.mutate(
                session.session_id,
                lambda current: current.mark_failure_code(failure_code).mark_error(),
            )

    async def _observe_tool_call(self, session: Session, tool_call: ToolCall) -> None:
        if self._tool_call_observer is None:
            return
        try:
            await self._tool_call_observer(session, tool_call)
        except Exception:
            self._sessions.mutate(session.session_id, lambda current: current.mark_error())
            raise

    def _settle_usage(
        self,
        *,
        session_id: UUID,
        request: ToolInvocationRequest,
        llm_tokens: int,
        usage_details: ToolCallUsage | None,
        settled_cost_usd: float | None,
        actual_cost_usd: float | None,
        actual_cost_provider: str | None,
    ) -> tuple[Session, bool]:
        budget_exhausted_by_this_call = False

        def mutate(current: Session) -> Session:
            nonlocal budget_exhausted_by_this_call

            if current.status not in {SessionStatus.ACTIVE, SessionStatus.EXHAUSTED}:
                raise RuntimeError(f"session {session_id} became {current.status.value} during tool accounting")

            was_already_exhausted = current.status is SessionStatus.EXHAUSTED
            session_for_accounting = _session_for_usage_accounting(current)
            updated = self._record_usage(
                session_for_accounting,
                request,
                llm_tokens,
                usage_details,
                settled_cost_usd,
                actual_cost_usd,
                actual_cost_provider,
            )
            exhausted = _mark_session_exhausted_if_needed(updated)
            if exhausted.status is SessionStatus.EXHAUSTED and not was_already_exhausted:
                budget_exhausted_by_this_call = True
            if was_already_exhausted and exhausted.status is not SessionStatus.EXHAUSTED:
                return exhausted.mark_exhausted()
            return exhausted

        updated_session = self._sessions.mutate(session_id, mutate)
        return updated_session, budget_exhausted_by_this_call

    def _log_success(self, log_context: dict[str, object], result: _ExecutionResult) -> None:
        log_fields = {
            **log_context,
            "event": "tool_call_success",
            "receipt_id": result.receipt.receipt_id,
            "llm_tokens": result.llm_tokens,
            "usage": asdict(result.usage_details) if result.usage_details else None,
        }
        if log_context.get("provider_credential_source") == ProviderCredentialSource.PLATFORM.value:
            tool_logger.info("tool call completed", extra=log_fields)
            return

        response_preview = _summarize_value(result.response_payload, limit=500)
        results_preview = _summarize_value(result.results, limit=200)

        tool_logger.info(
            "tool call completed: response_preview=%s results_preview=%s",
            response_preview,
            results_preview,
            extra={
                **log_fields,
                "response_preview": response_preview,
                "results_preview": results_preview,
            },
        )

    def _log_failure(self, log_context: dict[str, object], exc: Exception) -> None:
        platform_credentials = log_context.get("provider_credential_source") == ProviderCredentialSource.PLATFORM.value
        tool_logger.error(
            "tool call failed",
            extra={
                **log_context,
                "event": "tool_call_error",
                "error": "tool execution failed" if platform_credentials else str(exc),
                "error_type": exc.__class__.__name__,
                **_platform_tool_proxy_failure_metadata(exc),
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


def _normalize_invocation_output(value: object) -> ToolInvocationOutput:
    if isinstance(value, ToolInvocationOutput):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("tool invoker must return a JSON object or ToolInvocationOutput")
    public_payload = _normalize_payload(value)
    if not isinstance(public_payload, dict):
        raise ValueError("tool invoker JSON object normalized to a non-object payload")
    return ToolInvocationOutput(public_payload=public_payload)


def _settled_success_cost(
    tool_name: ToolName,
    *,
    actual_cost_usd: float | None,
    actual_cost_provider: str | None,
    actual_cost_evidence: JsonObject | None,
) -> float | None:
    if tool_name not in _PROVIDER_BACKED_TOOL_NAMES:
        return None
    if actual_cost_usd is None:
        if (
            tool_name == "embed_text"
            and actual_cost_provider == "openrouter"
            and actual_cost_evidence is not None
            and actual_cost_evidence.get("settlement_source") == "unavailable"
        ):
            return None
        raise ValueError(f"{tool_name} succeeded without actual_cost_usd")
    if isinstance(actual_cost_usd, bool) or not isinstance(actual_cost_usd, int | float):
        raise ValueError("actual_cost_usd must be numeric")
    if not math.isfinite(actual_cost_usd):
        raise ValueError("actual_cost_usd must be finite")
    if actual_cost_usd < 0.0:
        raise ValueError("actual_cost_usd must be non-negative")
    return actual_cost_usd


def _merge_execution_facts(
    execution: ToolExecutionFacts | None,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> ToolExecutionFacts:
    return ToolExecutionFacts(
        elapsed_ms=None if execution is None else execution.elapsed_ms,
        ttft_ms=None if execution is None else execution.ttft_ms,
        started_at=started_at,
        finished_at=finished_at,
    )


def _elapsed_ms_between(started_at: datetime, finished_at: datetime) -> float:
    return (finished_at - started_at).total_seconds() * 1000.0


def _tool_failure_outcome(
    exc: BaseException,
) -> ToolCallOutcome:
    if isinstance(exc, (asyncio.CancelledError, ToolInvocationTimeoutError)):
        return ToolCallOutcome.TIMEOUT
    if isinstance(exc, ToolProviderError):
        return ToolCallOutcome.PROVIDER_ERROR
    if isinstance(exc, BudgetExceededError):
        return ToolCallOutcome.BUDGET_EXCEEDED
    return ToolCallOutcome.INTERNAL_ERROR


def _failed_receipt_error_extra(
    exc: BaseException,
    *,
    provider_credential_source: ProviderCredentialSource,
) -> dict[str, JsonValue]:
    if provider_credential_source is ProviderCredentialSource.PLATFORM:
        extra = {
            "error_type": exc.__class__.__name__,
            "error_message": "tool execution failed",
        }
        if exc.__cause__ is not None:
            extra["error_cause_type"] = exc.__cause__.__class__.__name__
        extra.update(_platform_tool_proxy_failure_metadata(exc))
        return extra
    source = exc.__cause__ or exc
    error_message = str(source) or source.__class__.__name__
    extra = {
        "error_type": exc.__class__.__name__,
        "error_message": error_message,
    }
    if exc.__cause__ is not None:
        extra["error_cause_type"] = exc.__cause__.__class__.__name__
        extra["error_cause_message"] = error_message
    extra.update(_platform_tool_proxy_failure_metadata(exc))
    return extra


def _platform_tool_proxy_failure_metadata(exc: BaseException) -> dict[str, JsonValue]:
    extra: dict[str, JsonValue] = {}
    if isinstance(exc, ToolProviderError):
        extra["provider_failure_code"] = exc.failure_code.value
        if exc.provider is not None:
            extra["provider"] = exc.provider
        if exc.http_status is not None:
            extra["provider_http_status"] = exc.http_status
    error_code = getattr(exc, "error_code", None)
    if isinstance(error_code, str) and error_code:
        extra["platform_tool_proxy_error_code"] = error_code
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        extra["platform_tool_proxy_status_code"] = str(status_code)
    return extra


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


def _build_budget_snapshot(session: Session) -> ToolBudgetSnapshot:
    used_budget_usd = session.usage.total_cost_usd
    return ToolBudgetSnapshot(
        session_budget_usd=session.budget_usd,
        session_hard_limit_usd=session.effective_hard_limit_usd,
        session_used_budget_usd=used_budget_usd,
        session_remaining_budget_usd=max(session.budget_usd - used_budget_usd, 0.0),
    )


def _mark_session_exhausted_if_needed(session: Session) -> Session:
    if session.usage.total_cost_usd >= session.effective_hard_limit_usd:
        return session.mark_exhausted()
    return session


def _session_for_usage_accounting(
    session: Session,
) -> Session:
    if session.status is not SessionStatus.EXHAUSTED:
        return session
    return replace(session, status=SessionStatus.ACTIVE)


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
    parsed_payload = _parse_search_tool_payload(tool_name, payload)
    results: list[SearchToolResult] = []
    for entry in parsed_payload.entries:
        if entry.url is None:
            continue

        results.append(
            SearchToolResult(
                index=len(results),
                result_id=uuid4().hex,
                url=entry.url,
                note=entry.note,
                title=entry.title,
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


@dataclass(frozen=True, slots=True)
class _SearchToolPayload:
    entries: tuple[_SearchResultPayload, ...]


@dataclass(frozen=True, slots=True)
class _SearchResultPayload:
    url: str | None
    note: str | None
    title: str | None


def _parse_search_tool_payload(tool_name: SearchToolName, payload: object) -> _SearchToolPayload:
    payload_mapping = _mapping_with_string_keys(payload)
    if payload_mapping is None:
        return _SearchToolPayload(entries=())
    data = payload_mapping.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        return _SearchToolPayload(entries=())

    url_key, note_key, title_key = _SEARCH_RESULT_FIELDS[tool_name]
    entries: list[_SearchResultPayload] = []
    for entry in data:
        parsed_entry = _parse_search_result_payload(entry, url_key=url_key, note_key=note_key, title_key=title_key)
        if parsed_entry is not None:
            entries.append(parsed_entry)
    return _SearchToolPayload(entries=tuple(entries))


def _parse_search_result_payload(
    value: object,
    *,
    url_key: str,
    note_key: str,
    title_key: str,
) -> _SearchResultPayload | None:
    entry_mapping = _mapping_with_string_keys(value)
    if entry_mapping is None:
        return None
    return _SearchResultPayload(
        url=_coerce_str(entry_mapping.get(url_key)),
        note=_coerce_str(entry_mapping.get(note_key)),
        title=_coerce_str(entry_mapping.get(title_key)),
    )


def _mapping_with_string_keys(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _require_string_key_mapping(value: object, *, label: str) -> dict[str, object]:
    mapping = _mapping_with_string_keys(value)
    if mapping is None:
        raise ValueError(label)
    return mapping


def _extract_llm_usage(
    request: ToolInvocationRequest,
    payload: Mapping[str, object | None] | Sequence[object] | object,
) -> tuple[int, ToolCallUsage | None]:
    if request.tool not in LLM_TOOLS:
        raise ValueError(f"expected llm tool request, got {request.tool!r}")

    parsed_payload = _parse_llm_usage_payload(payload)
    request_payload = tool_payload_for_invocation(request)
    provider = _extract_llm_provider(request_payload)
    model = _extract_llm_model(request_payload)

    usage_obj = parsed_payload.llm_response.usage
    if usage_obj is None:
        keys = ", ".join(str(key) for key in sorted(parsed_payload.payload_mapping.keys())) or "none"
        raise ValueError(
            f"llm tool response missing 'usage' field (payload keys: {keys})",
        )

    prompt = usage_obj.prompt_tokens
    completion = usage_obj.completion_tokens
    total = usage_obj.total_tokens
    reasoning = usage_obj.reasoning_tokens

    if prompt is None and completion is None and total is None and reasoning is None:
        return 0, ToolCallUsage(
            provider=provider,
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            reasoning_tokens=None,
        )

    resolved_total = _resolve_llm_total_tokens(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
    )
    usage_details = ToolCallUsage(
        provider=provider,
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=resolved_total,
        reasoning_tokens=reasoning,
    )
    return resolved_total, usage_details


def _extract_llm_provider(payload: Mapping[str, JsonValue]) -> str:
    provider = payload.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    raise ValueError("llm tool request must include a 'provider' payload value")


def _resolve_llm_total_tokens(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    reasoning_tokens: int | None,
) -> int:
    if total_tokens is not None:
        return total_tokens
    return (prompt_tokens or 0) + (completion_tokens or 0) + (reasoning_tokens or 0)


@dataclass(frozen=True, slots=True)
class _LlmUsagePayload:
    payload_mapping: dict[str, object]
    llm_response: LlmResponse


def _extract_llm_model(
    request_payload: Mapping[str, JsonValue],
) -> str:
    request_model = request_payload.get("model")
    if isinstance(request_model, str) and request_model.strip():
        return request_model.strip()

    raise ValueError("llm tool request must include a 'model' payload value")


def _parse_llm_usage_payload(payload: object) -> _LlmUsagePayload:
    payload_mapping = _require_string_key_mapping(payload, label="llm tool response must be a mapping")
    llm_response = LlmResponse.from_payload(payload_mapping)
    return _LlmUsagePayload(payload_mapping=payload_mapping, llm_response=llm_response)


SENSITIVE_KEY_SUBSTRINGS = (
    "token",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "password",
)


def _debug_tool_event(event: str, data_factory: Callable[[], dict[str, JsonValue]]) -> None:
    if not tool_logger.isEnabledFor(logging.DEBUG):
        return
    tool_logger.debug(event, extra={"data": data_factory()})


def _debug_error_data(exc: Exception) -> dict[str, str]:
    return {"type": exc.__class__.__name__, "message": str(exc)}


def _debug_payload(session: Session, payload: JsonValue | None) -> JsonValue | None:
    if session.provider_credential_source is ProviderCredentialSource.PLATFORM:
        return None
    return payload


def _debug_error_for_session(session: Session, exc: Exception) -> dict[str, str]:
    if session.provider_credential_source is ProviderCredentialSource.PLATFORM:
        return {"type": exc.__class__.__name__, "message": "tool execution failed"}
    return _debug_error_data(exc)


def _tool_call_debug_data(
    *,
    call_id: str,
    session: Session,
    request: ToolInvocationRequest,
    request_payload: JsonValue | None,
    response_payload: JsonValue | None,
    usage: JsonValue | None,
    cost_usd: float | None,
    actual_cost_evidence: JsonValue | None = None,
    budget: JsonValue | None,
    elapsed_ms: float | None,
    error: JsonValue | None,
) -> dict[str, JsonValue]:
    return {
        "call_id": call_id,
        "session_id": str(session.session_id),
        "task_id": str(session.task_id),
        "uid": session.uid,
        "attempt": session.active_attempt,
        "tool_name": request.tool,
        "request": _normalize_payload(request_payload),
        "response": _normalize_payload(response_payload),
        "usage": usage,
        "cost_usd": cost_usd,
        "actual_cost_evidence": actual_cost_evidence,
        "budget": budget,
        "elapsed_ms": elapsed_ms,
        "error": error,
    }


def _elapsed_ms_from_receipt(receipt: ToolCall) -> float | None:
    execution = receipt.details.execution
    return None if execution is None else execution.elapsed_ms


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in SENSITIVE_KEY_SUBSTRINGS)


def _build_tool_log_context(request: ToolInvocationRequest, session: Session) -> dict[str, object]:
    context: dict[str, object] = {
        "tool_name": request.tool,
        "session_id": str(session.session_id),
        "task_id": str(session.task_id),
        "attempt": session.active_attempt,
        "uid": session.uid,
        "provider_credential_source": session.provider_credential_source.value,
    }
    if session.provider_credential_source is ProviderCredentialSource.MINER:
        context["tool_args"] = _summarize_args(request.args)
        context["tool_kwargs"] = _sanitize_kwargs(request.kwargs)
    return context


def _summarize_args(args: Sequence[object]) -> tuple[str, ...]:
    return tuple(_summarize_value(arg) for arg in args)


def _sanitize_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in kwargs.items():
        if _is_sensitive_key(key):
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


async def execute_tool_with_concurrency_permit(
    executor: _ToolExecutionPort,
    limiter: ToolConcurrencyLimiter,
    invocation: ToolInvocationRequest,
) -> ToolInvocationResult:
    await limiter.acquire_async(invocation)
    try:
        return await executor.execute(invocation)
    finally:
        limiter.release(invocation)


__all__ = ["ToolExecutor", "ToolInvoker", "ToolCallUsage", "execute_tool_with_concurrency_permit"]
