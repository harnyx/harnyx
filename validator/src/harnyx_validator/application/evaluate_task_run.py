"""Use case for orchestrating a generic miner task run."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.ports.session_registry import SessionRegistryPort
from harnyx_commons.domain.miner_task import EvaluationDetails, MinerTaskErrorCode
from harnyx_commons.domain.session import Session, SessionUsage
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_commons.domain.tool_usage import (
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from harnyx_commons.llm.pricing import price_search
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.miner_task_scoring import EvaluationScoringService
from harnyx_commons.tools.types import SearchToolName, is_search_tool
from harnyx_validator.application.dto.evaluation import (
    EntrypointInvocationRequest,
    MinerTaskRunRequest,
    TaskRunOutcome,
    TokenUsageSummary,
)
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker
from harnyx_validator.domain.evaluation import MinerTaskRun

logger = logging.getLogger("harnyx_validator.task_run")
measurement_logger = logging.getLogger("harnyx_validator.measurement")


def _elapsed_ms(*, issued_at: datetime, completed_at: datetime) -> float:
    return (completed_at - issued_at).total_seconds() * 1000.0


def _monotonic_elapsed_ms(*, started_at: float, completed_at: float) -> float:
    return round((completed_at - started_at) * 1000.0, 3)


def _log_scoring_finished(
    *,
    batch_id: UUID,
    session_id: UUID,
    artifact_id: UUID,
    task_id: UUID,
    uid: int,
    invocation_ms: float,
    scoring_ms: float,
    orchestration_ms: float,
    comparison_score: float | None,
    total_score: float | None,
    outcome: str,
    error_code: str | None,
) -> None:
    measurement_logger.info(
        "miner-task scoring finished",
        extra={
            "data": {
                "batch_id": str(batch_id),
                "session_id": str(session_id),
                "artifact_id": str(artifact_id),
                "task_id": str(task_id),
                "uid": uid,
                "invocation_ms": invocation_ms,
                "scoring_ms": scoring_ms,
                "orchestration_ms": orchestration_ms,
                "comparison_score": comparison_score,
                "total_score": total_score,
                "outcome": outcome,
                "error_code": error_code,
            }
        },
    )


def _scoring_error_code(exc: Exception) -> str:
    if isinstance(exc, LlmRetryExhaustedError):
        return str(MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED)
    return str(MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE)


class UsageSummarizer:
    """Summarizes tool and LLM usage for a miner task run."""

    def summarize(
        self,
        session: Session,
        receipts: Sequence[ToolCall],
    ) -> tuple[TokenUsageSummary, ToolUsageSummary]:
        usage = TokenUsageSummary.from_usage(session.usage)
        total_tool_usage = self._summarize_tool_usage(
            receipts=receipts,
            budget=session.usage,
        )
        return usage, total_tool_usage

    def _summarize_tool_usage(
        self,
        *,
        receipts: Sequence[ToolCall],
        budget: SessionUsage,
    ) -> ToolUsageSummary:
        search_summary, total_search_cost = self._summarize_search_usage(receipts)
        llm_summary, total_llm_cost = self._summarize_llm_usage(budget, receipts)
        return ToolUsageSummary(
            search_tool=search_summary,
            search_tool_cost=total_search_cost,
            llm=llm_summary,
            llm_cost=total_llm_cost,
            reference_total_cost_usd=budget.reference_total_cost_usd,
            reference_cost_by_provider=dict(budget.reference_cost_by_provider),
            actual_total_cost_usd=budget.actual_total_cost_usd,
            actual_cost_by_provider=dict(budget.actual_cost_by_provider),
        )

    def _summarize_search_usage(
        self,
        receipts: Sequence[ToolCall],
    ) -> tuple[SearchToolUsageSummary, float]:
        total_cost = 0.0
        actual_cost_total: float | None = 0.0
        call_count = 0

        for receipt in receipts:
            if not is_search_tool(receipt.tool):
                continue
            if not receipt.is_successful():
                continue
            cost = self._search_cost(receipt.tool, receipt)
            total_cost += cost
            actual_cost_total = _accumulate_actual_summary_cost(
                actual_cost_total,
                cost_usd=cost,
            )
            call_count += 1

        return (
            SearchToolUsageSummary(
                call_count=call_count,
                cost=round(total_cost, 6),
                reference_cost=round(total_cost, 6),
                actual_cost=actual_cost_total,
            ),
            total_cost,
        )

    def _summarize_llm_usage(
        self,
        budget: SessionUsage,
        receipts: Sequence[ToolCall],
    ) -> tuple[LlmUsageSummary, float]:
        call_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0
        total_cost = 0.0
        providers: dict[str, dict[str, LlmModelUsageCost]] = {}
        settled_cost_by_provider_model = _settled_llm_cost_by_provider_model(receipts)

        for provider, models in budget.llm_usage_totals.items():
            for model, totals in models.items():
                cost = settled_cost_by_provider_model.get((provider, model), 0.0)
                provider_models = providers.setdefault(provider, {})
                provider_models[model] = LlmModelUsageCost(
                    usage=totals,
                    cost=cost,
                    reference_cost=cost,
                    actual_cost=cost,
                )
                call_count += totals.call_count
                prompt_tokens += totals.prompt_tokens
                completion_tokens += totals.completion_tokens
                total_tokens += totals.total_tokens
                reasoning_tokens += totals.reasoning_tokens
                total_cost += cost

        return (
            LlmUsageSummary(
                call_count=call_count,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
                providers=providers,
                cost=round(total_cost, 6),
                reference_cost=round(total_cost, 6),
                actual_cost=_actual_llm_total_cost(receipts),
            ),
            total_cost,
        )

    @staticmethod
    def _search_cost(tool: SearchToolName, receipt: ToolCall) -> float:
        cost = (
            float(receipt.details.cost_usd)
            if receipt.details.cost_usd is not None
            else float(price_search(tool, referenceable_results=len(receipt.details.results)))
        )
        actual_cost = _actual_receipt_cost(receipt)
        if actual_cost is not None and actual_cost != cost:
            raise ValueError(f"{receipt.tool} receipt cost_usd and actual_cost_usd must match")
        return cost


def _actual_receipt_cost(receipt: ToolCall) -> float | None:
    return None if receipt.details.actual_cost_usd is None else float(receipt.details.actual_cost_usd)


def _receipt_cost(receipt: ToolCall) -> float:
    if receipt.details.cost_usd is None:
        raise ValueError(f"{receipt.tool} receipt missing cost_usd")
    cost = float(receipt.details.cost_usd)
    actual_cost = _actual_receipt_cost(receipt)
    if actual_cost is not None and actual_cost != cost:
        raise ValueError(f"{receipt.tool} receipt cost_usd and actual_cost_usd must match")
    return cost


def _accumulate_actual_summary_cost(
    current: float | None,
    *,
    cost_usd: float,
) -> float | None:
    if current is None:
        return None
    return current + cost_usd


def _actual_llm_total_cost(receipts: Sequence[ToolCall]) -> float | None:
    total: float | None = 0.0
    for receipt in receipts:
        if receipt.tool != "llm_chat":
            continue
        if not receipt.is_successful():
            continue
        total = _accumulate_actual_summary_cost(
            total,
            cost_usd=_receipt_cost(receipt),
        )
    return total


def _settled_llm_cost_by_provider_model(receipts: Sequence[ToolCall]) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = {}
    for receipt in receipts:
        if receipt.tool != "llm_chat":
            continue
        if not receipt.is_successful():
            continue
        model = _receipt_model(receipt)
        if model is None:
            continue
        provider = _receipt_provider(receipt)
        key = (provider, model)
        totals[key] = totals.get(key, 0.0) + _receipt_cost(receipt)
    return totals


def _receipt_model(receipt: ToolCall) -> str | None:
    payload = receipt.details.request_payload
    if not isinstance(payload, dict):
        return None
    kwargs = payload.get("kwargs")
    if isinstance(kwargs, dict):
        model = kwargs.get("model")
        return model if isinstance(model, str) else None
    args = payload.get("args")
    if isinstance(args, list) and args and isinstance(args[0], dict):
        model = args[0].get("model")
        return model if isinstance(model, str) else None
    return None


def _receipt_provider(receipt: ToolCall) -> str:
    payload = receipt.details.request_payload
    if isinstance(payload, dict):
        kwargs = payload.get("kwargs")
        if isinstance(kwargs, dict):
            provider = kwargs.get("provider")
            if isinstance(provider, str) and provider.strip():
                return provider.strip()
        args = payload.get("args")
        if isinstance(args, list) and args and isinstance(args[0], dict):
            provider = args[0].get("provider")
            if isinstance(provider, str) and provider.strip():
                return provider.strip()
    raise ValueError("llm_chat receipt requires request provider")


class TaskRunOrchestrator:
    """Coordinates entrypoint invocation and scoring for one miner task."""

    def __init__(
        self,
        entrypoint_invoker: EntrypointInvoker,
        receipt_log: ReceiptLogPort,
        scoring_service: EvaluationScoringService,
        session_registry: SessionRegistryPort,
        *,
        clock: Callable[[], datetime],
        usage_summarizer: UsageSummarizer | None = None,
    ) -> None:
        self._invoker = entrypoint_invoker
        self._receipts = receipt_log
        self._scoring = scoring_service
        self._sessions = session_registry
        self._clock = clock
        self._usage = usage_summarizer or UsageSummarizer()

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        orchestration_started_at = time.monotonic()
        invocation_started_at = orchestration_started_at
        invocation = await self._invoker.invoke(
            EntrypointInvocationRequest(
                session_id=request.session_id,
                token=request.token,
                uid=request.uid,
                query=request.task.query,
            ),
        )
        invocation_ms = _monotonic_elapsed_ms(
            started_at=invocation_started_at,
            completed_at=time.monotonic(),
        )
        invocation_completed_at = self._clock()
        scoring_started_at = time.monotonic()
        try:
            score_breakdown = await self._scoring.score(
                task=request.task,
                response=invocation.response,
            )
        except Exception as exc:
            _log_scoring_finished(
                batch_id=request.batch_id,
                session_id=request.session_id,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                uid=request.uid,
                invocation_ms=invocation_ms,
                scoring_ms=_monotonic_elapsed_ms(
                    started_at=scoring_started_at,
                    completed_at=time.monotonic(),
                ),
                orchestration_ms=_monotonic_elapsed_ms(
                    started_at=orchestration_started_at,
                    completed_at=time.monotonic(),
                ),
                comparison_score=None,
                total_score=None,
                outcome="error",
                error_code=_scoring_error_code(exc),
            )
            raise
        scoring_ms = _monotonic_elapsed_ms(
            started_at=scoring_started_at,
            completed_at=time.monotonic(),
        )
        session = self._require_session(request.session_id)
        completed_at = self._clock()
        usage, total_tool_usage = self._usage.summarize(session, invocation.tool_receipts)
        details = EvaluationDetails(
            score_breakdown=score_breakdown,
            total_tool_usage=total_tool_usage,
            elapsed_ms=_elapsed_ms(issued_at=session.issued_at, completed_at=invocation_completed_at),
        )
        run = MinerTaskRun(
            session_id=request.session_id,
            uid=request.uid,
            artifact_id=request.artifact_id,
            task_id=request.task.task_id,
            response=invocation.response,
            details=details,
            completed_at=completed_at,
        )
        _log_scoring_finished(
            batch_id=request.batch_id,
            session_id=request.session_id,
            artifact_id=request.artifact_id,
            task_id=request.task.task_id,
            uid=request.uid,
            invocation_ms=invocation_ms,
            scoring_ms=scoring_ms,
            orchestration_ms=_monotonic_elapsed_ms(
                started_at=orchestration_started_at,
                completed_at=time.monotonic(),
            ),
            comparison_score=score_breakdown.comparison_score,
            total_score=score_breakdown.total_score,
            outcome="ok",
            error_code=None,
        )
        self._receipts.clear_session(request.session_id)
        return TaskRunOutcome(
            run=run,
            tool_receipts=invocation.tool_receipts,
            usage=usage,
        )

    def _require_session(self, session_id: UUID) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found while summarizing miner task usage")
        return session


__all__ = ["TaskRunOrchestrator", "UsageSummarizer"]
