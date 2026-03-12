"""Use case for orchestrating a generic miner task run."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from uuid import UUID

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.application.ports.session_registry import SessionRegistryPort
from caster_commons.domain.miner_task import EvaluationDetails
from caster_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from caster_commons.domain.tool_call import ToolCall
from caster_commons.domain.tool_usage import (
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from caster_commons.llm.pricing import ALLOWED_TOOL_MODELS, parse_tool_model, price_llm, price_search
from caster_commons.llm.schema import LlmUsage
from caster_commons.tools.types import SearchToolName, is_search_tool
from caster_validator.application.dto.evaluation import (
    EntrypointInvocationRequest,
    MinerTaskRunRequest,
    TaskRunOutcome,
    TokenUsageSummary,
)
from caster_validator.application.invoke_entrypoint import EntrypointInvoker
from caster_validator.application.services.evaluation_scoring import EvaluationScoringService
from caster_validator.domain.evaluation import MinerTaskRun

logger = logging.getLogger("caster_validator.task_run")


def _elapsed_ms(*, issued_at: datetime, completed_at: datetime) -> float:
    return (completed_at - issued_at).total_seconds() * 1000.0


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
        llm_summary, total_llm_cost = self._summarize_llm_usage(budget)
        return ToolUsageSummary(
            search_tool=search_summary,
            search_tool_cost=total_search_cost,
            llm=llm_summary,
            llm_cost=total_llm_cost,
        )

    def _summarize_search_usage(
        self,
        receipts: Sequence[ToolCall],
    ) -> tuple[SearchToolUsageSummary, float]:
        total_cost = 0.0
        call_count = 0

        for receipt in receipts:
            if not is_search_tool(receipt.tool):
                continue
            total_cost += self._search_cost(receipt.tool, receipt)
            call_count += 1

        return (
            SearchToolUsageSummary(call_count=call_count, cost=round(total_cost, 6)),
            total_cost,
        )

    def _summarize_llm_usage(self, budget: SessionUsage) -> tuple[LlmUsageSummary, float]:
        call_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        providers: dict[str, dict[str, LlmModelUsageCost]] = {}

        for _, models in budget.llm_usage_totals.items():
            for model, totals in models.items():
                try:
                    tool_model = parse_tool_model(model)
                except ValueError:
                    continue
                if tool_model not in ALLOWED_TOOL_MODELS:
                    continue
                cost = price_llm(
                    tool_model,
                    self._llm_usage_totals_to_usage(totals),
                )
                model_provider = tool_model.split("/", 1)[0]
                provider_models = providers.setdefault(model_provider, {})
                provider_models[str(tool_model)] = LlmModelUsageCost(usage=totals, cost=cost)
                call_count += totals.call_count
                prompt_tokens += totals.prompt_tokens
                completion_tokens += totals.completion_tokens
                total_tokens += totals.total_tokens
                total_cost += cost

        return (
            LlmUsageSummary(
                call_count=call_count,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                providers=providers,
                cost=round(total_cost, 6),
            ),
            total_cost,
        )

    @staticmethod
    def _search_cost(tool: SearchToolName, receipt: ToolCall) -> float:
        if receipt.metadata.cost_usd is not None:
            return float(receipt.metadata.cost_usd)
        return float(price_search(tool))

    @staticmethod
    def _llm_usage_totals_to_usage(totals: LlmUsageTotals) -> LlmUsage:
        return LlmUsage(
            prompt_tokens=totals.prompt_tokens or 0,
            completion_tokens=totals.completion_tokens or 0,
            total_tokens=totals.total_tokens or 0,
        )


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
        invocation = await self._invoker.invoke(
            EntrypointInvocationRequest(
                session_id=request.session_id,
                token=request.token,
                uid=request.uid,
                query=request.task.query,
            ),
        )
        invocation_completed_at = self._clock()
        score_breakdown = await self._scoring.score(
            task=request.task,
            response=invocation.response,
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
