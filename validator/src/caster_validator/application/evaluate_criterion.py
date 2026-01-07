"""Use case for orchestrating miner criterion evaluations."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from typing import cast
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.application.ports.session_registry import SessionRegistryPort
from caster_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from caster_commons.domain.tool_call import (
    SearchToolResult,
    ToolCall,
    ToolResultPolicy,
)
from caster_commons.json_types import JsonObject, JsonValue
from caster_commons.llm.pricing import ALLOWED_TOOL_MODELS, parse_tool_model, price_llm, price_search
from caster_commons.llm.schema import LlmUsage
from caster_commons.tools.types import SearchToolName, is_citation_source, is_search_tool
from caster_validator.application.dto.evaluation import (
    EntrypointInvocationRequest,
    EntrypointInvocationResult,
    EvaluationOutcome,
    EvaluationRequest,
    TokenUsageSummary,
)
from caster_validator.application.invoke_entrypoint import EntrypointInvoker
from caster_validator.application.services.evaluation_scoring import EvaluationScore, EvaluationScoringService
from caster_validator.domain.evaluation import MinerAnswer, MinerCitation, MinerEvaluation

logger = logging.getLogger("caster_validator.evaluation")


class _SandboxCitationPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    note: str | None = None
    receipt_id: str = Field(min_length=1)
    result_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("result_id", "result_hash"),
    )


class _SandboxEvaluationPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: int
    justification: str
    citations: tuple[_SandboxCitationPayload, ...] = ()


class UsageSummarizer:
    """Summarizes tool and LLM usage for an evaluation closeout."""

    def summarize(
        self,
        session: Session,
        receipts: Sequence[ToolCall],
    ) -> tuple[TokenUsageSummary, JsonObject]:
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
    ) -> JsonObject:
        search_summary, total_search_cost = self._summarize_search_usage(receipts)
        llm_summary, total_llm_cost = self._summarize_llm_usage(budget)
        return {
            "search_tool": search_summary,
            "search_tool_cost": total_search_cost,
            "llm": llm_summary,
            "llm_cost": total_llm_cost,
        }

    def _summarize_search_usage(
        self,
        receipts: Sequence[ToolCall],
    ) -> tuple[JsonObject, float]:
        total_cost = 0.0
        call_count = 0

        for receipt in receipts:
            if not is_search_tool(receipt.tool):
                continue
            total_cost += self._search_cost(receipt.tool, receipt)
            call_count += 1

        usage_summary: JsonObject = {
            "call_count": call_count,
            "cost": round(total_cost, 6),
        }
        return usage_summary, total_cost

    def _summarize_llm_usage(self, budget: SessionUsage) -> tuple[JsonObject, float]:
        call_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        providers: dict[str, JsonValue] = {}

        for provider, models in budget.llm_usage_totals.items():
            provider_models: dict[str, JsonValue] = {}
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
                provider_models[model] = {
                    "usage": {
                        "prompt_tokens": totals.prompt_tokens,
                        "completion_tokens": totals.completion_tokens,
                        "total_tokens": totals.total_tokens,
                        "call_count": totals.call_count,
                    },
                    "cost": cost,
                }
                call_count += totals.call_count
                prompt_tokens += totals.prompt_tokens or 0
                completion_tokens += totals.completion_tokens or 0
                total_tokens += totals.total_tokens or 0
                total_cost += cost

            if provider_models:
                providers[provider] = provider_models

        llm_summary: JsonObject = {
            "call_count": call_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "providers": providers,
            "cost": round(total_cost, 6),
        }
        return llm_summary, total_cost

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


class EvaluationOrchestrator:
    """Coordinates entrypoint invocation and evaluation persistence."""

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

    async def evaluate(self, request: EvaluationRequest) -> EvaluationOutcome:
        """Execute the evaluation for the provided request."""
        invocation = await self._invoke_entrypoint(request)
        evaluation = self._build_evaluation(request, invocation.result)
        evaluation, dropped_citations = self._hydrate_citations(
            evaluation,
            invocation.tool_receipts,
            request.session_id,
        )
        score = await self._score_evaluation_async(evaluation, invocation, request)
        session = self._require_session(request.session_id)
        usage, total_tool_usage = self._usage.summarize(session, invocation.tool_receipts)
        self._receipts.clear_session(request.session_id)
        return EvaluationOutcome(
            evaluation=evaluation,
            score=score,
            tool_receipts=invocation.tool_receipts,
            usage=usage,
            total_tool_usage=total_tool_usage,
        )

    async def _invoke_entrypoint(self, request: EvaluationRequest) -> EntrypointInvocationResult:
        return await self._invoker.invoke(
            EntrypointInvocationRequest(
                session_id=request.session_id,
                token=request.token,
                uid=request.uid,
                entrypoint=request.entrypoint,
                payload=request.payload,
                context=request.context,
            ),
        )

    async def _score_evaluation_async(
        self,
        evaluation: MinerEvaluation,
        invocation: EntrypointInvocationResult,
        request: EvaluationRequest,
    ) -> EvaluationScore:
        return await self._scoring.score(
            claim_text=request.claim.text,
            evaluation=evaluation,
            reference_answer=request.claim.reference_answer,
            tool_receipts=invocation.tool_receipts,
            session_id=request.session_id,
        )

    def _require_session(self, session_id: UUID) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise LookupError(f"session {session_id} not found during evaluation closeout")
        return session

    def _build_evaluation(
        self,
        request: EvaluationRequest,
        sandbox_result: object,
    ) -> MinerEvaluation:
        payload = _SandboxEvaluationPayload.model_validate(sandbox_result)
        request.claim.rubric.verdict_options.validate(payload.verdict)

        citations = tuple(
            MinerCitation(
                url=citation.url,
                note=citation.note,
                receipt_id=citation.receipt_id,
                result_id=citation.result_id,
            )
            for citation in payload.citations
        )

        miner_answer = MinerAnswer(
            verdict=payload.verdict,
            justification=payload.justification,
            citations=citations,
        )

        return MinerEvaluation(
            evaluation_id=request.evaluation_id,
            session_id=request.session_id,
            uid=request.uid,
            claim_id=request.claim.claim_id,
            rubric=request.claim.rubric,
            miner_answer=miner_answer,
            completed_at=self._clock(),
        )

    def _hydrate_citations(
        self,
        evaluation: MinerEvaluation,
        receipts: Sequence[ToolCall],
        session_id: UUID,
    ) -> tuple[MinerEvaluation, tuple[str, ...]]:
        if not evaluation.miner_answer.citations:
            return evaluation, ()

        receipt_index = {
            receipt.receipt_id: receipt
            for receipt in receipts
            if receipt.session_id == session_id
        }

        canonical: list[MinerCitation] = []
        invalid: list[str] = []

        for citation in evaluation.miner_answer.citations:
            receipt = receipt_index.get(citation.receipt_id)
            if receipt is None:
                invalid.append(citation.receipt_id)
                continue

            if not is_citation_source(receipt.tool):
                invalid.append(citation.receipt_id)
                continue

            if receipt.metadata.result_policy is not ToolResultPolicy.REFERENCEABLE:
                invalid.append(citation.receipt_id)
                continue

            result = next(
                (res for res in receipt.metadata.results if res.result_id == citation.result_id),
                None,
            )
            if result is None:
                invalid.append(citation.receipt_id)
                continue

            search_result = cast(SearchToolResult, result)
            canonical.append(
                MinerCitation(
                    url=search_result.url,
                    note=search_result.note,
                    receipt_id=citation.receipt_id,
                    result_id=citation.result_id,
                ),
            )

        updated_answer = replace(evaluation.miner_answer, citations=tuple(canonical))
        if invalid:
            logger.warning(
                "dropping invalid citations from miner submission",
                extra={
                    "session_id": str(session_id),
                    "invalid_receipt_ids": tuple(invalid),
                    "dropped_count": len(invalid),
                },
            )
        return replace(evaluation, miner_answer=updated_answer), tuple(invalid)

__all__ = ["EvaluationOrchestrator", "UsageSummarizer"]
