"""Reuse author research and finish a grounded reference within the existing stage."""

from __future__ import annotations

import asyncio
import json
from time import monotonic

import httpx
from openai import AsyncOpenAI

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest
from harnyx_commons.observability.langfuse import start_llm_generation, update_generation_best_effort
from harnyx_commons.tools.parallel import ParallelClient

from .agent_runner import ACTIVE_CAPTURE, RETRY_POLICY, CallCapture, exposed, retry_request
from .contracts import (
    BatchTerminalGenerationError,
    CandidateSession,
    CandidateStageError,
    QuestionDraft,
    ReviewDecision,
    TerminalReference,
)
from .prompts import REFERENCE_SYSTEM
from .reference_evidence import EvidenceStore, ReferenceSources, reference_has_factual_correction, render_reference
from .reference_runner import REFERENCE_MODEL, ReferenceAgentRunner
from .source_fetch import MAX_RESPONSE_BYTES, PublicSourceFetcher

REFERENCE_TIMEOUT_SECONDS = 600.0


def build_reference_parallel() -> ParallelClient:
    """Validate reference-only settings and create the runner-owned provider."""
    try:
        settings = LlmSettings()
        key = settings.parallel_api_key.get_secret_value()
        if not key.strip():
            raise ValueError("Parallel API key is required for reference sources")
        return ParallelClient(
            base_url=settings.parallel_base_url,
            api_key=key,
            max_concurrent=settings.parallel_max_concurrent,
            max_response_bytes=MAX_RESPONSE_BYTES,
            retry_policy=RETRY_POLICY,
            include_payloads_in_logs=False,
        )
    except ValueError as exc:
        raise BatchTerminalGenerationError(
            "provider_configuration", "Invalid reference provider configuration", stage="reference"
        ) from exc


async def derive_reference(
    draft: QuestionDraft,
    verification: ReviewDecision | None,
    session: CandidateSession,
    deadline: float,
    *,
    client: AsyncOpenAI,
    slots: asyncio.Semaphore,
    fetcher: PublicSourceFetcher,
    parallel: ParallelClient,
    capture_full_requests: bool = False,
) -> TerminalReference:
    capture = CallCapture(
        session, "reference", REFERENCE_MODEL, "openai", physical_accounting=True, retain_requests=capture_full_requests
    )

    async def call() -> TerminalReference:
        attempt_deadline = min(deadline, monotonic() + REFERENCE_TIMEOUT_SECONDS)
        async with asyncio.timeout_at(attempt_deadline):
            try:
                prior_usage = capture.tool_usage
                prior_calls, prior_receipts = capture.attempted, capture.usage_records
                prior_available_cost = capture.available_cost_usd
                prior_model_cost, prior_priced_receipts = capture.available_model_cost_usd, capture.priced_usage_records
                store = EvidenceStore()
                sources = ReferenceSources(fetcher, parallel, store, capture)
                runner = ReferenceAgentRunner(
                    client,
                    slots,
                    validate_output=lambda output: render_reference(output, store, draft),
                )
                inputs = json.dumps(
                    {
                        "work_order": (
                            "Write the reference for this fixed question using quoted research "
                            "and acquired source evidence."
                        ),
                        "current_draft": draft.model_dump(mode="json"),
                        "quoted_author_research": exposed(session.author_history),
                        "source_support": [source.model_dump(mode="json") for source in session.source_support],
                        "verification": verification.model_dump(by_alias=True) if verification else None,
                    },
                    ensure_ascii=False,
                )
                request = LlmRequest(
                    provider="openai",
                    model=REFERENCE_MODEL,
                    temperature=None,
                    max_output_tokens=None,
                    reasoning_effort="high",
                    use_case="miner_task_generation.reference",
                    messages=(
                        LlmMessage(role="system", content=(LlmMessageContentPart.input_text(REFERENCE_SYSTEM),)),
                        LlmMessage(role="user", content=(LlmMessageContentPart.input_text(inputs),)),
                    ),
                )
                with start_llm_generation(provider_label="openai", request=request) as generation:
                    try:
                        output = await runner.run(inputs, sources, capture)
                        try:
                            answer = render_reference(output, store, draft)
                        except (ValueError, RecursionError) as exc:
                            raise CandidateStageError("contract_invalid", "reference", str(exc)) from exc
                        update_generation_best_effort(generation, output=output.model_dump(mode="json"))
                    finally:
                        usage = capture.tool_usage.llm
                        calls = capture.attempted - prior_calls
                        receipts = capture.usage_records - prior_receipts
                        missing = max(0, calls - receipts)
                        model_cost = (
                            capture.available_model_cost_usd - prior_model_cost
                            if calls == capture.priced_usage_records - prior_priced_receipts
                            else None
                        )
                        update_generation_best_effort(
                            generation,
                            usage_details={
                                "input": usage.prompt_tokens - prior_usage.llm.prompt_tokens,
                                "output": usage.completion_tokens - prior_usage.llm.completion_tokens,
                                "total": usage.total_tokens - prior_usage.llm.total_tokens,
                                "reasoning": usage.reasoning_tokens - prior_usage.llm.reasoning_tokens,
                            },
                            cost_details={"total": model_cost} if model_cost is not None else None,
                            metadata={
                                "source_states": sources.counts,
                                "attempted_calls": calls,
                                "usage_records": receipts,
                                "missing_usage_calls": missing,
                                "model_actual_cost_usd": model_cost,
                                "available_cost_usd": capture.available_cost_usd - prior_available_cost,
                            },
                        )
                terminal = TerminalReference(
                    reference_answer=answer,
                    factual_correction=reference_has_factual_correction(output, draft),
                )
            except ToolProviderError as exc:
                if exc.http_status in {401, 403, 404}:
                    raise BatchTerminalGenerationError(
                        "provider_auth" if exc.http_status != 404 else "provider_configuration",
                        "Parallel reference provider authentication/configuration failed",
                        stage="reference",
                    ) from exc
                if (
                    exc.http_status == 429
                    or (exc.http_status is not None and exc.http_status >= 500)
                    or isinstance(exc.__cause__, httpx.TransportError)
                ):
                    raise CandidateStageError(
                        "transient_provider", "reference", "Parallel request exhausted retries"
                    ) from exc
                raise
            if monotonic() >= attempt_deadline:
                raise TimeoutError("Reference deadline reached during finalization")
            return terminal

    token = ACTIVE_CAPTURE.set(capture)
    try:
        return await retry_request(call, capture, deadline, solver=False)
    finally:
        ACTIVE_CAPTURE.reset(token)
        capture.finish()
