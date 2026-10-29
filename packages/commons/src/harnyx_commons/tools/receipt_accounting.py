"""Canonical accounting projection of recorded invocations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

from harnyx_commons.domain.session import LlmUsageTotals, SessionUsage
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_commons.domain.tool_usage import LlmModelUsageCost, LlmUsageSummary, ToolUsageSummary
from harnyx_commons.tools.provider_billing import (
    RECOVERABLE_INFERENCE_PROVIDERS,
    RECOVERABLE_INFERENCE_TOOLS,
    OpenRouterAccounting,
)
from harnyx_commons.tools.types import ToolName, is_search_tool


def openrouter_accounting(receipt: ToolCall) -> OpenRouterAccounting | None:
    """Decode explicit miner accounting; generic/internal receipts retain their policy."""
    if receipt.tool not in RECOVERABLE_INFERENCE_TOOLS or _receipt_llm_provider(receipt) != "openrouter":
        return None
    raw = (receipt.details.extra or {}).get("actual_cost_evidence")
    if not isinstance(raw, dict):
        return None
    # Error carriers wrap provider evidence; successful and stored receipts do not.
    evidence = raw.get("provider_usage", raw)
    if not isinstance(evidence, dict) or "inference" not in evidence:
        return None
    return OpenRouterAccounting.from_fields(
        actual_cost_usd=receipt.details.actual_cost_usd,
        reference_cost_usd=receipt.details.reference_cost_usd,
        evidence=evidence,
    )


def include_llm_receipt(receipt: ToolCall) -> bool:
    fact = openrouter_accounting(receipt)
    if fact is not None:
        return fact.counts_as_inference
    extra = receipt.details.extra or {}
    if receipt.is_successful():
        return True
    if "provider_failure_accounted" in extra:
        return extra["provider_failure_accounted"] is True
    if "openrouter_failure_accounted" in extra:
        return extra["openrouter_failure_accounted"] is True
    return bool(extra.get("account_provider_failures") and extra.get("provider_dispatched", True))


def recover_inference_usage(submitted: ToolUsageSummary, receipts: Sequence[ToolCall]) -> ToolUsageSummary:
    """Replace durable hosted inference costs under each provider's receipt policy."""
    for provider in RECOVERABLE_INFERENCE_PROVIDERS:
        submitted = _recover_provider_usage(submitted, receipts, provider)
    return submitted


def _recover_provider_usage(
    submitted: ToolUsageSummary, receipts: Sequence[ToolCall], provider: str
) -> ToolUsageSummary:
    """Replace Platform-owned inference costs, preserving embedding and other-provider policies."""
    decisions = tuple(r for r in receipts if r.tool == "decision_query" and _receipt_llm_provider(r) == provider)
    reranks = tuple(r for r in receipts if r.tool == "rerank" and _receipt_llm_provider(r) == provider)
    prior_rerank_reference = submitted.rerank_reference_cost_by_provider.get(provider, 0.0)
    rerank_reference = sum(r.details.cost_usd or 0.0 for r in reranks)
    prior_decision_reference = submitted.decision_reference_cost_by_provider.get(provider, 0.0)
    decision_reference = sum(r.details.cost_usd or 0.0 for r in decisions)
    chat = tuple(r for r in receipts if r.tool == "llm_chat" and _receipt_llm_provider(r) == provider)
    if not chat and not decisions and not reranks:
        return submitted
    canonical = usage_from_receipts((*chat, *decisions, *reranks))
    normalized, canonical_reference = summarize_llm_usage(canonical, chat)
    previous = submitted.llm.providers.get(provider, {})
    previous_reference = sum(model.reference_cost for model in previous.values())
    reference_delta = (
        canonical_reference - previous_reference + decision_reference - prior_decision_reference
        + rerank_reference - prior_rerank_reference
    )
    providers = {provider: dict(models) for provider, models in submitted.llm.providers.items()}
    providers[provider] = normalized.providers.get(provider, {})
    entries = [entry for models in providers.values() for entry in models.values()]
    actual_llm = (
        None
        if any(entry.actual_cost is None for entry in entries)
        else sum(entry.actual_cost or 0.0 for entry in entries)
    )
    llm_cost = sum(entry.reference_cost for entry in entries)
    llm = LlmUsageSummary(
        providers=providers,
        call_count=sum(entry.usage.call_count for entry in entries),
        prompt_tokens=sum(entry.usage.prompt_tokens for entry in entries),
        completion_tokens=sum(entry.usage.completion_tokens for entry in entries),
        total_tokens=sum(entry.usage.total_tokens for entry in entries),
        reasoning_tokens=sum(entry.usage.reasoning_tokens for entry in entries),
        cost=round(llm_cost, 6),
        reference_cost=round(llm_cost, 6),
        actual_cost=actual_llm,
    )
    reference_providers = dict(submitted.reference_cost_by_provider)
    reference_providers[provider] = reference_providers.get(provider, 0.0) + reference_delta
    actual_providers = dict(submitted.actual_cost_by_provider)
    # Only embedding remains outside authoritative inference recovery. Its known
    # reference and actual amounts are equal; preserve the submitted contribution.
    nonchat_known = (
        submitted.reference_cost_by_provider.get(provider, previous_reference)
        - previous_reference
        - prior_decision_reference
        - prior_rerank_reference
    )
    if nonchat_known < -1e-12:
        raise ValueError("Inference provider reference is smaller than its chat contribution")
    actual_providers[provider] = max(nonchat_known, 0.0) + sum(
        r.details.actual_cost_usd or 0.0 for r in (*chat, *decisions, *reranks)
    )
    actual_total = submitted.actual_total_cost_usd
    if actual_total is not None:
        actual_total = (
            None
            if canonical.actual_total_cost_usd is None
            else (
                actual_total - submitted.actual_cost_by_provider.get(provider, 0.0) + actual_providers[provider]
            )
        )
    return replace(
        submitted,
        llm=llm,
        llm_cost=llm_cost,
        reference_total_cost_usd=submitted.reference_total_cost_usd + reference_delta,
        reference_cost_by_provider=reference_providers,
        decision_reference_cost_by_provider={
            **submitted.decision_reference_cost_by_provider,
            provider: decision_reference,
        },
        rerank_reference_cost_by_provider={
            **submitted.rerank_reference_cost_by_provider,
            **(
                {provider: rerank_reference}
                if reranks or provider in submitted.rerank_reference_cost_by_provider else {}
            ),
        },
        actual_total_cost_usd=actual_total,
        actual_cost_by_provider=actual_providers,
    )


def decision_reference_costs(receipts: Sequence[ToolCall]) -> dict[str, float]:
    return _reference_costs(receipts, tool="decision_query")


def rerank_reference_costs(receipts: Sequence[ToolCall]) -> dict[str, float]:
    return _reference_costs(receipts, tool="rerank")


def _reference_costs(receipts: Sequence[ToolCall], *, tool: ToolName) -> dict[str, float]:
    costs: dict[str, float] = {}
    for receipt in receipts:
        if receipt.tool != tool or receipt.details.cost_usd is None:
            continue
        provider = _receipt_request_provider(receipt)
        if provider is not None:
            costs[provider] = costs.get(provider, 0.0) + receipt.details.cost_usd
    return costs


def summarize_llm_usage(budget: SessionUsage, receipts: Sequence[ToolCall]) -> tuple[LlmUsageSummary, float]:
    """Use the same invocation inclusion and identity rules as session accounting."""
    reference: dict[tuple[str, str], float] = {}
    actual: dict[tuple[str, str], float | None] = {}
    actual_total: float | None = 0.0
    for receipt in receipts:
        if receipt.tool != "llm_chat" or not include_llm_receipt(receipt):
            continue
        charge = receipt.details.actual_cost_usd
        actual_total = None if actual_total is None or charge is None else actual_total + charge
        model = _receipt_llm_model(receipt)
        if model is None:
            continue
        key = (_receipt_llm_provider(receipt), model)
        reference[key] = reference.get(key, 0.0) + (receipt.details.cost_usd or 0.0)
        previous = actual.get(key, 0.0)
        actual[key] = None if previous is None or charge is None else previous + charge
    providers = {
        provider: {
            model: LlmModelUsageCost(
                usage=totals,
                cost=reference.get((provider, model), 0.0),
                reference_cost=reference.get((provider, model), 0.0),
                actual_cost=actual.get((provider, model), 0.0),
            )
            for model, totals in models.items()
        }
        for provider, models in budget.llm_usage_totals.items()
    }
    totals = [entry for models in budget.llm_usage_totals.values() for entry in models.values()]
    cost = sum(entry.cost for models in providers.values() for entry in models.values())
    return LlmUsageSummary(
        call_count=sum(entry.call_count for entry in totals),
        prompt_tokens=sum(entry.prompt_tokens for entry in totals),
        completion_tokens=sum(entry.completion_tokens for entry in totals),
        total_tokens=sum(entry.total_tokens for entry in totals),
        reasoning_tokens=sum(entry.reasoning_tokens for entry in totals),
        providers=providers,
        cost=round(cost, 6),
        reference_cost=round(cost, 6),
        actual_cost=actual_total,
    ), cost


def usage_from_receipts(receipts: Sequence[ToolCall], *, initial: SessionUsage | None = None) -> SessionUsage:
    initial = initial or SessionUsage()
    llm_usage_totals = {provider: dict(models) for provider, models in initial.llm_usage_totals.items()}
    cost_by_provider = dict(initial.cost_by_provider)
    actual_cost_by_provider = dict(initial.actual_cost_by_provider)
    total_cost_usd = initial.total_cost_usd
    actual_total_cost_usd = initial.actual_total_cost_usd
    llm_tokens_last_call = initial.llm_tokens_last_call

    for receipt in receipts:
        session_cost = _receipt_session_cost(receipt)
        if session_cost is not None:
            provider = _receipt_session_cost_provider(receipt)
            total_cost_usd += session_cost
            if provider is not None:
                cost_by_provider[provider] = cost_by_provider.get(provider, 0.0) + session_cost
        actual = _receipt_actual_cost(receipt)
        actual_provider = _receipt_session_cost_provider(receipt)
        if actual is not None and actual_provider is not None:
            actual_cost_by_provider[actual_provider] = actual_cost_by_provider.get(actual_provider, 0.0) + actual
        actual_total_cost_usd = _accumulate_receipt_actual_cost(
            actual_total_cost_usd,
            receipt=receipt,
            cost_usd=session_cost,
        )
        if receipt.tool != "llm_chat" or not include_llm_receipt(receipt):
            continue
        model = _receipt_llm_model(receipt)
        usage = _receipt_llm_usage(receipt)
        if model is None:
            continue
        if usage is None:
            usage = _ReceiptLlmUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                reasoning_tokens=None,
            )
        if usage is None:
            continue
        provider = _receipt_llm_provider(receipt)
        provider_totals = llm_usage_totals.setdefault(provider, {})
        existing = provider_totals.get(model, LlmUsageTotals())
        provider_totals[model] = existing.accumulate(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )
        llm_tokens_last_call = usage.total_tokens

    return SessionUsage(
        total_cost_usd=total_cost_usd,
        cost_by_provider=cost_by_provider,
        reference_total_cost_usd=total_cost_usd,
        reference_cost_by_provider=cost_by_provider,
        actual_total_cost_usd=actual_total_cost_usd,
        actual_cost_by_provider=actual_cost_by_provider,
        llm_tokens_last_call=llm_tokens_last_call,
        llm_usage_totals=llm_usage_totals,
    )


@dataclass(frozen=True, slots=True)
class _ReceiptLlmUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None


def _receipt_llm_usage(receipt: ToolCall) -> _ReceiptLlmUsage | None:
    fact = openrouter_accounting(receipt)
    if fact is not None:
        usage = None if fact.llm_usage is None else asdict(fact.llm_usage)
    else:
        response_payload = receipt.details.response_payload
        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _non_negative_int(usage.get("prompt_tokens"))
    completion_tokens = _non_negative_int(usage.get("completion_tokens"))
    total_tokens = _non_negative_int(usage.get("total_tokens"))
    reasoning_tokens = _non_negative_int(usage.get("reasoning_tokens"))
    resolved_total = (
        total_tokens
        if total_tokens is not None
        else (prompt_tokens or 0) + (completion_tokens or 0) + (reasoning_tokens or 0)
    )
    return _ReceiptLlmUsage(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        total_tokens=resolved_total,
        reasoning_tokens=reasoning_tokens,
    )


def _receipt_llm_model(receipt: ToolCall) -> str | None:
    request_payload = receipt.details.request_payload
    if not isinstance(request_payload, dict):
        return None
    direct_model = request_payload.get("model")
    if isinstance(direct_model, str):
        return direct_model
    kwargs = request_payload.get("kwargs")
    if isinstance(kwargs, dict):
        kwargs_model = kwargs.get("model")
        if isinstance(kwargs_model, str):
            return kwargs_model
    args = request_payload.get("args")
    if isinstance(args, list) and args:
        first_arg = args[0]
        if isinstance(first_arg, dict):
            arg_model = first_arg.get("model")
            if isinstance(arg_model, str):
                return arg_model
    return None


def _receipt_llm_provider(receipt: ToolCall) -> str:
    provider = _receipt_request_provider(receipt)
    if provider is None or not provider.strip():
        raise ValueError("llm_chat receipt requires request provider")
    return provider.strip()


def _receipt_actual_cost(receipt: ToolCall) -> float | None:
    return None if receipt.details.actual_cost_usd is None else float(receipt.details.actual_cost_usd)


def _receipt_session_cost(receipt: ToolCall) -> float | None:
    cost_usd = None if receipt.details.cost_usd is None else float(receipt.details.cost_usd)
    return cost_usd


def _accumulate_receipt_actual_cost(
    current: float | None,
    *,
    receipt: ToolCall,
    cost_usd: float | None,
) -> float | None:
    actual = _receipt_actual_cost(receipt)
    if actual is not None:
        return None if current is None else current + actual
    if cost_usd is None:
        if (
            receipt.tool in {"decision_query", "rerank"}
            or (receipt.details.extra or {}).get("reference_cost_unavailable")
        ):
            return None
        if (receipt.details.extra or {}).get("account_provider_failures") and (receipt.details.extra or {}).get(
            "provider_dispatched", True
        ):
            return None
        if (
            receipt.is_successful()
            and receipt.details.extra is not None
            and receipt.details.extra.get("actual_cost_settlement_source") == "unavailable"
        ):
            return None
        return current
    if current is None:
        return None
    return None if receipt.details.actual_cost_provider is not None else current + cost_usd


def _receipt_session_cost_provider(receipt: ToolCall) -> str | None:
    if receipt.tool == "llm_chat":
        _receipt_llm_provider(receipt)
    if receipt.details.actual_cost_provider is not None:
        return receipt.details.actual_cost_provider
    return _receipt_request_cost_provider(receipt)


def _receipt_request_cost_provider(receipt: ToolCall) -> str | None:
    provider = _receipt_request_provider(receipt)
    if receipt.tool in {"decision_query", "rerank"}:
        return provider if provider in {"openrouter", "ai_gateway", "cloudflare", "jev", "reax"} else None
    if provider is not None and provider.strip():
        return provider.strip()
    if receipt.tool == "llm_chat":
        raise ValueError("llm_chat receipt requires request provider")
    return "desearch" if is_search_tool(receipt.tool) else "chutes"


def _receipt_request_provider(receipt: ToolCall) -> str | None:
    request_payload = receipt.details.request_payload
    if isinstance(request_payload, dict):
        direct_provider = request_payload.get("provider")
        if isinstance(direct_provider, str):
            return direct_provider
        kwargs = request_payload.get("kwargs")
        if isinstance(kwargs, dict):
            kwargs_provider = kwargs.get("provider")
            if isinstance(kwargs_provider, str):
                return kwargs_provider
        args = request_payload.get("args")
        if isinstance(args, list) and args:
            first_arg = args[0]
            if isinstance(first_arg, dict):
                arg_provider = first_arg.get("provider")
                if isinstance(arg_provider, str):
                    return arg_provider
    return None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, int) and value >= 0:
        return value
    return None
