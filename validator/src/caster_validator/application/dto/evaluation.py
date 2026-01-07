"""DTOs for claim evaluation use cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from caster_commons.domain.claim import EvaluationClaim
from caster_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from caster_commons.domain.tool_call import ToolCall
from caster_commons.json_types import JsonObject, JsonValue
from caster_validator.application.services.evaluation_scoring import EvaluationScore
from caster_validator.domain.evaluation import MinerEvaluation


@dataclass(frozen=True)
class TokenUsageSummary:
    """Aggregated LLM usage totals grouped by provider and model."""

    by_provider: dict[str, dict[str, LlmUsageTotals]]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    call_count: int

    @classmethod
    def empty(cls) -> TokenUsageSummary:
        return cls(
            by_provider={},
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
            call_count=0,
        )

    @classmethod
    def from_totals(
        cls,
        totals: Mapping[str, Mapping[str, LlmUsageTotals]],
    ) -> TokenUsageSummary:
        if not totals:
            raise ValueError("llm usage totals missing")

        prompt = 0
        completion = 0
        total = 0
        calls = 0

        providers: dict[str, dict[str, LlmUsageTotals]] = {}
        for provider, models in totals.items():
            if not models:
                continue

            provider_models: dict[str, LlmUsageTotals] = {}
            for model, usage in models.items():
                provider_models[model] = usage
                prompt += usage.prompt_tokens
                completion += usage.completion_tokens
                total += usage.total_tokens
                calls += usage.call_count

            if provider_models:
                providers[provider] = provider_models

        if not providers:
            raise ValueError("llm usage totals missing")

        return cls(
            by_provider=providers,
            total_prompt_tokens=prompt,
            total_completion_tokens=completion,
            total_tokens=total,
            call_count=calls,
        )

    @classmethod
    def from_usage(cls, usage: SessionUsage) -> TokenUsageSummary:
        if not usage.llm_usage_totals:
            return cls.empty()
        totals = usage.require_usage_totals()
        return cls.from_totals(totals)

    def merge(self, other: TokenUsageSummary) -> TokenUsageSummary:
        providers: dict[str, dict[str, LlmUsageTotals]] = {}

        def _merge_source(summary: TokenUsageSummary) -> None:
            for provider, models in summary.by_provider.items():
                provider_models = providers.setdefault(provider, {})
                for model, usage in models.items():
                    existing = provider_models.get(model)
                    if existing is None:
                        provider_models[model] = usage
                    else:
                        provider_models[model] = LlmUsageTotals(
                            prompt_tokens=existing.prompt_tokens + usage.prompt_tokens,
                            completion_tokens=existing.completion_tokens + usage.completion_tokens,
                            total_tokens=existing.total_tokens + usage.total_tokens,
                            call_count=existing.call_count + usage.call_count,
                        )

        _merge_source(self)
        _merge_source(other)

        return TokenUsageSummary(
            by_provider=providers,
            total_prompt_tokens=self.total_prompt_tokens + other.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens + other.total_completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            call_count=self.call_count + other.call_count,
        )


@dataclass(frozen=True)
class ScriptArtifactSpec:
    """Script artifact metadata supplied by the platform."""

    uid: int
    digest: str
    size_bytes: int
    artifact_id: str | None = None


@dataclass(frozen=True)
class EvaluationBatchSpec:
    """Evaluation batch supplied by the platform."""

    run_id: UUID
    entrypoint: str
    cutoff_at_iso: str
    created_at_iso: str
    uids: tuple[int, ...]
    claims: tuple[EvaluationClaim, ...]
    artifacts: tuple[ScriptArtifactSpec, ...]


@dataclass(frozen=True)
class EntrypointInvocationRequest:
    """Input payload for invoking a miner entrypoint."""

    session_id: UUID
    token: str
    uid: int
    entrypoint: str
    payload: Mapping[str, JsonValue]
    context: Mapping[str, JsonValue]


@dataclass(frozen=True)
class EntrypointInvocationResult:
    """Response returned by the sandbox entrypoint."""

    result: Mapping[str, JsonValue]
    tool_receipts: Sequence[ToolCall]


@dataclass(frozen=True)
class EvaluationOutcome:
    """Aggregate outcome of running a claim evaluation."""

    evaluation: MinerEvaluation
    score: EvaluationScore
    tool_receipts: Sequence[ToolCall]
    usage: TokenUsageSummary
    total_tool_usage: JsonObject | None = None  # filled by orchestrator


@dataclass(frozen=True)
class ScoredEvaluation:
    """Miner evaluation paired with its computed score."""

    evaluation: MinerEvaluation
    score: EvaluationScore
    usage: TokenUsageSummary
    total_tool_usage: JsonObject | None = None


@dataclass(frozen=True)
class EvaluationRequest:
    """Input payload for orchestrating a full claim evaluation."""

    session_id: UUID
    token: str
    uid: int
    entrypoint: str
    payload: Mapping[str, JsonValue]
    context: Mapping[str, JsonValue]
    claim: EvaluationClaim
    evaluation_id: UUID


@dataclass(frozen=True)
class EvaluationBatchResult:
    """Outcome of running a batch of claim evaluations."""

    run_id: UUID
    claims: Sequence[EvaluationClaim]
    evaluations: Sequence[ScoredEvaluation]
    uids: Sequence[int]


@dataclass(frozen=True)
class EvaluationCloseout:
    """Payload persisted when an evaluation completes."""

    run_id: UUID
    validator_uid: int
    outcome: EvaluationOutcome
    session: Session


__all__ = [
    "TokenUsageSummary",
    "EntrypointInvocationRequest",
    "EntrypointInvocationResult",
    "EvaluationOutcome",
    "EvaluationScore",
    "EvaluationCloseout",
    "EvaluationRequest",
    "EvaluationBatchResult",
    "ScoredEvaluation",
    "ScriptArtifactSpec",
    "EvaluationBatchSpec",
]
