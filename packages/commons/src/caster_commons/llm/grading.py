"""LLM-based justification grader helpers (structured output)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from caster_commons.domain.verdict import VerdictOptions
from caster_commons.llm.json_utils import pydantic_postprocessor
from caster_commons.llm.provider import LlmProviderPort
from caster_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmUsage

_SYSTEM_PROMPT = (
    "You are a strict grader. Given a claim, a reference justification, and a miner "
    "justification, determine if the miner's reasoning aligns with the reference - meaning "
    "it identifies similar key facts and reaches consistent conclusions without contradicting "
    "the reference reasoning. Use only the provided text; do not invent facts or use external tools."
)


class JustificationGrade(BaseModel):
    rationale: str
    support_ok: bool


@dataclass(frozen=True)
class JustificationGraderConfig:
    provider: str
    model: str
    temperature: float | None = None
    max_output_tokens: int | None = 1024
    reasoning_effort: str | None = None


class JustificationGrader:
    """Tiny wrapper that asks an LLM to judge justification quality."""

    def __init__(self, provider: LlmProviderPort, config: JustificationGraderConfig) -> None:
        self._provider = provider
        self._config = config

    async def grade(
        self,
        *,
        claim_text: str,
        reference_verdict: int,
        reference_justification: str,
        miner_verdict: int,
        miner_justification: str,
        verdict_options: VerdictOptions,
        miner_citations: Sequence[str] | None = None,
    ) -> JustificationGrade:
        grade, _ = await self.grade_with_usage(
            claim_text=claim_text,
            reference_verdict=reference_verdict,
            reference_justification=reference_justification,
            miner_verdict=miner_verdict,
            miner_justification=miner_justification,
            verdict_options=verdict_options,
            miner_citations=miner_citations,
        )
        return grade

    async def grade_with_usage(
        self,
        *,
        claim_text: str,
        reference_verdict: int,
        reference_justification: str,
        miner_verdict: int,
        miner_justification: str,
        verdict_options: VerdictOptions,
        miner_citations: Sequence[str] | None = None,
    ) -> tuple[JustificationGrade, LlmUsage]:
        citations_block = _format_citations(miner_citations)
        user_content = (
            f"Claim: {claim_text}\n\n"
            f"Reference verdict: {verdict_options.description_for(reference_verdict)}\n"
            f"Reference justification: {reference_justification}\n\n"
            f"Miner verdict: {verdict_options.description_for(miner_verdict)}\n"
            f"Miner justification: {miner_justification}\n\n"
            f"Miner citations:\n{citations_block}\n\n"
            "Reply with JSON only."
        )

        request = LlmRequest(
            provider=self._config.provider,
            model=self._config.model,
            messages=(
                LlmMessage(
                    role="system",
                    content=(LlmMessageContentPart.input_text(_SYSTEM_PROMPT),),
                ),
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text(user_content),),
                ),
            ),
            output_mode="structured",
            output_schema=JustificationGrade,
            postprocessor=pydantic_postprocessor(JustificationGrade),
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            reasoning_effort=self._config.reasoning_effort,
        )

        response = await self._provider.invoke(request)
        parsed = response.postprocessed
        if parsed is None:
            raise RuntimeError("LLM grader did not return structured output")
        return JustificationGrade.model_validate(parsed), response.usage

    async def aclose(self) -> None:
        await self._provider.aclose()


def _format_citations(citations: Sequence[str] | None) -> str:
    if not citations:
        return "None"
    lines = [f"- {entry}" for entry in citations if entry]
    return "\n".join(lines) if lines else "None"


__all__ = [
    "JustificationGrade",
    "JustificationGrader",
    "JustificationGraderConfig",
]
