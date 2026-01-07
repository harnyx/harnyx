"""Scoring helpers for miner evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.domain.claim import ReferenceAnswer
from caster_commons.domain.tool_call import ToolCall
from caster_commons.llm.grading import JustificationGrader
from caster_validator.domain.evaluation import MinerEvaluation


@dataclass(frozen=True)
class EvaluationScore:
    """Structured score emitted for a miner evaluation."""

    verdict_score: float
    support_score: float
    justification_pass: bool
    failed_citation_ids: tuple[str, ...]
    grader_rationale: str | None = None

    @property
    def total(self) -> float:
        """Return the aggregate score used for weight updates."""
        return self.verdict_score + self.support_score


class EvaluationScoringService:
    """Evaluates miner responses against the curated reference answer."""

    _VERDICT_WEIGHT = 0.5
    _SUPPORT_WEIGHT = 0.5

    def __init__(self, receipt_log: ReceiptLogPort, grader: JustificationGrader) -> None:
        self._receipts = receipt_log
        self._grader = grader

    async def score(
        self,
        *,
        claim_text: str,
        evaluation: MinerEvaluation,
        reference_answer: ReferenceAnswer,
        tool_receipts: Sequence[ToolCall],
        session_id: UUID,
    ) -> EvaluationScore:
        """Return a score describing how well the miner supported the reference answer."""

        # Gate 1: verdict must match reference; otherwise zero out early.
        verdict_matches = evaluation.miner_answer.verdict == reference_answer.verdict
        if not verdict_matches:
            return EvaluationScore(
                verdict_score=0.0,
                support_score=0.0,
                justification_pass=False,
                failed_citation_ids=(),
                grader_rationale="verdict diverges from reference answer",
            )

        grade = await self._grader.grade(
            claim_text=claim_text,
            reference_verdict=reference_answer.verdict,
            reference_justification=reference_answer.justification,
            miner_verdict=evaluation.miner_answer.verdict,
            miner_justification=evaluation.miner_answer.justification,
            verdict_options=evaluation.rubric.verdict_options,
            miner_citations=tuple(
                citation.note
                or citation.url
                or citation.receipt_id
                for citation in evaluation.miner_answer.citations
            ),
        )

        support_pass = bool(grade.support_ok)
        return EvaluationScore(
            verdict_score=self._VERDICT_WEIGHT,
            support_score=self._SUPPORT_WEIGHT if support_pass else 0.0,
            justification_pass=support_pass,
            failed_citation_ids=(),
            grader_rationale=grade.rationale,
        )


__all__ = ["EvaluationScoringService", "EvaluationScore"]
