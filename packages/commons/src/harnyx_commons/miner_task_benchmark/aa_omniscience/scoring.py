"""Original AA four-class grading with the approved GA replacement judge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest
from harnyx_commons.miner_task_benchmark.aa_omniscience.loader import load_aa_omniscience_grader_prompt


class OmniscienceGrade(StrEnum):
    CORRECT = "A"
    INCORRECT = "B"
    PARTIAL_ANSWER = "C"
    NOT_ATTEMPTED = "D"


@dataclass(frozen=True, slots=True)
class OmniscienceDecision:
    grade: OmniscienceGrade

    @property
    def is_correct(self) -> bool:
        return self.grade is OmniscienceGrade.CORRECT


class OmniscienceScoringService:
    def __init__(self, provider: LlmProviderPort) -> None:
        self._provider = provider

    async def score(
        self, *, question: str, reference_answer: str, generated_answer: str
    ) -> OmniscienceDecision:
        prompt = load_aa_omniscience_grader_prompt().format(
            question=question, target=reference_answer, predicted_answer=generated_answer
        )
        response = await self._provider.invoke(
            LlmRequest(
                provider="openrouter",
                model="google/gemini-2.5-flash",
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text(prompt),)),),
                temperature=None,
                max_output_tokens=None,
                extra={"reasoning": {"enabled": True}, "provider": {"require_parameters": True}},
                timeout=300.0,
                allow_postprocess_recovery=False,
                use_case="aa_omniscience_original_prompt_judge",
            )
        )
        if len(response.choices) != 1 or response.finish_reason != "stop":
            raise ValueError(f"AA judge did not finish one answer: {response!r}")
        try:
            grade = OmniscienceGrade((response.raw_text or "").strip())
        except ValueError as exc:
            raise ValueError(f"AA judge returned an invalid grade: {response.raw_text!r}") from exc
        return OmniscienceDecision(grade=grade)
