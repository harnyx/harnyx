from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from caster_miner_sdk.api import LlmChatResult, llm_chat, search_web
from caster_miner_sdk.criterion_evaluation import (
    CriterionEvaluationRequest,
    CriterionEvaluationResponse,
    CriterionEvaluationVerdict,
)
from caster_miner_sdk.decorators import entrypoint
from caster_miner_sdk.llm import LlmMessageContentPart
from caster_miner_sdk.verdict import VerdictOptions

MAX_EVIDENCE_RESULTS = 3
CHUTES_MODEL = "openai/gpt-oss-120b"
CHUTES_TEMPERATURE = 0.2


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    url: str
    note: str | None
    title: str | None
    result_id: str


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    receipt_id: str
    items: tuple[EvidenceItem, ...]


@dataclass(frozen=True, slots=True)
class LlmVerdict:
    verdict: int
    justification: str


@entrypoint("evaluate_criterion")
async def evaluate_criterion(request: object) -> CriterionEvaluationResponse:
    payload = CriterionEvaluationRequest.model_validate(request)
    claim_text = payload.claim_text
    rubric_title = payload.rubric_title
    rubric_description = payload.rubric_description
    verdict_options = payload.verdict_options_domain()

    evidence = await _gather_evidence(claim_text)
    verdict = await _adjudicate_criterion(
        claim_text=claim_text,
        rubric_title=rubric_title,
        rubric_description=rubric_description,
        evidence=evidence.items,
        verdict_options=verdict_options,
    )

    citations = [_build_citation(evidence.receipt_id, item) for item in evidence.items]

    return {
        "verdict": verdict.verdict,
        "justification": verdict.justification,
        "citations": citations,
    }


async def _gather_evidence(query: str) -> EvidenceBundle:
    response = await search_web(query, num=max(MAX_EVIDENCE_RESULTS, 5))
    items: list[EvidenceItem] = []
    for result in response.results:
        if result.url is None:
            raise RuntimeError("search_web result missing url")
        items.append(
            EvidenceItem(
                url=result.url,
                note=result.note,
                title=result.title,
                result_id=result.result_id,
            )
        )
        if len(items) >= MAX_EVIDENCE_RESULTS:
            break
    if not items:
        raise RuntimeError("search_web returned no evidence results")
    return EvidenceBundle(receipt_id=response.receipt_id, items=tuple(items))


async def _adjudicate_criterion(
    *,
    claim_text: str,
    rubric_title: str,
    rubric_description: str,
    evidence: Sequence[EvidenceItem],
    verdict_options: VerdictOptions,
) -> LlmVerdict:
    messages = _build_llm_messages(
        claim_text=claim_text,
        rubric_title=rubric_title,
        rubric_description=rubric_description,
        evidence=evidence,
        verdict_options=verdict_options,
    )
    payload = await llm_chat(
        messages=messages,
        model=CHUTES_MODEL,
        temperature=CHUTES_TEMPERATURE,
    )
    content = _extract_assistant_content(payload)
    answer = CriterionEvaluationVerdict.from_llm_content(content)
    verdict_value = verdict_options.validate(answer.verdict)

    return LlmVerdict(verdict=verdict_value, justification=answer.justification)


def _build_llm_messages(
    *,
    claim_text: str,
    rubric_title: str,
    rubric_description: str,
    evidence: Sequence[EvidenceItem],
    verdict_options: VerdictOptions,
) -> list[dict[str, str]]:
    evidence_lines = []
    for index, item in enumerate(evidence, start=1):
        summary = item.note or item.title or "No snippet provided"
        evidence_lines.append(f"{index}. {summary} (source: {item.url})")
    evidence_block = "\n".join(evidence_lines)
    verdict_instructions = _verdict_instructions(verdict_options)

    user_content = (
        "Assess the claim using the provided evidence.\n"
        f"Claim: {claim_text}\n"
        f"Rubric Title: {rubric_title}\n"
        f"Rubric Description: {rubric_description}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"{verdict_instructions}\n"
        "Return compact JSON of the form {\"verdict\": <integer>, "
        "\"justification\": \"...\"}. Justification must cite the evidence indices used."
    )

    return [
        {
            "role": "system",
            "content": (
                "You evaluate factual claims. Cite evidence indices and keep answers concise."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _extract_assistant_content(payload: LlmChatResult) -> str:
    choices = payload.llm.choices
    if not choices:
        raise RuntimeError("chutes response missing choices")
    message = choices[0].message
    content = _join_content_parts(message.content).strip()
    if not content:
        raise RuntimeError("chutes response missing assistant content")
    return content


def _join_content_parts(parts: Sequence[LlmMessageContentPart]) -> str:
    fragments: list[str] = []
    for part in parts:
        text = (part.text or "").strip()
        if text:
            fragments.append(text)
    return "\n".join(fragments)


def _verdict_instructions(options: VerdictOptions) -> str:
    allowed = _allowed_values(options)
    allowed_text = _format_allowed_values(allowed)
    lines = [f'Set "verdict" to {allowed_text}.']
    lines.extend(f"- Use {option.value} for {option.description}." for option in options.options)
    return "\n".join(lines)


def _allowed_values(options: VerdictOptions) -> tuple[int, ...]:
    return tuple(entry.value for entry in options.options)


def _format_allowed_values(values: Sequence[int]) -> str:
    text_values = [str(value) for value in values]
    if len(text_values) == 1:
        return text_values[0]
    if len(text_values) == 2:
        return " or ".join(text_values)
    return ", ".join(text_values[:-1]) + f", or {text_values[-1]}"


def _build_citation(receipt_id: str, item: EvidenceItem) -> dict[str, str]:
    return {
        "receipt_id": receipt_id,
        "result_id": item.result_id,
    }
