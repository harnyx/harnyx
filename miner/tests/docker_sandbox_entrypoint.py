from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from harnyx_miner_sdk.api import LlmChatResult, llm_chat, search_web
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.llm import LlmMessageContentPart
from harnyx_miner_sdk.query import Query, Response

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

@entrypoint("query")
async def query(query: Query) -> Response:
    evidence = await _gather_evidence(query.text)
    answer = await _answer_query(
        query_text=query.text,
        evidence=evidence.items,
    )
    citations = ", ".join(_citation_summary(item) for item in evidence.items)
    return Response(text=f"{answer}\n\nSources: {citations}")


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


async def _answer_query(
    *,
    query_text: str,
    evidence: Sequence[EvidenceItem],
) -> str:
    messages = _build_llm_messages(
        query_text=query_text,
        evidence=evidence,
    )
    payload = await llm_chat(
        messages=messages,
        model=CHUTES_MODEL,
        temperature=CHUTES_TEMPERATURE,
    )
    content = _extract_assistant_content(payload)
    return content


def _build_llm_messages(
    *,
    query_text: str,
    evidence: Sequence[EvidenceItem],
) -> list[dict[str, str]]:
    evidence_lines = []
    for index, item in enumerate(evidence, start=1):
        summary = item.note or item.title or "No snippet provided"
        evidence_lines.append(f"{index}. {summary} (source: {item.url})")
    evidence_block = "\n".join(evidence_lines)

    user_content = (
        "Answer the user's query using the provided evidence.\n"
        f"Query: {query_text}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Give a compact plain-text answer and mention the evidence indices you used."
    )

    return [
        {
            "role": "system",
            "content": (
                "You answer queries from evidence. Cite evidence indices and keep answers concise."
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


def _citation_summary(item: EvidenceItem) -> str:
    summary = item.note or item.title or item.url
    return f"{summary} ({item.url})"
