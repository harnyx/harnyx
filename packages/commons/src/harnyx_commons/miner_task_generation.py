"""Generic miner-task dataset generation."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from harnyx_commons.domain.miner_task import AnswerCitation, MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.json_types import JsonObject
from harnyx_commons.llm.json_utils import pydantic_postprocessor
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_types import VERTEX_PROVIDER, LlmProviderName
from harnyx_commons.llm.schema import (
    GroundedLlmRequest,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
    PostprocessResult,
)

logger = logging.getLogger("harnyx_commons.miner_task_generation")

GENERATED_OUTPUT_SCHEMA_MAX_DEPTH = 8
GENERATED_OUTPUT_SCHEMA_MAX_NODES = 128
GENERATED_OUTPUT_SCHEMA_MAX_LEAVES = 64
_GENERATED_OUTPUT_SCHEMA_PROPERTY_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_GENERATED_OUTPUT_SCHEMA_ANNOTATIONS = frozenset({"description", "title"})
_GENERATED_OUTPUT_ARRAY_ASSERTIONS = frozenset({"maxItems", "minItems"})
_GENERATED_OUTPUT_STRING_ASSERTIONS = frozenset({"maxLength", "minLength"})
_GENERATED_OUTPUT_VALUE_ASSERTIONS = frozenset(
    {"const", "enum", "exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum", "multipleOf"}
)
_GENERATED_OUTPUT_PRIMITIVE_SCHEMA_TYPES = frozenset({"string", "integer", "number", "boolean"})

_TASK_GENERATION_SYSTEM_PROMPT = (
    "You generate evaluation tasks for a generic query-answering system.\n"
    "Your job is to produce standalone user queries whose correct answers require "
    "externally retrieved information, not memorized facts, textbook knowledge, or "
    "broad background knowledge alone.\n"
    "A valid query is a natural user question with a definite answer already "
    "documented in public sources as of the timestamp.\n"
    "A valid query must require at least one independent-source synthesis move. "
    "Examples include comparing separately sourced entity values; reconciling "
    "period and basis differences; resolving source disagreement; combining "
    "source-backed subclaims that no single evidence item can answer completely.\n"
    "Each query must be answerable in plain text, self-contained, and usable without extra attachments.\n"
    "Each query must be specific enough that a grounded answer can be checked against real public sources.\n"
    "Prefer queries about topics that are likely to have publicly accessible, recently "
    "published sources such as official announcements, public filings, major news "
    "coverage, results pages, rankings pages, or company or institution updates.\n"
    "Do not use rubric, verdict, grading, citation-receipt, or JSON-output language inside the query itself.\n"
    "\n"
    "Valid query patterns include:\n"
    "- current or recently changed facts about named entities\n"
    "- exact figures, dates, rankings, vote results, release details, leadership "
    "changes, policy changes, or status changes\n"
    "- comparisons that require current information about more than one entity\n"
    "- synthesis questions that require combining several current facts from independent sources\n"
    "- questions that require checking period, basis, jurisdiction, methodology, "
    "and official-versus-secondary-source disagreement\n"
    "\n"
    "Invalid query patterns include:\n"
    "- plot summaries of literature, films, mythology, or other classic works\n"
    "- textbook explanations of science, math, history, economics, or other school-style topics\n"
    "- generic definitions or broad overviews\n"
    "- broad historical summaries\n"
    "- timeless compare-and-contrast questions\n"
    "- generic health, diet, or wellness explainers\n"
    "- vague prompts like 'what are the latest developments in AI' or 'what is happening in the world of X'\n"
    "- 'most recent' questions whose answer is an old, widely remembered event, appointment, or result\n"
    "- invented sequel, version, update, regulation, mission, award, or report "
    "names unless the named thing clearly exists in public sources as of the "
    "timestamp\n"
    "- questions whose main answer is likely to be 'this has not been "
    "released', 'this did not happen', or 'no such thing exists'\n"
    "- batches that create difficulty mainly through false premises instead of answerable retrieval\n"
    "- questions about facts that change so slowly that a strong model would answer them acceptably from memory\n"
    "- questions whose answers change within hours or days, such as live prices, live "
    "scores, weather, or in-progress events, because the reference answer would go "
    "stale too quickly\n"
    "- any question whose correct answer can be produced acceptably without retrieval\n"
    "- simple single-entity lookups where one page answers every field, such as "
    "approval date plus ingredient, launch date plus vehicle, CEO plus previous "
    "employer, award winner plus listed directors, and match winner plus final score\n"
    "- questions that mostly ask a model to copy the top search result instead of "
    "reconcile evidence\n"
    "\n"
    "Bad query examples:\n"
    '- "Can you summarize the plot of Hamlet?" because it is classic knowledge and does not require retrieval.\n'
    '- "Explain photosynthesis in simple terms." because it is a textbook explanation.\n'
    '- "List the main causes of the French Revolution." because it is a broad history summary.\n'
    '- "What is the difference between a supernova and a black hole?" because it is a timeless comparison.\n'
    '- "What are the health benefits of a Mediterranean diet?" because it is a generic wellness explainer.\n'
    '- "What are the latest developments in artificial intelligence?" because it '
    "invites vague trend talk instead of exact retrieval.\n"
    '- "What is the current population of the United States?" because it changes too '
    "slowly and a memorized answer would often be close enough.\n"
    '- "What is the current price of Bitcoin?" because it changes too quickly and the '
    "answer may be stale before evaluation.\n"
    "\n"
    "Good query examples:\n"
    '- "Which two official filings report different period bases for the named companies, '
    'which company is higher after normalizing the periods, and what figures support that?"\n'
    '- "Which official result and independent contemporaneous report agree on the named '
    'policy outcome, and what detail differs between their descriptions?"\n'
    '- "Which named regulatory announcement and affected company filing describe the same '
    'rule change, what obligation differs between the two descriptions, and when does it take effect?"\n'
    '- "For the named award result, which official winner page and independent event '
    'coverage agree on the winner, and what credited role detail must be reconciled between them?"\n'
    "\n"
    "Diversity requirements:\n"
    "- The batch should span multiple subject areas and should not collapse into one domain.\n"
    "- The batch should span multiple query shapes and should not collapse into "
    "repeated versions of the same template.\n"
    '- For example, "Who is the current CEO of X?" and "Who currently leads Y?" are '
    "the same underlying template with different nouns, so only one such structure "
    "should appear in a batch.\n"
    "- In a 10-task batch, use several query families: cross-entity comparison, "
    "finalized official result, recent leadership or policy transition, and "
    "multi-source synthesis.\n"
    "- Make cross-source comparison and synthesis the default shape, while still "
    "spanning subject areas.\n"
    "- A batch may include a small number of direct lookup questions only when the "
    "answer requires resolving ambiguity across sources.\n"
    "- At most one task may be a premise-correction task, and only when the "
    "correction itself is clearly verifiable from public evidence.\n"
    "- Do not let more than a small minority of tasks use the same year-framed "
    "'what happened in YEAR' structure.\n"
    "\n"
    "Final check: if a well-informed person without access to search or reference "
    "material could still answer the question correctly, the query is too easy and "
    "must be rejected.\n"
    "Final check: if one source can answer every requested field without reconciliation, "
    "the query is too shallow and must be rejected.\n"
    "Return only the requested JSON payload."
)

_TASK_GENERATION_USER_PROMPT = (
    "Current timestamp: {timestamp}\n"
    "Generate exactly {generation_task_total} user queries.\n"
    "Every query must have a definite affirmative answer already documented by this "
    "timestamp, except for the allowed small premise-correction minority.\n"
    "Favor questions that require independent-source synthesis and cannot be "
    "answered shallowly from one evidence item.\n"
    "Most queries should require an independent-source synthesis step. Preferred "
    "patterns include comparing multiple current sources, reconciling periods and "
    "bases, and combining separately sourced current facts while the batch still "
    "spans varied subject areas.\n"
    "Do not generate vague, generic, evergreen, unreleased-item, invented-version, "
    "or old-memory 'most recent' questions.\n"
    "Do not generate simple lookup tasks where a top result can provide every requested field.\n"
    "At most one task may be a premise-correction task.\n"
    "Do not generate more than one query with the same underlying structure.\n"
    "Before including each query, silently check that every named event, document, "
    "version, release, treaty, award, audit, landing, opening, or transition exists "
    "by the current timestamp; the intended answer is affirmative and publicly "
    "verifiable unless this is the single allowed premise-correction task; at least "
    "two query-required subclaims likely need separate public sources; and no single "
    "top result can answer every requested field. If any check fails, discard that "
    "query and generate another.\n"
    "Do not create difficulty by stacking fragile status words such as finalized, "
    "signed, audited, newly opened, released, first, official, or successfully "
    "completed unless each exact status is already publicly documented by the "
    "current timestamp.\n"
    "Remember: if the answer would be materially the same regardless of when the "
    "question is asked, the query is invalid even if it mentions a current entity.\n"
    'Return JSON with shape: {{"tasks": [{{"text": "..."}}]}}.\n'
    "Keep each query concise, natural, distinct from the others, and realistic for an actual user."
)

_REFERENCE_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's query directly in JSON.\n"
    "Your job is to produce a reference answer for evaluation that is factually "
    "correct, specific, direct, and grounded in retrieved evidence.\n"
    "Factual-correctness credit comes only from claims supported by retrieved "
    "evidence. Unsupported time-sensitive claims actively weaken the answer. A "
    "shorter fully grounded answer is better than a longer answer with unverifiable "
    "details.\n"
    "Use retrieved evidence as the source of truth for time-sensitive factual claims. "
    "Do not state time-sensitive facts from prior knowledge unless they are "
    "supported by retrieved evidence. This applies to the core answer, not just "
    "extra details.\n"
    "Stable, widely established facts may be stated directly. Any time-sensitive "
    "claim, current status, recent change, evolving figure, latest result, or "
    "comparison must be supported by retrieved evidence or omitted.\n"
    "Every concrete claim about current names, recent dates, current numbers, "
    "current rankings, current prices, latest releases, current leadership, recent "
    "outcomes, current status, recent changes, or comparisons of current values "
    "must be supported by retrieved evidence.\n"
    "If a claim is not supported by retrieved evidence, leave it out instead of "
    "guessing. If the core fact needed to answer the question cannot be verified, "
    "say that directly and briefly instead of inventing an answer.\n"
    "A citation note is scorer-visible evidence. Write each note as a compact "
    "factual grounding snippet that states the exact fact it supports.\n"
    "Do not write citation notes as labels such as 'mentions this topic' or "
    "'describes the issue'.\n"
    "Every load-bearing claim in the answer must be traceable to at least one "
    "citation note that explicitly supports that claim.\n"
    "First identify the query-required subclaims internally. The final answer must "
    "cover only the subclaims supported by retrieved evidence.\n"
    "For comparison and synthesis queries, cite each entity, period, value, and "
    "reconciled conclusion with targeted notes. Use independent sources when the "
    "query depends on independent-source synthesis.\n"
    "Do not let one broad citation stand in for multiple unsupported subclaims.\n"
    "When these rules conflict, grounding takes priority over completeness. A "
    "partial answer composed entirely from verified evidence is better than a "
    "complete answer that includes unverified claims.\n"
    "Do not pad the answer with generic trend talk, filler, or broad background "
    "material that is not needed to answer the query.\n"
    "Give the strongest affirmative answer that the evidence supports. If "
    "evidence is strong, answer directly and specifically with the relevant "
    "concrete fields and targeted citations.\n"
    "If evidence is partial, include only verified claim fragments and make the "
    "missing fragment explicit.\n"
    "For comparisons, verify that compared values use the same reporting period "
    "and basis before answering.\n"
    "For financial, vote, award, leadership, ranking, and current-status claims, "
    "prefer primary or official sources when available.\n"
    "For negative or premise-rejecting answers, cite evidence that grounds the "
    "rejection itself, not merely a nearby fact.\n"
    "For premise-heavy or time-sensitive queries, verify the existence or status "
    "premise first. If it is not verified, stop after a concise evidence-backed "
    "correction.\n"
    "\n"
    "Good answer fragment:\n"
    '- Query: "Who is the current CEO of Starbucks, and when did they assume the role?"\n'
    '- Good answer: "Brian Niccol became CEO of Starbucks on September 9, 2024."\n'
    "- Why this is good: it gives the supported name and date directly, with no filler.\n"
    "\n"
    "Bad answer fragment:\n"
    '- Query: "Who is the current CEO of Starbucks, and when did they assume the role?"\n'
    '- Bad answer: "Starbucks has undergone several leadership transitions in recent '
    'years, and the current CEO took over in late 2024."\n'
    "- Why this is bad: it is vague, padded, and weak on the core fact.\n"
    "\n"
    "Bad answer fragment:\n"
    '- Query: "What was the final vote on the EU AI Act?"\n'
    '- Bad answer: "The European Parliament voted 523 to 46 in favor of the EU AI Act on March 13, 2024."\n'
    "- Why this is bad if unsupported: precise time-sensitive numbers and dates are "
    "harmful when they are not backed by retrieved evidence.\n"
    "\n"
    "Fallback answer fragment:\n"
    "- If the available retrieved evidence does not clearly establish the answer, say "
    "that you could not verify the key fact from available evidence and do not "
    "invent the missing detail.\n"
    "- Never return a negative, 'could not verify', 'no such event', or false-premise "
    "answer with omitted or empty citations.\n"
    "\n"
    "False-premise examples:\n"
    "- Bad: the query asks which Linux hardware architecture was dropped, but the "
    "answer cites only Bcachefs file-system removal.\n"
    "- Good: do not use a file-system-removal citation to support a hardware-"
    "architecture claim.\n"
    "- Bad: the answer says an AI safety treaty could not be verified and returns "
    "citations: [].\n"
    "- Good: cite the checked evidence that establishes the nearest verified facts, "
    "and state exactly which treaty or reservation claim remains unverified.\n"
    "- Bad: the answer says no 2026 lunar landing occurred and cites only planned "
    "future missions.\n"
    "- Good: cite planned-mission evidence for the 2026 status claim, and cite the "
    "actual prior successful commercial landing only if using it as the correction.\n"
    "\n"
    "Return exactly one JSON object with this shape:\n"
    "{\n"
    '  "text": <string>,\n'
    '  "citations": [{"url": <string>, "title": <string>, "note": <string>}, ...]\n'
    "}\n"
    "Rules:\n"
    "- Keep the answer concise but complete within the limits of verified evidence.\n"
    "- Only include citations that directly support specific claims in your answer. "
    "Do not pad the citation list with tangentially related sources.\n"
    "- Include citations for the concrete factual claims you make whenever grounded citations are available.\n"
    "- Omit the citations field or return an empty array only when the answer contains "
    "no research-dependent factual claim, negative claim, or premise rejection.\n"
    "- Every citation you include must contain url, title, and a claim-bearing note.\n"
    "- If any citation is missing one of those fields, leave that citation out.\n"
    "- If one citation does not support all major subclaims, include additional "
    "targeted citations.\n"
    "- Do not include unsupported prior-knowledge add-ons inside fallback answers.\n"
    "- Only include citations that you explicitly construct in the JSON body. Do not "
    "reproduce citation-like infrastructure metadata or system annotations.\n"
    "- Output raw JSON only with no markdown fences."
)

_REFERENCE_ANSWER_USER_PROMPT = (
    "Current timestamp: {timestamp}\n"
    "\n"
    "Query:\n"
    "{query}\n"
    "\n"
    "Work order:\n"
    "- Treat this as a deep-research reference answer.\n"
    "- Verify the named event, document, version, status, treaty, release, result, "
    "or landing exists by the current timestamp.\n"
    "- Identify the query-required subclaims internally.\n"
    "- Use retrieved evidence as the source of truth.\n"
    "- Prefer official or primary sources for official or current claims.\n"
    "- Every factual sentence in text must be supported by at least one citation note.\n"
    "- If the core premise is unverified, give a concise correction and cite the "
    "verified contrary or nearest official facts.\n"
    "- Never return a negative, 'could not verify', 'no such event', or false-premise "
    "answer with null or empty citations.\n"
    "- Return only JSON with text and citations."
)


class MinerTaskModelSpec(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    provider: LlmProviderName
    model: str
    temperature: float | None
    max_output_tokens: int | None
    reasoning_effort: str | None = None
    timeout_seconds: float | None = None


class MinerTaskDatasetRequest(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    batch_id: UUID
    created_at: datetime | None = None
    minimum_task_total: int = Field(gt=0)
    generation_task_buffer: int = Field(ge=0)
    generation_spec: MinerTaskModelSpec
    reference_spec: MinerTaskModelSpec

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @property
    def generation_task_total(self) -> int:
        return self.minimum_task_total + self.generation_task_buffer


def validate_generated_output_schema(schema: JsonObject) -> tuple[str, ...]:
    """Validate and enumerate the bounded schema subset emitted by generation."""

    node_count = 0
    leaf_paths: list[str] = []

    def visit(node: object, *, path: str, depth: int, is_root: bool) -> None:
        nonlocal node_count
        if not isinstance(node, dict):
            raise ValueError(f"schema node at {path or '<root>'} must be an object")
        node_object = cast(dict[str, object], node)
        node_count += 1
        if depth > GENERATED_OUTPUT_SCHEMA_MAX_DEPTH:
            raise ValueError("generated output schema exceeds maximum depth")
        if node_count > GENERATED_OUTPUT_SCHEMA_MAX_NODES:
            raise ValueError("generated output schema exceeds maximum node count")

        schema_type = node_object.get("type")
        if schema_type == "object":
            required_object_keys = {
                "type",
                "properties",
                "required",
                "additionalProperties",
            }
            allowed_object_keys = required_object_keys | _GENERATED_OUTPUT_SCHEMA_ANNOTATIONS
            if is_root:
                allowed_object_keys.add("$schema")
            actual_object_keys = set(node_object) - ({"$schema"} if is_root else set())
            if not required_object_keys <= actual_object_keys or not actual_object_keys <= allowed_object_keys:
                missing_keys = sorted(required_object_keys - actual_object_keys)
                extra_keys = sorted(actual_object_keys - allowed_object_keys)
                raise ValueError(
                    "object schemas require type, properties, required, and additionalProperties with only "
                    "bounded annotations; "
                    f"missing={missing_keys}, extra={extra_keys}"
                )
            properties = node_object.get("properties")
            required = node_object.get("required")
            if not isinstance(properties, dict) or not properties:
                raise ValueError("object schemas require non-empty properties")
            properties_object = cast(dict[str, object], properties)
            if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
                raise ValueError("object schemas require a string required list")
            required_names = cast(list[str], required)
            if len(required_names) != len(set(required_names)) or set(required_names) != set(properties_object):
                raise ValueError("every object property must be required exactly once")
            if node_object.get("additionalProperties") is not False:
                raise ValueError("object schemas require additionalProperties=false")
            for property_name, child in properties_object.items():
                if not isinstance(property_name, str) or not _GENERATED_OUTPUT_SCHEMA_PROPERTY_NAME.fullmatch(
                    property_name
                ):
                    raise ValueError(f"unsafe generated property name: {property_name!r}")
                child_path = f"{path}.{property_name}" if path else property_name
                visit(child, path=child_path, depth=depth + 1, is_root=False)
            return

        if is_root:
            raise ValueError("generated output schema root must be an object")
        if schema_type == "array":
            required_array_keys = {"type", "items"}
            allowed_array_keys = (
                required_array_keys
                | _GENERATED_OUTPUT_SCHEMA_ANNOTATIONS
                | _GENERATED_OUTPUT_ARRAY_ASSERTIONS
            )
            if not required_array_keys <= set(node_object) or not set(node_object) <= allowed_array_keys:
                missing_keys = sorted(required_array_keys - set(node_object))
                extra_keys = sorted(set(node_object) - allowed_array_keys)
                raise ValueError(
                    "array schemas require type and items with only bounded annotations and assertions; "
                    f"missing={missing_keys}, extra={extra_keys}"
                )
            for assertion_name in _GENERATED_OUTPUT_ARRAY_ASSERTIONS:
                assertion = node_object.get(assertion_name)
                if assertion_name in node_object and (
                    not isinstance(assertion, int) or isinstance(assertion, bool)
                ):
                    raise ValueError(f"generated {assertion_name} must be a JSON integer")
            if node_object.get("maxItems") == 0:
                raise ValueError(
                    "generated output schema constraints must not disclose the answer: "
                    "maxItems=0 pins an empty array"
                )
            if (
                "minItems" in node_object
                and "maxItems" in node_object
                and node_object["minItems"] == node_object["maxItems"]
            ):
                raise ValueError(
                    "generated output schema constraints must not disclose the answer: "
                    "minItems and maxItems must differ"
                )
            visit(
                node_object.get("items"),
                path=f"{path}[]",
                depth=depth + 1,
                is_root=False,
            )
            return
        if schema_type not in _GENERATED_OUTPUT_PRIMITIVE_SCHEMA_TYPES:
            raise ValueError(f"unsupported generated schema type: {schema_type!r}")
        allowed_primitive_keys = {"type"} | _GENERATED_OUTPUT_SCHEMA_ANNOTATIONS
        if schema_type == "string":
            allowed_primitive_keys |= _GENERATED_OUTPUT_STRING_ASSERTIONS
            for assertion_name in _GENERATED_OUTPUT_STRING_ASSERTIONS:
                assertion = node_object.get(assertion_name)
                if assertion_name in node_object and (
                    not isinstance(assertion, int) or isinstance(assertion, bool)
                ):
                    raise ValueError(f"generated {assertion_name} must be a JSON integer")
            if node_object.get("maxLength") == 0:
                raise ValueError(
                    "generated output schema constraints must not disclose the answer: "
                    "maxLength=0 pins an empty string"
                )
            if (
                "minLength" in node_object
                and "maxLength" in node_object
                and node_object["minLength"] == node_object["maxLength"]
            ):
                raise ValueError(
                    "generated output schema constraints must not disclose the answer: "
                    "minLength and maxLength must differ"
                )
        answer_assertions = sorted(set(node_object) & _GENERATED_OUTPUT_VALUE_ASSERTIONS)
        if answer_assertions:
            raise ValueError(
                "generated output schema constraints must not disclose the answer: "
                f"value assertions are not allowed: {answer_assertions}"
            )
        unknown_keys = sorted(set(node_object) - allowed_primitive_keys)
        if unknown_keys:
            raise ValueError(f"generated output schema contains unsafe keywords: {unknown_keys}")
        leaf_paths.append(path)
        if len(leaf_paths) > GENERATED_OUTPUT_SCHEMA_MAX_LEAVES:
            raise ValueError("generated output schema exceeds maximum leaf count")

    visit(schema, path="", depth=1, is_root=True)
    return tuple(leaf_paths)


class _GeneratedTaskPayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    text: str = Field(min_length=1)


class _GeneratedTaskBatchPayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    tasks: list[_GeneratedTaskPayload]


class _ReferenceAnswerCitationPayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    # This is the raw LLM-output payload shape. Fields stay nullable here so we
    # can parse a response that mixes complete and incomplete citations, then
    # dismiss only the incomplete citations in _complete_reference_citations(...)
    # instead of failing the entire reference answer.
    url: str | None = None
    note: str | None = None
    title: str | None = None


class _ReferenceAnswerPayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    text: str = Field(min_length=1)
    citations: list[_ReferenceAnswerCitationPayload] | None = None


def _complete_reference_citations(
    citations: list[_ReferenceAnswerCitationPayload] | None,
) -> tuple[AnswerCitation, ...] | None:
    if not citations:
        return None
    complete = tuple(
        AnswerCitation(
            url=citation.url,
            note=citation.note,
            title=citation.title,
        )
        for citation in citations
        if citation.url and citation.note and citation.title
    )
    if not complete:
        return None
    return complete


@dataclass(frozen=True, slots=True)
class _PreparedGeneratedTaskBatch:
    selected_tasks: tuple[_GeneratedTaskPayload, ...]
    raw_task_total: int
    unique_task_total: int


def _deduplicate_generated_tasks(
    tasks: list[_GeneratedTaskPayload],
) -> tuple[_GeneratedTaskPayload, ...]:
    seen: set[str] = set()
    unique_tasks: list[_GeneratedTaskPayload] = []
    for task in tasks:
        if task.text in seen:
            continue
        seen.add(task.text)
        unique_tasks.append(task)
    return tuple(unique_tasks)


def _generated_task_batch_postprocessor(
    *,
    minimum_task_total: int,
) -> Callable[[LlmResponse], PostprocessResult]:
    shape_postprocessor = pydantic_postprocessor(_GeneratedTaskBatchPayload)

    def _postprocess(response: LlmResponse) -> PostprocessResult:
        shape_result = shape_postprocessor(response)
        if not shape_result.ok:
            return shape_result
        parsed = shape_result.processed
        if not isinstance(parsed, _GeneratedTaskBatchPayload):
            return PostprocessResult(
                ok=False,
                retryable=False,
                reason="dataset postprocessor returned unexpected payload",
                processed=None,
            )

        unique_tasks = _deduplicate_generated_tasks(parsed.tasks)
        unique_task_total = len(unique_tasks)
        if unique_task_total < minimum_task_total:
            return PostprocessResult(
                ok=False,
                retryable=True,
                reason=(
                    "generated unique task count below minimum_task_total: "
                    f"unique={unique_task_total} minimum={minimum_task_total}"
                ),
                processed=None,
            )

        return PostprocessResult(
            ok=True,
            retryable=False,
            reason=None,
            processed=_PreparedGeneratedTaskBatch(
                selected_tasks=unique_tasks[:minimum_task_total],
                raw_task_total=len(parsed.tasks),
                unique_task_total=unique_task_total,
            ),
        )

    return _postprocess


def build_miner_task_model_request(
    *,
    spec: MinerTaskModelSpec,
    system_prompt: str,
    user_prompt: str,
    use_case: str,
    output_mode: Literal["text", "json_object", "structured"],
    output_schema: type[BaseModel] | None = None,
    postprocessor: Callable[[LlmResponse], PostprocessResult] | None = None,
    require_grounding: bool = False,
) -> GroundedLlmRequest | LlmRequest:
    messages = (
        LlmMessage(
            role="system",
            content=(LlmMessageContentPart.input_text(system_prompt),),
        ),
        LlmMessage(
            role="user",
            content=(LlmMessageContentPart.input_text(user_prompt),),
        ),
    )
    if output_mode == "structured" and output_schema is None:
        raise ValueError("structured output requires output_schema")
    resolved_postprocessor = postprocessor
    if resolved_postprocessor is None and output_schema is not None:
        resolved_postprocessor = pydantic_postprocessor(output_schema)
    if require_grounding:
        if spec.provider != VERTEX_PROVIDER:
            raise ValueError(f"grounded mode not supported for provider '{spec.provider}'")
        return GroundedLlmRequest(
            provider=spec.provider,
            model=spec.model,
            messages=messages,
            temperature=spec.temperature,
            max_output_tokens=spec.max_output_tokens,
            reasoning_effort=spec.reasoning_effort,
            timeout_seconds=spec.timeout_seconds,
            use_case=use_case,
            postprocessor=resolved_postprocessor,
        )
    return LlmRequest(
        provider=spec.provider,
        model=spec.model,
        messages=messages,
        temperature=spec.temperature,
        max_output_tokens=spec.max_output_tokens,
        reasoning_effort=spec.reasoning_effort,
        timeout_seconds=spec.timeout_seconds,
        use_case=use_case,
        output_mode=output_mode,
        output_schema=output_schema,
        postprocessor=resolved_postprocessor,
    )


class MinerTaskDatasetBuilder:
    """Builds generic query tasks and reference answers for miner-task batches."""

    def __init__(
        self,
        *,
        generation_llm: LlmProviderPort,
        reference_llm: LlmProviderPort,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._generation_llm = generation_llm
        self._reference_llm = reference_llm
        self._clock = clock

    async def build(
        self,
        request: MinerTaskDatasetRequest,
    ) -> tuple[MinerTask, ...]:
        tasks, _, _ = await self.build_with_usage(request)
        return tasks

    async def build_with_usage(
        self,
        request: MinerTaskDatasetRequest,
    ) -> tuple[tuple[MinerTask, ...], tuple[LlmUsage, ...], tuple[LlmUsage, ...]]:
        generated_queries, generation_usage = await self._generate_queries(request)
        tasks: list[MinerTask] = []
        reference_usages: list[LlmUsage] = []
        for query in generated_queries:
            reference_answer, usage = await self._generate_reference_answer(
                query=query,
                spec=request.reference_spec,
            )
            tasks.append(
                MinerTask(
                    task_id=uuid4(),
                    query=query,
                    reference_answer=reference_answer,
                )
            )
            reference_usages.append(usage)
        return tuple(tasks), (generation_usage,), tuple(reference_usages)

    async def _generate_queries(
        self,
        request: MinerTaskDatasetRequest,
    ) -> tuple[tuple[Query, ...], LlmUsage]:
        response = await self._generation_llm.invoke(
            build_miner_task_model_request(
                spec=request.generation_spec,
                system_prompt=_TASK_GENERATION_SYSTEM_PROMPT,
                user_prompt=_TASK_GENERATION_USER_PROMPT.format(
                    timestamp=self._clock().isoformat(),
                    generation_task_total=request.generation_task_total,
                ),
                use_case="miner_task_dataset_generation",
                output_mode="structured",
                output_schema=_GeneratedTaskBatchPayload,
                postprocessor=_generated_task_batch_postprocessor(
                    minimum_task_total=request.minimum_task_total,
                ),
                require_grounding=False,
            )
        )
        prepared = response.postprocessed
        if not isinstance(prepared, _PreparedGeneratedTaskBatch):
            raise RuntimeError("dataset postprocessor returned unexpected payload")
        raw_task_total = prepared.raw_task_total
        unique_task_total = prepared.unique_task_total
        if unique_task_total != raw_task_total:
            logger.warning(
                "miner-task dataset dropped duplicate generated tasks",
                extra={
                    "data": {
                        "batch_id": str(request.batch_id),
                        "provider": request.generation_spec.provider,
                        "model": request.generation_spec.model,
                        "raw_task_total": raw_task_total,
                        "unique_task_total": unique_task_total,
                    }
                },
            )
        if unique_task_total < request.generation_task_total:
            logger.warning(
                "miner-task dataset generated fewer tasks than requested",
                extra={
                    "data": {
                        "batch_id": str(request.batch_id),
                        "provider": request.generation_spec.provider,
                        "model": request.generation_spec.model,
                        "minimum_task_total": request.minimum_task_total,
                        "generation_task_total": request.generation_task_total,
                        "raw_task_total": raw_task_total,
                        "unique_task_total": unique_task_total,
                    }
                },
            )
        selected_tasks = prepared.selected_tasks
        queries = tuple(Query(text=task.text) for task in selected_tasks)
        return queries, response.usage

    async def _generate_reference_answer(
        self,
        *,
        query: Query,
        spec: MinerTaskModelSpec,
    ) -> tuple[ReferenceAnswer, LlmUsage]:
        response = await self._reference_llm.invoke(
            build_miner_task_model_request(
                spec=spec,
                system_prompt=_REFERENCE_ANSWER_SYSTEM_PROMPT,
                user_prompt=_REFERENCE_ANSWER_USER_PROMPT.format(
                    timestamp=self._clock().isoformat(),
                    query=query.text,
                ),
                use_case="miner_task_reference_answer",
                output_mode="structured",
                output_schema=_ReferenceAnswerPayload,
                require_grounding=True,
            )
        )
        parsed = response.postprocessed
        if not isinstance(parsed, _ReferenceAnswerPayload):
            raise RuntimeError("reference answer generator returned unexpected payload")
        return (
            ReferenceAnswer(
                text=parsed.text,
                citations=_complete_reference_citations(parsed.citations),
            ),
            response.usage,
        )


__all__ = [
    "MinerTaskDatasetBuilder",
    "MinerTaskDatasetRequest",
    "MinerTaskModelSpec",
    "build_miner_task_model_request",
    "validate_generated_output_schema",
]
