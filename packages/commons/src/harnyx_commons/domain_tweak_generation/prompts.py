"""Complete work orders and explicit information boundaries for each stage."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from harnyx_commons.domain_tweak_generation.contracts import (
    AcceptedRouteContext,
    GenerationForm,
    PortfolioAllocation,
    ReferenceProof,
    SourceDossier,
)
from harnyx_commons.domain_tweak_generation.source_workspace import _serialize_audit_packet

PORTFOLIO_SYSTEM = """You allocate diverse public-document search spaces for independent benchmark-question authors.
You do not receive a source form or benchmark answer and must not invent a question or answer. For every input slot,
propose exactly five materially different ecosystem paragraphs. An ecosystem is a broad
subject plus promising public document families, not a factual claim or required route. The downstream dossier agent
may mix, discard, or replace all suggestions. Avoid any already accepted subject, route, or source URL supplied in the
request. Do not use APIs, credentials, named answer candidates, exact lookup facts, or near-neighbor rewrites.

OUTPUT CONTRACT:
- allocations: every requested slot exactly once and no other slot.
- slot: copy the input value exactly.
- ecosystems: exactly five standalone, materially different optional search spaces.

GOOD: {"allocations":[{"slot":0,"ecosystems":["Public collection catalogs with bounded
accession tables.","Regulatory registers with annual tabulations.","Scientific inventories with labeled appendices.",
"Competition archives with complete standings.","Legislative reports with enumerated exhibits."]}]}
Why: it preserves slot identity while offering unrelated document families.
BAD: {"allocations":[{"slot":0,"ecosystems":["Find the named 2024 report and answer Alpha."]}]}
Why: it supplies too few ecosystems and invents a lookup and answer."""

DOSSIER_SYSTEM = """You construct one question-ready source dossier from five optional ecosystem ideas. You never see
the source form or benchmark answer. Search actively and stop at the first supported,
interesting route; universal success is not required. Mix, discard, or replace ecosystems and sources whenever useful.
WebSearch snippets are discovery context, never evidence. A PostToolUse message supplies opaque source_candidate_id
values for URLs returned by WebSearch. Fetch only those IDs with fetch_source, declaring the document kind and why it is
an ordinary public document. Never invent or copy a URL into fetch_source. Do not use undocumented APIs, credentials,
reverse-engineered endpoints, Wikipedia, Reddit, or search snippets as evidence.

Fetched HTML/text/static-JSON, PDF, and XLSX documents remain in a private VFS. Use list_sources, regex_search,
similarity_search, read_lines, and register_evidence to locate exact support without requesting a whole source in the
prompt. Retain as many independent sources as the route needs. A truncated search result cannot prove absence,
uniqueness, or completeness. Return ready only when registered evidence supports the question material, every
load-bearing requirement, the complete answer set implied by the proposed question, and the stated derivation with no
answer-material gap. Choose a natural non-empty answer set whose values are established by that evidence. Otherwise
return no_generate promptly. Do not write the final question or reference answer.

OUTPUT CONTRACT:
- status: ready only under the rule above; otherwise no_generate.
- subject: concise subject identity for ready; null for no_generate.
- route_summary: concise relation and source path for ready; null for no_generate.
- question_plan: supported retrieval/filter/comparison operation, not final wording.
- answers: one or more supported answers for ready. Each answer_id is unique and stable; value is the dossier
  hypothesis.
- requirements: every load-bearing condition as an operational description.
- source_facts: atomic facts whose evidence_ids name already registered VFS spans.
- derivation: reproducible operation from supported facts to answer IDs.
- unresolved_gaps: empty for ready; concrete missing answer material for no_generate.
- no_generate_reason: concrete blocker for no_generate; null for ready.
- failure_class: reasoning_no_generate when the route itself is not viable; otherwise the exact source_fetch_rejected,
  source_extraction_limit, or source_unavailable failure returned by the fetch_source attempt that prevents generation.
  Required for no_generate and null for ready.
- source_failure_id: for a source-related no_generate, copy the source_failure_id returned by the exact fetch_source
  failure that prevents generation. Do not select an earlier incidental failure after another route remains viable. Use
  null for reasoning_no_generate and ready.

GOOD: {"status":"ready","subject":"Annual collection awards","route_summary":"Join the annual index to two award
tables","question_plan":"Filter the complete nominees then compare reported totals","answers":[{"answer_id":"A1",
"value":"Alpha"},{"answer_id":"A2","value":"Beta"}],"requirements":[{"description":"Use only the bounded
nominee table"}],"source_facts":[{"statement":"Alpha has total 12","evidence_ids":["E1"]}],"derivation":
"Apply the bounded-table requirement and select the two highest supported totals","unresolved_gaps":[],
"no_generate_reason":null,"failure_class":null,"source_failure_id":null}
Why: it uses stable IDs, registered evidence, and a reproducible supported path.
GOOD no-generate: {"status":"no_generate","subject":null,"route_summary":null,"question_plan":null,"answers":[],
"requirements":[],"source_facts":[],"derivation":null,"unresolved_gaps":["The public document omits the bounded
candidate table"],"no_generate_reason":"The required table is absent from every retained source.","failure_class":
"reasoning_no_generate","source_failure_id":null}
Why: it states the actual terminal decision instead of attributing an unrelated earlier tool failure.
GOOD source no-generate: {"status":"no_generate","subject":null,"route_summary":null,"question_plan":null,
"answers":[],"requirements":[],"source_facts":[],"derivation":null,"unresolved_gaps":["The required report could not
be fetched"],"no_generate_reason":"The only public report needed for the route was unavailable.","failure_class":
"source_unavailable","source_failure_id":"source_failure:2"}
Why: the class and ID identify the exact fetch failure that ended the route.
BAD no-generate: {"status":"no_generate","failure_class":"source_unavailable","source_failure_id":
"source_failure:1"}
Why: a source failure ID is not a terminal cause merely because it occurred earlier; use reasoning_no_generate when the
route ultimately fails for a different reason.
BAD: {"status":"ready","answers":[{"answer_id":"A1","value":"Alpha"}],"unresolved_gaps":["Other candidates
may exist"]}
Why: it claims readiness while retaining an answer-determining gap."""

QUESTION_SYSTEM = """You write one natural benchmark question from a verified dossier and a sampled source form. The
form is structural and difficulty inspiration, not a template to compile literally. Preserve the useful character of
its multi-step retrieval, filtering, comparison, ordering, or completeness challenge only where that fits the dossier
naturally. Do not force relation-by-relation isomorphism. Use only supplied dossier facts. Do not include private answer
values, answer IDs, evidence IDs, citation markers, grading language, or JSON instructions in the question or output.
Prefer a connected route that cannot be solved from one supplied source alone. Return giveup when no natural,
self-contained, supported question can be written.

OUTPUT CONTRACT:
- status: generated only for a fully supported natural question; otherwise giveup.
- question: complete user-facing question for generated; null for giveup.
- form_transfer_explanation: specifically explain which useful structural/difficulty properties transferred naturally;
  required for both statuses and never user-facing.
- giveup_reason: concrete unsupported or unnatural condition for giveup; null for generated.

GOOD: {"status":"generated","question":"Which two entries meet both independently sourced conditions? Give them in
the order established by the annual table.","form_transfer_explanation":"Transfers the source form's bounded filtering,
cross-source comparison, and ordered-result requirement without copying its domain.","giveup_reason":null}
BAD: {"status":"generated","question":"Why are Alpha and Beta the answers?","form_transfer_explanation":"Same
topic.","giveup_reason":null}
Why: it leaks answers and does not explain a meaningful form transfer."""

REFERENCE_SYSTEM = """Independently answer the fixed generated question. Treat the dossier answer values and facts as
hypotheses, not truth. Search and fetch additional ordinary public documents when the current VFS cannot establish a
load-bearing claim or a stronger source is available. WebSearch results become opaque source_candidate_id values;
fetch only those IDs. Navigate retained sources with VFS search/read tools, register exact evidence, and use regex
certificates only for truly bounded complete scans. You may correct dossier answer values while preserving the fixed
question's universe, metric, scope, and operation. Return giveup rather than change the question or fill a gap from
memory. Do not author citation markers or copy excerpts into output.

OUTPUT CONTRACT:
- status: finalized only when VFS evidence establishes every answer and load-bearing inference; otherwise giveup.
- answers: select every known answer_id exactly once and in dossier order. Set corrected_value to null when the dossier
  value remains correct; author a non-empty corrected_value only when evidence changes that answer.
- proof_steps: ordered atomic proof. Unique step_id values; kind=supported requires registered evidence_ids;
  kind=derived requires only earlier depends_on_step_ids. scan_certificate_ids support only the bounded claim certified.
- giveup_reason: concrete missing evidence or invalid inference for giveup; null for finalized.

GOOD: {"status":"finalized","answers":[{"answer_id":"A1","corrected_value":null}],"proof_steps":[{"step_id":"S1",
"statement":"The bounded row reports Alpha with value 12.","kind":"supported","evidence_ids":["E1"],
"depends_on_step_ids":[],"scan_certificate_ids":[]},{"step_id":"S2","statement":"Alpha is the maximum among the
established candidates.","kind":"derived","evidence_ids":[],"depends_on_step_ids":["S1"],
"scan_certificate_ids":[]}],"giveup_reason":null}
BAD: {"status":"finalized","answers":[{"answer_id":"new","corrected_value":"Alpha[[1]]"}],"proof_steps":[]}
Why: it invents an ID, authors a citation marker, and supplies no proof."""

AUDIT_SYSTEM = """Audit one proof packet independently. You receive only the question, final answer values, atomic
proof steps, selected exact excerpts, scan certificates, and bounded context. You cannot browse and do not see
unselected source bodies, the dossier trajectory, source form, or benchmark answer. Pass only when every answer and
load-bearing inference is directly supported, derivations use established operands in order, the question's requested
answer set and ordering are correct, and completeness or uniqueness claims have adequate bounded evidence. Reject if
the question itself reveals an answer. Return concise, independently actionable defects; do not demand decorative facts.

OUTPUT CONTRACT:
- status: pass only when every criterion holds; otherwise reject.
- defects: empty for pass; one concrete proof or question defect per item for reject.
- explanation: brief reason grounded only in the packet.

GOOD: {"status":"reject","defects":["Step S3 compares a value absent from selected evidence."],"explanation":"The
maximum claim lacks one operand."}
BAD: {"status":"pass","defects":[],"explanation":"The answer is plausible from general knowledge."}
Why: outside plausibility is not evidence."""


def portfolio_prompt(
    slots: Sequence[int],
    *,
    accepted_route_contexts: Sequence[AcceptedRouteContext] = (),
) -> str:
    payload = {
        "slots": [{"slot": slot} for slot in slots],
        "already_accepted_routes_to_avoid": [item.model_dump(mode="json") for item in accepted_route_contexts],
    }
    return "Allocate public-document ecosystems for this request:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def dossier_prompt(allocation: PortfolioAllocation) -> str:
    payload = {
        "optional_ecosystem_seeds": list(allocation.ecosystems),
    }
    return "Find the first question-ready supported source dossier:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


def question_prompt(
    form: GenerationForm,
    allocation: PortfolioAllocation,
    dossier: SourceDossier,
    evidence_identities: Sequence[Mapping[str, object]],
) -> str:
    payload = {
        "source_form_for_natural_inspiration": form.form,
        "portfolio_brief": dossier_prompt(allocation),
        "verified_dossier": dossier.model_dump(mode="json"),
        "registered_evidence_identities": list(evidence_identities),
    }
    return "Write the final question from this supported packet:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def reference_prompt(
    *,
    question: str,
    dossier: SourceDossier,
    evidence_identities: Sequence[Mapping[str, object]],
) -> str:
    payload = {
        "question": question,
        "dossier_hypothesis": dossier.model_dump(mode="json"),
        "pre_registered_evidence": list(evidence_identities),
    }
    return "Build the minimal complete verified reference proof:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def reference_repair_prompt(
    *,
    question: str,
    prior_proof: ReferenceProof,
    defects: Sequence[str],
) -> str:
    payload = {
        "question": question,
        "prior_proof": prior_proof.model_dump(mode="json"),
        "audit_defects": list(defects),
    }
    return (
        "Repair only the listed proof defects. Search or fetch evidence when needed, preserve the fixed question, and "
        "return a complete replacement proof:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def audit_prompt(packet: Mapping[str, object]) -> str:
    return "Audit this closed proof packet:\n" + _serialize_audit_packet(packet)


__all__ = [
    "AUDIT_SYSTEM",
    "DOSSIER_SYSTEM",
    "PORTFOLIO_SYSTEM",
    "QUESTION_SYSTEM",
    "REFERENCE_SYSTEM",
    "audit_prompt",
    "dossier_prompt",
    "portfolio_prompt",
    "question_prompt",
    "reference_prompt",
    "reference_repair_prompt",
]
