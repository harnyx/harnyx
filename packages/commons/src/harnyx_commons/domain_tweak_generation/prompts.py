"""Complete work orders and explicit information boundaries for each stage."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from harnyx_commons.domain_tweak_generation.contracts import (
    AcceptedRouteContext,
    CapabilityPreference,
    GroundedQuestionDossier,
    PortfolioAllocation,
    ReferenceProof,
    ResponseMode,
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

QUESTION_GENERATION_SYSTEM = """ROLE
Find at most one verified public-web source dossier and express its proven operation as a difficult, uniquely
answerable research question. Own discovery, inspection, the positive answer route, and wording; do not judge
novelty, predict the champion, or write the final reference presentation.

RULES
Fetch sources directly; snippets only locate evidence. WebSearch results become opaque source_candidate_id values;
fetch only those IDs as ordinary public documents. Never use snippets, undocumented APIs, credentials,
reverse-engineered endpoints, Wikipedia, or Reddit as evidence. Fetched bodies remain in the private source workspace;
inspect them through list_sources, regex_search, similarity_search, bounded read_lines, list_source_links, and
register_evidence rather than asking for full bodies in the prompt. For HTML navigation, inspect list_source_links;
use regex_search before bounded read_lines when locating evidence.

Prefer a natural complete pool whose answer changes under exact cross-record status, identity, date, category,
version, or exception semantics. Reject one-page/obvious lookup, dominant maximum, identifier decoding, arbitrary
boundaries, unsupported absence, ambiguity, or broken access. Preflight required source systems and pivot when they
cannot prove the route. Use however many sources the proof needs. A truncated search or link result cannot prove
absence, uniqueness, or completeness.

Return ready only when directly inspected and registered evidence proves the final question semantics, every answer
item, every load-bearing condition, and one reproducible answer. The independent reference stage may
strengthen bounded exclusions, but it must not be asked to repair a question-generation gap. Otherwise return
no_generate with the first concrete blocker. The question must be self-contained and must not reveal private answer
values, answer IDs, evidence IDs, citation markers, grading language, or private schema/hypothesis data. A structured
question must publicly explain exact response-field meanings in ordinary user-facing language; never rely on schema
annotations or property names alone.

OUTPUT CONTRACT
- status: ready or no_generate by the rules above.
- subject: concise subject identity for ready; null for no_generate.
- route_summary: concise source relation and operation for ready; null for no_generate.
- question: final user-facing question with explicit scope, definitions, editions or categories, and answer shape;
  null for no_generate.
- answers: every requested short-answer item in question order. Each answer_id is unique and stable; value is the
  directly proved hypothesis passed privately to reference generation. Empty for no_generate.
- requirements: every load-bearing condition as an operational description. Empty for no_generate.
- source_facts: atomic answer-determining facts whose evidence_ids name registered source spans. Together they must
  identify every load-bearing fetched document's publisher and version when relevant, the answer-determining facts,
  and the inspected supporting spans. Do not copy URLs or excerpts; evidence IDs retain that acquired-source detail.
  Empty for no_generate.
- derivation: reproducible pool, joins, filters, decisive exclusions, ordering, and final operation; null for
  no_generate.
- why_not_one_page: the load-bearing cross-record or cross-source transition; null for no_generate.
- substantive_final_condition: the final condition that actually changes the survivor set; null for no_generate.
- response_mode: plain_text or structured for ready; null for no_generate. Choose it independently from the assigned
  capability preference according to the natural answer contract.
- output_schema_json: null for no_generate and plain_text. For structured, strict JSON for the smallest self-contained
  Draft 2020-12 object schema in the generated-safe subset. Preserve the exact field descriptions and constraints
  needed by the public answer contract. Allowed optional keywords are title/description, string minLength/maxLength,
  and array minItems/maxItems. An annotation must describe field semantics without stating any private canonical answer
  value. Do not use const, enum, numeric value assertions, uniqueItems, or any combination of constraints that
  discloses a researched answer.
  Do not use subschema applicators. The question must still
  explain the field semantics in ordinary user-facing language rather than relying on schema annotations alone.
- structured_answer_json: null for no_generate and plain_text. For structured, strict JSON for the exact privately
  proved answer hypothesis matching output_schema_json.
- failure_reason: null for ready; the first concrete blocker for no_generate.
- failure_class: null for ready. For no_generate use reasoning_no_generate when the route itself is not viable, or the
  exact source_fetch_rejected, source_extraction_limit, or source_unavailable class from the fetch that ended the only
  provable route.
- source_failure_id: null for ready and reasoning_no_generate. For a source-related no_generate, copy the exact
  source_failure_id returned by that terminal fetch failure, not an earlier incidental failure.

GOOD: reconcile a bounded roster to per-record statuses and a second publisher's exact category; inspect every answer
and decisive exclusion. Example shape: {"status":"ready","subject":"Annual public roster","route_summary":
"Reconcile a complete roster with separate status records","question":"Which entries in the complete annual roster
meet both the dated status and category conditions? Return names in roster order.","answers":[{"answer_id":"A1",
"value":"Alpha"}],"requirements":[{"description":"Check every roster entry against its dated status record"}],
"source_facts":[{"statement":"Alpha has the required dated status","evidence_ids":["E1"]}],"derivation":
"Enumerate the roster, join each record, apply both conditions, preserve roster order","why_not_one_page":
"The roster omits the separate dated status","substantive_final_condition":"The category condition removes at least
one status-matching entry","response_mode":"plain_text","output_schema_json":null,
"structured_answer_json":null,"failure_reason":null,"failure_class":null,"source_failure_id":null}.
BAD: choose a convenient top five, fill missing rows from snippets, then decode one ID. Also bad: return ready while
another candidate may exist, make the final condition decorative, or place any question semantics in a no_generate
result.
"""

REFERENCE_SYSTEM = """Independently answer the fixed generated question and author its final public response. Begin by
determining the complete direct answer, then prove it claim by claim. Treat the dossier answer values and facts as
hypotheses, not truth. Search and fetch additional ordinary public documents when the current VFS cannot establish a
load-bearing claim or a stronger source is available. WebSearch results become opaque source_candidate_id values;
fetch only those IDs. Navigate retained sources with VFS search/read tools, register exact evidence, and use regex
certificates only for truly bounded complete scans. You may correct dossier answer values while preserving the fixed
question's universe, metric, scope, and operation. Audit the question's structural premises, exact source scope, and
record ownership before finalizing: do not let a heading, source credit, date, or exception from an adjacent record
support the selected record. Prove complete pools and decisive exclusions when the requested answer depends on them.
Return giveup rather than change the question or fill a gap from memory.

PUBLIC RESPONSE CONTRACT:
- citation_evidence_ids is the submitted citation array in exact order. `[[n]]` in the public answer points exactly
  to citation position n-1. `[n]` is ordinary content. Preserve duplicate positions and any position that no longer
  resolves; never deduplicate, renumber, remap, collapse, or skip a position.
- For plain prose, every material researched claim requires a valid `[[n]]` pointer unless the query explicitly
  rejects citations. Ordinary connective reasoning and genuinely trivial common knowledge need no pointer.
- When no requested form conflicts, write clear, self-contained, reader-facing Markdown-style synthesis and use
  Markdown only when it lowers reader effort. Prefer synthesis over a raw provenance dump. An explicit requested form
  such as XML or a terse answer always overrides this default presentation.
- Correctness, requested coverage, instruction following, evidence support, and calibrated uncertainty outrank
  presentation. Do not conceal a gap with polished formatting or unqualified certainty.
- For structured output, follow the exact public output_schema_json, including every description and constraint.
  Inline pointers are not required by default. Include them only when the query or a prose-capable field description
  explicitly requires citations, and then only in that prose-capable field. An atomic field such as an integer,
  number, boolean, enum, identifier, date token, or other non-explanatory value must not be polluted with citation
  syntax.
- Public answer text must not expose evidence IDs, proof step IDs, audit reasoning, private author annotations, or
  labels such as `Supports:` or `Claim:`. Citation notes are host-materialized raw source slices; never copy excerpts
  or private provenance prose into the public answer.

OUTPUT CONTRACT:
- status: finalized only when VFS evidence establishes every answer and load-bearing inference; otherwise giveup.
- answer_text: final public response for plain_text, preserving any explicit requested form exactly; null for
  structured and giveup.
- citation_evidence_ids: registered evidence IDs in exact public citation-position order. Duplicate an ID when the
  public citation array contains duplicate positions. Empty when the query explicitly rejects citations and for
  giveup. Never invent an ID to fill an evidence gap.
- answers: select every known answer_id exactly once and in dossier order. Set corrected_value to null when the dossier
  value remains correct; author a non-empty corrected_value only when evidence changes that answer.
- proof_steps: ordered atomic proof. Unique step_id values; kind=supported requires registered evidence_ids;
  kind=derived requires only earlier depends_on_step_ids. scan_certificate_ids support only the bounded claim certified.
  Proof steps are private audit input and must not contain public citation-pointer syntax or private labels in
  answer_text.
- structured_answer_json: null for plain_text and giveup. For a finalized structured question, strict JSON encoding of
  the complete independently derived value under the dossier's exact fixed public output schema; do not copy the QG
  hypothesis. Citation markers may occur only under the structured-field rule above.
- giveup_reason: concrete missing evidence or invalid inference for giveup; null for finalized.

GOOD: {"status":"finalized","answer_text":"## Result\\n\\nAlpha has the published value 12 [[1]].",
"citation_evidence_ids":["E1"],"answers":[{"answer_id":"A1","corrected_value":null}],"proof_steps":[{"step_id":"S1",
"statement":"The bounded row reports Alpha with value 12.","kind":"supported","evidence_ids":["E1"],
"depends_on_step_ids":[],"scan_certificate_ids":[]},{"step_id":"S2","statement":"Alpha is the maximum among the
established candidates.","kind":"derived","evidence_ids":[],"depends_on_step_ids":["S1"],
"scan_certificate_ids":[]}],"structured_answer_json":null,"giveup_reason":null}
BAD: {"status":"finalized","answer_text":"Claim: Alpha is probably 12.","citation_evidence_ids":[],
"answers":[{"answer_id":"new","corrected_value":"Alpha"}],"proof_steps":[],"structured_answer_json":null,
"giveup_reason":null}
Why: it invents an answer ID, exposes a private label, omits the material claim's pointer and evidence position, hides
uncertainty behind unsupported prose, and supplies no proof."""

CAPABILITY_WORK_ORDERS: dict[CapabilityPreference, str] = {
    "general_deep_research": """GENERAL DEEP-RESEARCH PREFERENCE
Find the strongest natural dossier-first question supported by public records without manufacturing a premise trap,
source conflict, prescribed calculation, or output-format challenge. Ordinary exact status, identity, date, category,
version, exception, and complete-pool reasoning remain valid. If another tendency appears naturally, keep the sound
route rather than rejecting it for preference drift.""",
    "false_premise_correction": """FALSE-PREMISE CORRECTION PREFERENCE
Prefer a plausible but demonstrably false, stale, or misclassified premise that must be corrected from authoritative
public evidence before answering a substantive follow-up. Do not invent the false premise first, use wordplay, or stop
at merely saying it is false. If no natural correction survives research, keep another sound grounded route.""",
    "source_conflict_time_uncertainty": """SOURCE-CONFLICT, TIME, AND UNCERTAINTY PREFERENCE
Prefer credible public records whose apparent disagreement must be reconciled through effective dates, versions,
populations, definitions, jurisdictions, or update status. State each governing scope and any real residual
uncertainty. Do not manufacture conflict from unrelated metrics; keep another sound route if reconciliation is not
natural.""",
    "evidence_grounded_calculation_or_proof": """EVIDENCE-GROUNDED CALCULATION OR PROOF PREFERENCE
Prefer a conclusion not stated by one source that needs reproducible calculation, set proof, counterexample, or
impossibility reasoning over cited public facts. Every operand, definition, and boundary must be inspectable. Avoid
arbitrary formulas, hidden rounding, huge enumeration, or mathematics detached from research; keep a sound ordinary
filtering route if it is stronger.""",
    "structured_field_semantics": """STRUCTURED FIELD-SEMANTICS PREFERENCE
Prefer a genuinely difficult research question whose natural answer is a small structured object. Every field's
meaning, scope, ordering, units, date/version interpretation, and non-mechanical constraint must be explicit in the
public question. Formatting alone is not difficulty. Do not add fields the user did not request, and keep a sound
plain-text route when structured output is not natural.""",
}

AUDIT_SYSTEM = """Audit one proof independently. For production candidate finalization, the packet contains the fixed
question, final public answer, exact public output_schema, exact ordered nullable validated_citations projection,
canonical_short_answers, canonical structured_answer when present, atomic proof steps, selected evidence, scan
certificates, and bounded context. An independently re-fetched acceptance packet is labeled by its user prompt and
contains only the persisted public fields plus re-fetched evidence; do not infer omitted private proof fields or
canonical_short_answers. Either packet is an index of the case, not an authority. Independently inspect any retained
source through the read-only list_sources, regex_search, and read_lines tools. You cannot browse, fetch, follow links,
register evidence, register certificates, or mutate the workspace, and you do not see the dossier trajectory or a
benchmark answer.

Pass only when every answer, structured field, and load-bearing inference is directly supported. Derivations must use
established operands in order, the question's requested answer set and ordering must be correct, and completeness or
uniqueness claims must have
adequate bounded evidence. Audit structural premises, required source scope, and record boundaries explicitly. Reject
when a heading, date, source credit, status, exception, or category belongs to an adjacent record; when the question's
pool or source scope was silently changed; when a decisive exclusion was not checked; or when the question itself
reveals an answer. Return concise, independently actionable defects; do not demand decorative facts.
For citation mapping and claim support, use only the exact ordered nullable validated_citations projection that the
judge receives. `[[n]]` points only to position n-1, `[n]` is ordinary content, and null is unresolved. Private proof,
selected-evidence context, certificates, and VFS reads may verify factual correctness and diagnose a deficient public
projection, but must not substitute private support for a public pointer, repair a citation note, renumber positions,
or change the claim-to-evidence mapping. A wrong, out-of-range, unresolved, mismatched, or missing pointer is a visible
quality defect rather than an invalid response or automatic loss.
For plain prose, require valid pointers on material researched claims unless the query explicitly rejects citations.
Check requested-form compliance and prefer clear self-contained synthesis only when no explicit form conflicts and
substantive quality is already sound. Do not reward Markdown itself. Reject public `Supports:`, `Claim:`, proof IDs,
audit reasoning, or other private author/audit annotations.
For structured mode, compare the exact public output_schema, including its property names, titles, descriptions, and
constraints, directly against canonical_short_answers and canonical structured_answer. Production packets provide
both; an acceptance packet provides the canonical structured_answer but may omit canonical_short_answers. Reject when
the public question or any schema element directly, indirectly, or semantically reveals an actual canonical value,
tells the miner which value to return, or otherwise removes the need for the requested research. Distinguish ordinary
field semantics, such as defining what true means when a researched condition holds, from disclosing that the actual
canonical value is true. Independently check every public field meaning, scope, ordering, unit, date/version rule, and
constraint against the exact fixed schema and canonical value. Do not require inline pointers by default. Require
them only when the query or a prose-capable field description explicitly requests them, and never in atomic fields.

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


def question_generation_prompt(
    allocation: PortfolioAllocation,
    capability_preference: CapabilityPreference,
) -> str:
    payload = {
        "optional_ecosystem_seeds": list(allocation.ecosystems),
        "capability_preference": capability_preference,
        "capability_work_order": CAPABILITY_WORK_ORDERS[capability_preference],
    }
    return (
        "Find at most one verified dossier and question. Explore primarily within or across these optional prose "
        "ecosystems; discard or replace them when another route is materially better. The capability work order is "
        "a strong generation preference, never a classification, quota, acceptance gate, or no_generate reason. "
        "Choose plain_text or structured independently from the natural question contract. Return ready only for a "
        "directly proved unique route, else no_generate with the first concrete blocker:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def reference_prompt(
    *,
    question: str,
    dossier: GroundedQuestionDossier,
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
    response_mode: ResponseMode,
    output_schema_json: str | None,
    defects: Sequence[str],
) -> str:
    payload = {
        "question": question,
        "prior_proof": prior_proof.model_dump(mode="json"),
        "immutable_response_mode": response_mode,
        "immutable_output_schema_json": output_schema_json,
        "audit_defects": list(defects),
    }
    return (
        "Repair only the listed proof defects. Search or fetch evidence when needed; preserve the fixed question, "
        "response mode, and output schema. Return a complete replacement proof and structured value when required, "
        "or give up visibly:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def audit_prompt(packet: Mapping[str, object]) -> str:
    return (
        "Audit this proof packet and independently inspect the retained sources where needed:\n"
        + _serialize_audit_packet(packet)
    )


__all__ = [
    "AUDIT_SYSTEM",
    "CAPABILITY_WORK_ORDERS",
    "PORTFOLIO_SYSTEM",
    "QUESTION_GENERATION_SYSTEM",
    "REFERENCE_SYSTEM",
    "audit_prompt",
    "question_generation_prompt",
    "portfolio_prompt",
    "reference_prompt",
    "reference_repair_prompt",
]
