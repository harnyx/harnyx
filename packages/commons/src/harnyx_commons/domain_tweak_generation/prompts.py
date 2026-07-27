"""Complete work orders for source-aware domain-tweak generation stages."""

from __future__ import annotations

import json

from pydantic import BaseModel

from harnyx_commons.domain_tweak_generation.types import DomainTweakAdkPhase, DomainTweakReviewedQuestion
from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_REQUIREMENT_CATEGORIES,
    DomainTweakFormBlueprint,
    DomainTweakPairInput,
    DomainTweakQuestionPacket,
    DomainTweakReferenceAnswerOutput,
)


def phase_instruction(phase: DomainTweakAdkPhase) -> str:
    match phase:
        case "form_blueprint":
            return (
                "You analyze only the reusable information-acquisition form of a source question. "
                "Do not search, solve it, choose a new domain, or invent evidence."
            )
        case "question_generation":
            return (
                "You research one supplied domain and build one simple, grounded question, proposed "
                "short answer, claim manifest, and evidence declaration packet constrained by a form blueprint."
            )
        case "form_review":
            return (
                "You review only whether a generated question preserves an authoritative form blueprint "
                "and enumerate every concrete requirement in that generated question. Do not assess truth."
            )
        case "reference_answer_generation":
            return (
                "You reconstruct a supplied question-generation proposal from acquired source content, "
                "perform only bounded correction, and write a complete reader-facing reference answer."
            )
        case "semantic_support_gate":
            return (
                "You are a no-search evidence-quality gate. Decide whether the supplied answer and acquired "
                "source windows support every declared question requirement and material answer claim."
            )


def form_blueprint_prompt(pair_input: DomainTweakPairInput) -> str:
    return f"""WORK ORDER
Extract a compact form blueprint from the source question. A later search-capable builder will use the
blueprint to create a new question in another domain. Describe the source question's observable operation;
do not solve it or pre-decide facts whose feasibility requires research.

SOURCE QUESTION
{pair_input.deepsearchqa_form_target}

DECISION CRITERIA
Return `proceed` when the answer operation and stated-versus-retrieved boundary can be described without
solving the question. Return `no_generate` only when the source is malformed or lacks a coherent answer
operation. Ambiguity is not a reason to reject; record it without strengthening it.

PROCEDURE
1. Describe the operation: retrieve, compare, filter, rank, aggregate, identify, or a combination.
2. List properties whose removal would change that operation, answer cardinality, or retrieval boundary.
3. Separate visible wording and instance details that may change without changing the operation.
4. Describe which facts are stated and which must be retrieved, including source-attribution behavior.
5. State the answer shape: one item, exhaustive list, number, ranking, comparison, or another literal shape.
6. Record unresolved ordinary-language ambiguity. Do not convert a ranking into a threshold, a threshold
   into a ranking, or `majority` into `more than 50 percent` unless the source question requires that.

FIELD INTERPRETATION
- `status`: `proceed` under DECISION CRITERIA; otherwise `no_generate`.
- `operation`: literal information-acquisition and selection operation, with no new-domain mapping.
- `load_bearing_invariants`: observable properties the later question must preserve.
- `non_load_bearing_surface_features`: visible details that may vary without changing the operation.
- `retrieval_boundary`: facts stated in the question versus facts retrieved from sources.
- `answer_shape`: required output type and cardinality.
- `semantic_ambiguities`: diagnostic ambiguities in the source question. They warn the later builder not to
  silently strengthen the source operation; they are not features that the new question must reproduce.
- `no_generate_reason`: source-form defect for `no_generate`; null for `proceed`.

GOOD EXAMPLE
A closed-list question asks which option ranks highest according to a named table. A good blueprint preserves
closed-list selection, one retrieved measure per option, comparison, source attribution, and one-item output.
It treats the option names and exact number of options as surface details unless their count is queried.

BAD EXAMPLE
A blueprint selects a specific new product, requires exactly seven options, changes `highest` to `over 50%`,
and names a review site. This is invalid because it performs domain research and changes the operation.

OUTPUT
Return every schema field. For `no_generate`, use null analysis fields and empty arrays. Return only the
native structured response; do not add prose or a Markdown fence."""


def question_generation_prompt(
    pair_input: DomainTweakPairInput,
    blueprint: DomainTweakFormBlueprint,
) -> str:
    return f"""WORK ORDER
Research the domain seed and return one complete question, proposed short answer, claim manifest, and
evidence declaration packet that preserves the authoritative form blueprint. Search before committing to
the concrete source, candidate universe, metric, question, or answer. Prefer the easiest well-supported
instantiation; answer strength comes from support and explanation, not artificial difficulty.

CURRENT TIMESTAMP
{pair_input.timestamp.isoformat()}

DOMAIN SEED
{pair_input.deepresearch9k_domain_target}

AUTHORITATIVE FORM BLUEPRINT
{_json(blueprint)}

DECISION CRITERIA
Return `ready` only when the proposed short answer and every answer-determining external claim have direct
public evidence declarations capable of supporting the question as written. A claim may have no external
evidence only when it follows logically or mathematically from stated question facts or externally supported
operands. Return `no_generate` when no simple grounded mapping preserves the blueprint. Make the generated
question unambiguous where possible. Do not reproduce a source-question ambiguity merely because the blueprint
records it, and do not resolve one by changing the operation, threshold relation, or retrieval boundary.

PROCEDURE
1. Search for direct, public, preferably primary or canonical sources that naturally support the blueprint.
2. Fix the candidate universe, requested metric or field, scope, time basis, and answer before wording the question.
3. Keep the question no harder than needed. Do not add extra predicates merely for novelty.
4. State the complete proposed short answer and ordered `solution_steps` needed to reconstruct it.
5. List free-form factual or derived claims. Split claims when different evidence would support them.
6. For each external source, declare its direct URL, title if known, locator if useful, a relevant claimed
   excerpt, the exact claim IDs it supports, and how it supports them. Search queries are trajectory, not evidence.
7. Prefer direct measures. A proxy is permissible only when the packet includes evidence and reasoning that
   make the mapping defensible for the question's exact metric, scope, and margin. Do not silently relabel a field.
8. Construct a valid premise through the substantive packet. Do not emit a premise self-assessment, confidence,
   gap assessment, or instructions to the next agent.

FIELD INTERPRETATION
- `status`: `ready` under DECISION CRITERIA; otherwise `no_generate`.
- `question`: standalone benchmark question preserving the blueprint.
- `short_answer`: the exact proposed answer, not a plan or qualification.
- `solution_steps`: concise reconstruction path, including source operands and derivations.
- `claims[].claim_id`: stable unique ID reused by reference generation.
- `claims[].claim`: one material factual or derived assertion.
- `claims[].role`: `answer_determining` if changing it can change the answer; otherwise `explanatory`.
- `claims[].support_mode`: external source versus logical or mathematical derivation.
- `claims[].support_explanation`: why the declared support mode establishes this claim.
- `evidence_declarations[].evidence_id`: stable unique source declaration ID.
- `source_url`: direct public page URL, not a search-results or temporary redirect URL.
- `source_locator`: table, section, row, date, paragraph, or null.
- `claimed_excerpt`: short source text expected to locate the relevant content after deterministic fetch.
- `supported_claim_ids`: exact claim IDs this source supports.
- `support_explanation`: claim-to-source binding, including field equivalence or proxy defense when applicable.
- `no_generate_reason`: concrete form/domain obstacle; null for `ready`.

GOOD EXAMPLE
A packet uses an official table to define the candidate universe and values, gives each material assertion a
stable claim ID, declares the table URL and relevant row excerpt, and leaves a simple subtraction claim without
a direct source while explaining that it derives from two sourced operands.

BAD EXAMPLE
A packet cites a homepage, says a nearby field is "close enough" without equivalence evidence, declares that
it could not find the exact measure, or asks the next agent to verify it. Those are not support for the claim.

OUTPUT
Return every schema field. For `no_generate`, use null question fields and empty arrays. Return only the native
structured response; do not add prose or a Markdown fence."""


def form_review_prompt(
    blueprint: DomainTweakFormBlueprint,
    packet: DomainTweakQuestionPacket,
) -> str:
    return f"""WORK ORDER
Review the generated question only against the authoritative form blueprint, then enumerate every concrete,
load-bearing requirement in the generated question. Do not search. Do not assess whether its facts, premise,
answer, claims, or evidence are true; those belong to later source reconstruction and semantic support review.

AUTHORITATIVE FORM BLUEPRINT
{_json(blueprint)}

GENERATED QUESTION PACKET
{_json(packet)}

DECISION CRITERIA
Set `form_match` true only when the generated question preserves the blueprint's operation, retrieval boundary,
load-bearing invariants, and answer shape. Topic, surface wording, and the presence or absence of the source
question's recorded semantic ambiguities may differ. Do not reject a clearer generated question merely because
it does not reproduce an ambiguity. Reject an ambiguity resolution only when it changes the operation,
threshold relation, retrieval boundary, or answer shape. Extra context is harmless unless it adds a condition
that can change the answer. Enumerate requirements from the generated question even when `form_match` is false
so the rejection remains inspectable.

REQUIREMENT PROCEDURE
1. Give each independently adjudicable condition a stable unique `requirement_id`.
2. Include the candidate universe, every metric or field relation, scope, time qualifier, cardinality,
   completeness obligation, ranking operation, absence or universal-negative condition, and any other
   condition that can change the answer.
3. Choose `required_relation`: direct fact, derived calculation, field equivalence, exhaustive set, exact
   cardinality, absence, upper or lower bound, or other.
4. Emit exactly one audit row for every category in this list:
   {json.dumps(DOMAIN_TWEAK_REQUIREMENT_CATEGORIES)}
5. Set an audit row's `present` true exactly when at least one emitted requirement has that category. Explain
   why the category is present or absent; do not omit temporal or scope categories merely because they are short.

FIELD INTERPRETATION
- `form_match`: blueprint-preservation verdict only.
- `reviewer_feedback`: precise structural explanation; no premise or evidence verdict.
- `question_requirements`: exhaustive, uniquely identified conditions imposed by the generated question.
- `requirement_category_audit`: complete category-by-category omission check.

GOOD EXAMPLE
For "List every 2024 entry in table T whose value exceeds 10", separate requirements identify table T's
candidate universe, the 2024 time qualifier, the greater-than relation, and exhaustive-list completeness.
The audit marks those categories present and explains why ranking and absence are not present.

BAD EXAMPLE
The review says only "the form looks right", omits the year and completeness requirement, or rejects the
question because its evidence seems weak. The first two lose load-bearing requirements; the last exceeds scope.

OUTPUT
Return every schema field through the native structured response. Do not add prose or a Markdown fence."""


def reference_answer_prompt(
    reviewed_question: DomainTweakReviewedQuestion,
    *,
    source_packet: dict[str, object],
    question_generation_trajectory: dict[str, object],
) -> str:
    return f"""WORK ORDER
Reconstruct the question generator's proposed answer path from the supplied packet, observable search trajectory,
and deterministically acquired source content. Then either produce a complete reader-facing reference answer or
abandon this candidate. You may use Google Search for targeted checks and additional explanatory facts, but a
search result becomes citable evidence only after `acquire_sources` returns an acquired source window.

CURRENT TIMESTAMP
{reviewed_question.pair_input.timestamp.isoformat()}

QUESTION PACKET
{_json(reviewed_question.question_packet)}

FORM REVIEW AND CONCRETE QUESTION REQUIREMENTS
{_json(reviewed_question.form_review)}

OBSERVABLE QUESTION-GENERATION TRAJECTORY
{json.dumps(question_generation_trajectory, ensure_ascii=False, indent=2)}

DETERMINISTIC SOURCE ACQUISITION
{json.dumps(source_packet, ensure_ascii=False, indent=2)}

ROLE AND NON-SCOPE
The packet is a proposal, not an oracle and not an opponent. Follow its solution path, inspect the actual acquired
text, correct only bounded mistakes, strengthen weak support when possible, and write the best complete answer a
strong deep-research agent would give the reader. Never discuss the question generator, packet, tools, missing
access, your search effort, or any other internal circumstance in `reference_answer_text`.

DECISION CRITERIA
- `unchanged`: reconstruction supports the proposed short answer without changing its answer-determining path.
- `bounded_correction`: the same candidate universe, operation, metric, scope, premise, proxy mapping, and solution
  path survive, but a local value, calculation, entity inclusion, wording, or explanation requires correction.
- `abandon`: a defensible answer would require a materially new metric, scope, candidate universe, premise, proxy,
  operation, or solution path; required support remains unavailable; or a source/tool budget terminates the attempt.

PROCEDURE
1. Inspect every initial acquisition result. Treat QG-authored excerpts as locators, never as fetched evidence.
2. Use `read_cached_source(source_id, offset, limit)` to paginate acquired sources when the initial window is
   insufficient. Use offsets and at most the declared per-response limit.
3. For an unresolved known QG claim, use one batched `acquire_sources(claim_id, urls, offset, limit)` call per
   acquisition round. For new facts needed only to explain the answer, use the declared additional-explanation ID.
   Source windows are claim-bound: initial windows inherit their declaration's `supported_claim_ids`, cached reads
   retain those bindings, and supplemental windows bind to the `claim_id` passed to `acquire_sources`. To use an
   already cached URL for another claim, call `acquire_sources` again with that claim ID before attaching its
   returned window; a cache-only call establishes that binding without consuming another acquisition round.
4. Prefer direct evidence. When any claim has weaker support, make the strongest defensible argument from the
   available evidence: explain field equivalence, derivation, proxy validity, scope, and uncertainty that matter
   to a reader. Do not substitute "I could not find" or another internal circumstance for an argument.
5. A new answer-determining claim outside the QG claim IDs is material path drift and requires abandonment.
6. Write the answer first. For list or set questions, state the complete set near the top and defend inclusion,
   exclusion, time basis, calculations, and completeness required by the question.
7. Manifest every material final-answer claim. Reuse every QG answer-determining claim ID. The system-provided
   additional-explanation ID requires an acquired window and may be used only for sourced explanatory facts.
8. Attach only returned `window_id` values to claims. `citation_window_ids` must be the exact union of every
   claim's `evidence_window_ids`; do not omit a claim window or select an unattached window.

FIELD INTERPRETATION
- `status`: `finalized` only when the answer is defensible under DECISION CRITERIA; otherwise `abandon`.
- `answer_disposition`: `unchanged` or `bounded_correction`; null on abandon.
- `proposed_short_answer`: reconstructed final short answer after any bounded correction.
- `reference_answer_text`: polished standalone answer for the reader, with no workflow narration.
- `claims`: complete material claim manifest. `evidence_window_ids` names acquired windows supporting each claim;
  it may be empty only for a QG claim that is a valid logical or mathematical derivation explained in
  `support_explanation`, never for the additional-explanation claim.
- `citation_window_ids`: unique acquired windows that should become final citations.
- `abandon_reason`: concise internal typed-stage explanation; it is not reader-facing answer prose.

GOOD EXAMPLE
The proposed answer uses two official operands. After reading both windows, you correct one transcribed value,
keep the same comparison, explain the arithmetic, reuse the QG claim IDs, and cite both acquired windows. This is
a bounded correction and the final prose simply gives and defends the answer.

BAD EXAMPLE
You replace a workplace metric with residence data, invent a new candidate universe, or say "the supplied report
does not show this, so I cannot answer" without searching for the relevant direct source. Those are path drift or
internal workflow narration, not a completed reference answer.

OUTPUT
Return every schema field. An abandon output must use null answer fields and empty claim/citation arrays. Return
only the native structured response; do not add prose or a Markdown fence."""


def semantic_support_prompt(
    reviewed_question: DomainTweakReviewedQuestion,
    reference_output: DomainTweakReferenceAnswerOutput,
    *,
    evidence_windows: tuple[dict[str, object], ...],
) -> str:
    return f"""WORK ORDER
Without searching or rewriting the answer, decide whether the supplied reference answer and acquired source
windows would sufficiently persuade a careful judge. This is an evidence-quality gate, not an oracle-truth test:
strong direct evidence may justify confidence; weaker evidence requires a defensible bridge whose premises are
themselves supported. Internal circumstances such as what an agent found or failed to find are never evidence.

QUESTION
{reviewed_question.question_packet.question}

CONCRETE QUESTION REQUIREMENTS
{_json(reviewed_question.form_review.question_requirements)}

REFERENCE ANSWER OUTPUT
{_json(reference_output)}

ACQUIRED EVIDENCE WINDOWS
{json.dumps(evidence_windows, ensure_ascii=False, indent=2)}

DECISION CRITERIA
Return `pass` only when every supplied requirement and every manifested reference claim is supported at the
relation strength it needs, and the answer contains no material factual assertion omitted from its claim manifest.
External factual support must come from acquired windows. A logical or mathematical derivation may use no window
when its premises are stated in the question or supported operands and the derivation is valid. Return `abandon`
for any unsupported requirement, unsupported claim, or unmanifested material claim.

PROCEDURE
1. Emit exactly one `requirement_finding` for every supplied requirement ID, with no additions or omissions.
2. Check candidate universe, field identity or equivalence, scope, time basis, cardinality, completeness, ranking,
   absence, calculations, and proxy defense at the strength demanded by each requirement.
3. Emit exactly one `claim_finding` for every reference claim ID, with no additions or omissions.
4. Name only acquired window IDs that actually support the finding. The source-backed required relations require
   at least one acquired window: direct fact, field equivalence, exhaustive set, exact cardinality, absence, and
   upper or lower bound. A derived calculation may use empty evidence only when its supported premises make the
   derivation valid; an `other` relation remains subject to the full semantic decision criteria.
5. List any answer-determining or explanatory factual assertion in the answer text that is absent from the claim
   manifest under `unmanifested_material_claims`.
6. Do not improve, rewrite, or complete the answer. A pass preserves its text byte-for-byte downstream.

FIELD INTERPRETATION
- `support_status`: whether the exact requirement or claim is adequately defended, not merely topically related.
- `evidence_window_ids`: acquired windows that carry the support; never use URLs or invented IDs.
- `explanation`: concise claim-to-evidence or derivation analysis.
- `unmanifested_material_claims`: material answer assertions with no claim ID; empty only when none exist.
- `status`: `pass` only when all findings are supported and the unmanifested list is empty.
- `abandon_reason`: concise combined reason for rejection; null on pass.

GOOD EXAMPLE
A comparison requirement is supported by two windows containing the exact operands, and the finding explains the
calculation. A proxy-based claim passes only when windows support both the proxy values and the asserted mapping.

BAD EXAMPLE
A finding passes because a source is authoritative or mentions the topic, even though it does not establish the
requested field, time period, denominator, completeness, or proxy relationship. Authority is not claim binding.

OUTPUT
Return every schema field through the native structured response. Do not add prose or a Markdown fence."""


def feedback_prompt(feedback: tuple[str, ...]) -> str:
    return (
        "Your previous native structured response failed deterministic schema or identity validation. "
        "Correct only the listed structural defects while preserving the work order:\n"
        + "\n".join(f"- {item}" for item in feedback)
    )


def _json(value: BaseModel | tuple[BaseModel, ...]) -> str:
    if isinstance(value, BaseModel):
        payload: object = value.model_dump(mode="json")
    else:
        payload = [item.model_dump(mode="json") for item in value]
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "feedback_prompt",
    "form_blueprint_prompt",
    "form_review_prompt",
    "phase_instruction",
    "question_generation_prompt",
    "reference_answer_prompt",
    "semantic_support_prompt",
]
