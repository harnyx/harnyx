"""Frozen research prompts plus the existing source-first reference contract."""

from __future__ import annotations

from pathlib import Path

PROMPT_ROOT = Path(__file__).with_name("prompts")
PROMPT_PROVENANCE = "BRI-1528-structured-pipeline-handoff-20261003"


def role_prompt(role: str, mode: str) -> str:
    if role == "seed":
        name = "seed_prompt"
    elif role == "verifier":
        name = "structured_verifier_system_prompt"
    elif role == "solver":
        name = "structured_solver_system_prompt" if mode == "structured" else "solver_system_prompt"
    else:
        suffix = {"author": "generator", "reviewer": "reviewer", "analyst": "analyst"}[role]
        name = ("structured_" if mode == "structured" else "") + suffix + "_system_prompt"
    prompt = (PROMPT_ROOT / f"{name}.txt").read_text()
    prompt = prompt.replace("{construction_instructions}", (PROMPT_ROOT / "construction_instructions.txt").read_text())
    if role == "verifier" and mode == "plain_text":
        prompt = prompt.replace("structured research question", "research question")
        prompt = prompt.replace("and output schema", "")
        prompt = prompt.replace("question and schema given to solvers", "question given to solvers")
        prompt = prompt.replace(
            "structured answer, readable answer, components and scope", "answer, components and scope"
        )
        prompt = prompt.replace(
            "question and schema. The harness handles JSON/schema validation; you check meaning.",
            "question. You check factual meaning.",
        )
    return prompt


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
  resolves; never deduplicate, renumber, remap, collapse, or skip a position. Submit at least one registered evidence
  ID for every finalized reference. This requirement is independent of inline-pointer rules: an inline-pointer
  exemption never permits an empty citation_evidence_ids list.
- For plain prose, every material researched claim requires a valid `[[n]]` pointer unless the query explicitly
  rejects citations. Ordinary connective reasoning and genuinely trivial common knowledge need no pointer.
- For plain_text only, and only after correctness, requested coverage, exact source scope and snapshot, completeness,
  public citation grounding, calibrated uncertainty, and requested-form compliance are sound, shape presentation to
  the question when it calls for comparisons or inclusion/exclusion accounting. Align answer-determining before/after
  values; make every compared member, value, and direction directly checkable; separate in-scope results from
  out-of-scope items or decisive exclusions; and follow the question's sequence for multi-period included/excluded
  accounting.
- When no requested form conflicts, write clear, self-contained, reader-facing Markdown-style synthesis and use
  Markdown only when it lowers reader effort. Prefer synthesis over a raw provenance dump. Do not require a table or
  extra explanation. An explicit requested form such as XML or a terse answer always overrides this default
  presentation.
- Correctness, requested coverage, instruction following, evidence support, and calibrated uncertainty outrank
  presentation. Do not conceal a gap with polished formatting or unqualified certainty.
- For structured output, follow the exact public output_schema_json, including every description and constraint.
  Inline pointers are not required by default. Include them only when the query or a prose-capable field description
  explicitly requires citations, and then only in that prose-capable field. An atomic field such as an integer,
  number, boolean, enum, identifier, date token, or other non-explanatory value must not be polluted with citation
  syntax.
- Public answer text must not expose evidence IDs, proof step IDs, audit reasoning, private author annotations, or
  labels such as `Supports:` or `Claim:`. Citation excerpts are host-materialized raw source passages;
  start/end positions are extraction metadata.
  Never copy excerpts
  or private provenance prose into the public answer.

OUTPUT CONTRACT:
- status: finalized only when VFS evidence establishes every answer and load-bearing inference; otherwise giveup.
- answer_text: final public response for plain_text, preserving any explicit requested form exactly; null for
  structured and giveup.
- citation_evidence_ids: registered evidence IDs in exact public citation-position order. Duplicate an ID when the
  public citation array contains duplicate positions. Non-empty for every finalized reference. Empty only for giveup.
  Never invent an ID to fill an evidence gap.
- answers: select every known answer_id exactly once and in dossier order. Set corrected_value to null when the dossier
  value remains correct; author a non-empty corrected_value only when evidence changes that answer.
- proof_steps: ordered atomic proof. Unique step_id values; kind=supported requires registered evidence_ids;
  kind=derived requires only earlier depends_on_step_ids. scan_certificate_ids support only the bounded claim certified.
  Proof steps are private audit input and must not contain public citation-pointer syntax or private labels in
  answer_text.
- structured_answer_json: null for plain_text and giveup. For a finalized structured question, strict JSON encoding of
  the complete independently derived value under the dossier's exact fixed public output schema; do not copy the QG
  hypothesis. Citation markers may occur only under the structured-field rule above.
- note: optional public response-level explanation for finalized plain-text or structured answers. Actively decide
  whether a note is needed. Include it whenever the required answer alone is insufficient to explain why its decisive
  values and conclusions follow from the cited evidence and any necessary inference, including when the requested
  output format constrains the answer to atomic structured values or otherwise prevents that explanation. A reader
  should be able to understand from the note alone why the required answer is warranted. It may also qualify scope or
  correct a false premise. It cannot replace or repair answer_text/structured_answer_json, is not private reasoning,
  and is not a no-answer branch. Factual claims use `[[n]]` against the same exact ordered public
  citation_evidence_ids projection as the required answer; note is not evidence. Omit it when the required answer
  already explains itself and a note would only repeat it. Null for giveup.
- giveup_reason: concrete missing evidence or invalid inference for giveup; null for finalized.

GOOD: {"status":"finalized","answer_text":"## Result\\n\\nAlpha has the published value 12 [[1]].",
"citation_evidence_ids":["E1"],"answers":[{"answer_id":"A1","corrected_value":null}],"proof_steps":[{"step_id":"S1",
"statement":"The bounded row reports Alpha with value 12.","kind":"supported","evidence_ids":["E1"],
"depends_on_step_ids":[],"scan_certificate_ids":[]},{"step_id":"S2","statement":"Alpha is the maximum among the
established candidates.","kind":"derived","evidence_ids":[],"depends_on_step_ids":["S1"],
"scan_certificate_ids":[]}],"structured_answer_json":null,"note":null,"giveup_reason":null}
GOOD STRUCTURED: {"status":"finalized","answer_text":null,"citation_evidence_ids":["E1"],"answers":[{"answer_id":
"A1","corrected_value":null}],"proof_steps":[{"step_id":"S1","statement":"The bounded row reports value 12.",
"kind":"supported","evidence_ids":["E1"],"depends_on_step_ids":[],"scan_certificate_ids":[]}],
"structured_answer_json":"{\\"value\\":12}",
"note":"The cited row reports the requested value as 12, which is why the structured value field is 12 [[1]].",
"giveup_reason":null}
Why: the atomic structured value has no inline marker, while the separate public citation array retains its supporting
evidence position; the optional note makes the evidence-to-answer connection understandable without changing the
fixed output schema.
GOOD NOTE: beside `{"category":"Beta"}`, `note` may say "The premise names Alpha, but the cited classification places
the item in Beta; the structured answer uses that corrected category [[1]]." It concisely explains the correction,
answer, and evidence relationship in one public flow.
BAD NOTE: beside `{"value":12}`, `"note":"The answer is 12."` merely repeats the structured value and gives no
evidence-to-answer explanation. Omit the note instead.
BAD: {"status":"finalized","answer_text":"Claim: Alpha is probably 12.","citation_evidence_ids":[],
"answers":[{"answer_id":"new","corrected_value":"Alpha"}],"proof_steps":[],"structured_answer_json":null,
"note":"The answer is 12.","giveup_reason":null}
Why: it invents an answer ID, exposes a private label, omits the material claim's pointer and evidence position, hides
uncertainty behind unsupported prose, and supplies no proof."""
