"""Luna solver work orders and the adapted BRI-1594 sole HIGH author."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .contracts import ResponseMode

PROMPT_ROOT = Path(__file__).with_name("prompts")
PROMPT_PROVENANCE = "BRI-1594-r1604-codex-high-single-author"
BASELINE_SHA256 = "d88c10c44e32ebd7069615ef53a23457e8fc431efbe50cdd123668e6760988a1"


def role_prompt(role: str, mode: str) -> str:
    if role in {"solver", "grounded_solver"}:
        name = "structured_solver_system_prompt" if mode == "structured" else "solver_system_prompt"
    else:
        raise ValueError("Construction roles have been replaced by the sole Codex author")
    return (PROMPT_ROOT / f"{name}.txt").read_text()


def author_prompt(mode: ResponseMode) -> str:
    common = (PROMPT_ROOT / "codex_author_system_prompt.txt").read_text()
    common = common.replace("{{reference_construction_instructions}}", reference_construction_instructions())
    if mode == "structured":
        common = common.replace(
            "question-draft.json has only question, answer, answer_components, scope, source_support, revision_note.",
            "The structured draft has question, answer, answer_components, scope, source_support, "
            "revision_note, output_schema_json and structured_answer_json.",
        )
        common = common.replace("Do not add schema fields.", "Do not add fields beyond the assigned draft contract.")
        common += """
ASSIGNED RESPONSE MODE: structured. In addition to the six base fields, supply output_schema_json and
structured_answer_json. The public schema must describe an object under JSON Schema Draft2020-12, use only
offline/local references, and state exactly the public requested fields. structured_answer_json must satisfy
this schema and express the same verified facts as the readable answer/components. Solvers receive the
complete public question and that schema. Factual assessment is separate from JSON/schema formatting. Do not
add private intermediate proof fields to the public schema. A good structured schema represents the requested facts
with explicit meaning; a bad one appends arbitrary fields merely to defeat a shortcut.
"""
    else:
        common += """
ASSIGNED RESPONSE MODE: plain_text. Use exactly the six QuestionDraft fields, without schema or structured-
answer fields.
"""
    return common


def reference_construction_instructions() -> str:
    return (PROMPT_ROOT / "codex_reference_instructions.txt").read_text().strip()


def author_prompt_sha256(mode: ResponseMode) -> str:
    return hashlib.sha256(author_prompt(mode).encode()).hexdigest()
