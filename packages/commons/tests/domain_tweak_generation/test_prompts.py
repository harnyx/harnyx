import json

from harnyx_commons.domain_tweak_generation import (
    AcceptedRouteContext,
    GroundedQuestionDossier,
    PortfolioAllocation,
)
from harnyx_commons.domain_tweak_generation.prompts import (
    AUDIT_SYSTEM,
    PORTFOLIO_SYSTEM,
    QUESTION_GENERATION_SYSTEM,
    REFERENCE_SYSTEM,
    audit_prompt,
    portfolio_prompt,
    question_generation_prompt,
)
from harnyx_commons.domain_tweak_generation.source_workspace import _serialize_audit_packet


def test_portfolio_and_question_generation_prompts_have_no_source_form_boundary() -> None:
    """Future failure: discovery must not regain source-form or benchmark-answer leakage."""
    hidden_form = "SECRET FORM OPERATION"
    allocation = PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e"))

    assert hidden_form not in portfolio_prompt((0,))
    assert hidden_form not in question_generation_prompt(allocation)
    assert "source_form" not in question_generation_prompt(allocation)


def test_portfolio_prompt_carries_only_bounded_prior_route_context() -> None:
    """Future failure: refill diversity must remain request-local and payload bounded."""
    prompt = portfolio_prompt(
        (1,),
        accepted_route_contexts=(
            AcceptedRouteContext(
                subject="Subject",
                route_summary="Join the annual index to the published table",
                source_urls=("https://example.com/a", "https://example.org/b"),
            ),
        ),
    )

    assert "Join the annual index" in prompt
    assert "https://example.com/a" in prompt
    assert "answer_id" not in prompt


def test_every_llm_work_order_interprets_its_output_contract_and_examples() -> None:
    """Future failure: JSON Schema field names alone must not define stage semantics."""
    for work_order in (PORTFOLIO_SYSTEM, QUESTION_GENERATION_SYSTEM, REFERENCE_SYSTEM, AUDIT_SYSTEM):
        assert "OUTPUT CONTRACT" in work_order
        assert "GOOD:" in work_order
        assert "BAD:" in work_order
    assert "question itself reveals an answer" in " ".join(AUDIT_SYSTEM.split())


def test_audit_prompt_reuses_the_bounded_packet_serializer_without_format_drift() -> None:
    """Future failure: packet budgeting and the actual audit prompt must serialize identically."""
    packet = {
        "question": "Which value?",
        "canonical_short_answers": ["Alpha"],
        "proof_steps": [],
        "selected_evidence": [],
        "scan_certificates": [],
    }
    expected_json = json.dumps(packet, ensure_ascii=False, indent=2)

    prompt = audit_prompt(packet)

    assert _serialize_audit_packet(packet) == expected_json
    assert prompt == (
        "Audit this proof packet and independently inspect the retained sources where needed:\n" + expected_json
    )


def test_question_generation_work_order_preserves_ultra_agent_responsibility() -> None:
    """Future failure: final wording and source proof must not be split back into separate agents."""
    normalized = " ".join(QUESTION_GENERATION_SYSTEM.split())
    assert "Own discovery, inspection, the positive answer route, and wording" in normalized
    assert "inspect list_source_links; use regex_search before bounded read_lines" in normalized
    assert "publisher and version when relevant" in normalized
    assert "why_not_one_page" in QUESTION_GENERATION_SYSTEM
    assert "substantive_final_condition" in QUESTION_GENERATION_SYSTEM
    assert set(GroundedQuestionDossier.model_fields) >= {"question", "answers", "derivation", "source_facts"}
