import json

from harnyx_commons.domain_tweak_generation import (
    AcceptedRouteContext,
    DossierAnswer,
    DossierFact,
    DossierRequirement,
    GenerationForm,
    PortfolioAllocation,
    SourceDossier,
)
from harnyx_commons.domain_tweak_generation.prompts import (
    AUDIT_SYSTEM,
    DOSSIER_SYSTEM,
    PORTFOLIO_SYSTEM,
    QUESTION_SYSTEM,
    REFERENCE_SYSTEM,
    audit_prompt,
    dossier_prompt,
    portfolio_prompt,
    question_prompt,
)
from harnyx_commons.domain_tweak_generation.source_workspace import _serialize_audit_packet


def test_portfolio_and_dossier_prompts_are_form_blind() -> None:
    """Future failure: discovery must not regain source-form or benchmark-answer leakage."""
    hidden_form = "SECRET FORM OPERATION"
    form = GenerationForm(form_identity="f", source_index=1, form=hidden_form)
    allocation = PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e"))
    dossier = SourceDossier(
        status="ready",
        subject="subject",
        route_summary="route",
        question_plan="plan",
        answers=(DossierAnswer(answer_id="A1", value="x"), DossierAnswer(answer_id="A2", value="y")),
        requirements=(DossierRequirement(description="requirement"),),
        source_facts=(DossierFact(statement="fact", evidence_ids=("E1",)),),
        derivation="derive",
    )

    assert hidden_form not in portfolio_prompt((0,))
    assert hidden_form not in dossier_prompt(allocation)
    assert hidden_form in question_prompt(form, allocation, dossier, ())


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
    for work_order in (PORTFOLIO_SYSTEM, DOSSIER_SYSTEM, QUESTION_SYSTEM, REFERENCE_SYSTEM, AUDIT_SYSTEM):
        assert "OUTPUT CONTRACT:" in work_order
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
    assert prompt == "Audit this closed proof packet:\n" + expected_json
