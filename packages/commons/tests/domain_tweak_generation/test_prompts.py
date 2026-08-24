from harnyx_commons.domain_tweak_generation import (
    AcceptedRouteContext,
    PortfolioAllocation,
)
from harnyx_commons.domain_tweak_generation.prompts import (
    portfolio_prompt,
    question_generation_prompt,
)


def test_portfolio_and_question_generation_prompts_have_no_source_form_boundary() -> None:
    """Future failure: discovery must not regain source-form or benchmark-answer leakage."""
    hidden_form = "SECRET FORM OPERATION"
    allocation = PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e"))

    assert hidden_form not in portfolio_prompt((0,))
    prompt = question_generation_prompt(allocation, "general_deep_research", "plain_text")
    assert hidden_form not in prompt
    assert "source_form" not in prompt


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
